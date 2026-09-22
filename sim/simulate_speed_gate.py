#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
星闪(类蓝牙CS)测距速度卡限算法 - 仿真脚本 (v2: 修正后的相位差/标准差公式)

复现"设计报告"中描述的完整链路 —— 与用户二次确认后的现网实现一致:
  1) 多频点(信道)IQ信号模型
  2) 分别对两次测量的原始IQ取相位: theta1(k)=angle(IQ1(k)), theta2(k)=angle(IQ2(k))
  3) 相位直接相减(而非复数相减)并wrap到(-pi,pi]:
        dphi(k) = wrap( theta2(k) - theta1(k) )
  4) 用 dphi(k) 序列(按频点索引k排列)做 DFT/IFFT 峰值搜索, 拟合相位斜率,
     换算为两次测量间的位移 Delta_d_hat, 进而得到速度估计 v_hat = Delta_d_hat/dt
  5) 用相位差重建单位复数 IQ_diff(k) = exp(j*dphi(k)), 以其"复数标准差"
     (到均值向量的均方根偏离) 作为动/静判据:
        std = sqrt( mean_k( |IQ_diff(k) - mean_k(IQ_diff(k))|^2 ) )
     该 std 无量纲, 取值范围 [0, 1] (因 |IQ_diff(k)|=1): 全部同向(静止/低噪声)时
     接近 0; 均匀散布(强噪声或大幅相位跳变)时趋于 1。
  6) 两档限幅:
        档位1(低速): std < STD_TH_LOW                     -> range = |v_hat| * dt
        档位2(静止): std < STD_TH_STATIC or |v_hat| < V_TH -> range = 0.1 * |v_hat| * dt
        否则: 不限幅, 直接采用原始测距值

本脚本所有物理/协议层参数 (频点数, 频点间隔, K系数, 测量间隔dt, 噪声模型)
均为仿真占位假设, 已在代码中以 "TODO(实测)" 标出, 需要用实际星闪 CS 协议参数
与实测 IQ 数据重新标定, 详见报告正文"风险与待确认事项"一节。

运行:
    python3 simulate_speed_gate.py
输出:
    ./results/*.png  仿真图
    ./results/summary.json  关键数值结果, 供报告引用
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# 0. 配色 (取自公司数据可视化调色板, 保证色盲友好 / 明暗一致)
# --------------------------------------------------------------------------
C_BLUE = "#2a78d6"
C_ORANGE = "#eb6834"
C_AQUA = "#1baf7a"
C_YELLOW = "#eda100"
C_VIOLET = "#4a3aa7"
C_RED = "#e34948"
C_GOOD = "#0ca30c"
C_CRIT = "#d03b3b"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "grid.color": GRID,
        "font.size": 11,
        "axes.grid": True,
        "grid.linewidth": 0.8,
        "font.family": "WenQuanYi Zen Hei",
        "axes.unicode_minus": False,
    }
)

RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULT_DIR, exist_ok=True)

rng = np.random.default_rng(20260921)

# --------------------------------------------------------------------------
# 1. 仿真参数
#    以下已由需求方确认为真实值(2026-09-22): K, Delta_f/N_F, Delta_t, 扫频频段,
#    工作距离范围, DFT测速的搜索范围/分辨率, 三级速度定义。
#    单频点 SNR 现网无上报, 只有 RSSI 上报(已确认: 1m处约-60dBm, 20m处约-80dBm),
#    需要结合接收机噪底(热噪声+噪声系数NF)才能换算成SNR —— NF 是本仿真最后一个
#    占位假设 TODO(实测/查芯片规格书确认NF)。基线测距(d_raw)野值/抖动特性仍为
#    占位假设 —— TODO(实测)
# --------------------------------------------------------------------------
C_LIGHT = 3e8          # 光速 m/s
K_PATH = 2             # 已确认: 双程(K=2)
F_START = 2400e6       # 扫频起始频率 Hz (已确认)
F_STOP = 2479e6        # 扫频终止频率 Hz (已确认, 2.4G ISM频段)
DELTA_F = 1e6          # 频点间隔 Hz (已确认: 1MHz)
N_F = int(round((F_STOP - F_START) / DELTA_F)) + 1  # 已确认: 80个频点全用
DELTA_T = 0.2          # 相邻两次测量的时间间隔 s (已确认: 200ms, 5Hz测量率)
NFFT = 4096            # DFT/IFFT 补零点数, 用于在限定搜索范围内插值定位峰值
V_SEARCH_MAX = 5.0     # DFT测速的搜索范围 (已确认: +-5 m/s)
V_RESOLUTION = 0.05    # DFT测速的速度分辨率 (已确认: 0.05 m/s), 输出按此量化
D0_RANGE = (1.0, 50.0) # 仿真使用的典型工作距离范围, m (已确认工作距离0~100m+,
                       # 占位地取中段作为默认场景; 该范围在修正后的公式下对
                       # std/v_hat 数值基本无影响, 见4.3节循环不变性)

# --- RSSI -> SNR 换算(已确认RSSI实测点, NF为占位假设) ---
RSSI_AT_1M_DBM = -60.0     # 已确认: 1m处RSSI约-60dBm
RSSI_AT_20M_DBM = -80.0    # 已确认: 20m处RSSI约-80dBm
RX_NOISE_FIGURE_DB = 7.0   # 接收机噪声系数(NF)占位假设(典型2.4GHz低功耗射频芯片
                           # 量级4~10dB, 取中间值)                    TODO(实测/查规格书)
THERMAL_NOISE_DBM = -174.0 + 10 * np.log10(DELTA_F)  # 1MHz有效带宽热噪声基准
NOISE_FLOOR_DBM = THERMAL_NOISE_DBM + RX_NOISE_FIGURE_DB
# 由两个已确认RSSI点拟合的对数距离路径损耗指数(自由空间n=2, 此处<2可能反映近地反射
# 增强等实际环境效应, 仅基于2个点外推, 100m处外推结果需实测校验)
_PATH_LOSS_EXP = (RSSI_AT_1M_DBM - RSSI_AT_20M_DBM) / (10 * np.log10(20.0 / 1.0))


def snr_from_distance(d_m):
    """由已确认的RSSI实测点(1m/-60dBm, 20m/-80dBm)对数距离外推, 结合占位噪声系数
    换算出给定工作距离下的单频点SNR(dB)。"""
    rssi_dbm = RSSI_AT_1M_DBM - 10 * _PATH_LOSS_EXP * np.log10(d_m / 1.0)
    return rssi_dbm - NOISE_FLOOR_DBM

# 两档限幅阈值 (现网取值, 与用户确认)
STD_TH_LOW = 0.6       # 低速档标准差门限 (无量纲, 见模块说明)
STD_TH_STATIC = 0.3    # 静止档标准差门限 (无量纲)
V_TH_STATIC = 0.1      # m/s, 静止档速度门限
STATIC_RANGE_SCALE = 0.1  # 静止档限幅收紧系数


# --------------------------------------------------------------------------
# 2. 信号 / 测量模型
# --------------------------------------------------------------------------
def true_phase(d, n_f=N_F, delta_f=DELTA_F, k_path=K_PATH):
    """给定真实距离 d, 返回 N_F 个频点上的理想(无噪)相位 theta(k)."""
    freqs = np.arange(n_f) * delta_f
    return -2.0 * np.pi * k_path * freqs * d / C_LIGHT


def gen_iq(d, snr_db, n_f=N_F, generator=rng):
    """给定真实距离与每频点SNR(dB), 生成含噪 IQ 序列 (幅度归一化为1)."""
    theta = true_phase(d, n_f=n_f)
    signal = np.exp(1j * theta)
    snr_lin = 10 ** (snr_db / 10.0)
    noise_std = 1.0 / np.sqrt(2.0 * snr_lin)  # 每个I/Q分量的噪声标准差
    noise = generator.normal(0, noise_std, n_f) + 1j * generator.normal(0, noise_std, n_f)
    return signal + noise


def wrap_to_pi(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def phase_diff(iq_prev, iq_curr):
    """现网实现(二次确认后): 分别取相位再相减, 并wrap到(-pi,pi].
    Delta_phi(k) = wrap( angle(IQ2(k)) - angle(IQ1(k)) )
    """
    return wrap_to_pi(np.angle(iq_curr) - np.angle(iq_prev))


def circular_std(dphi):
    """用相位差重建单位复数 IQ_diff(k)=exp(j*dphi(k)), 求其到均值向量的
    均方根偏离, 作为"复数标准差"(无量纲, 取值范围[0,1])."""
    z = np.exp(1j * dphi)
    mean_vec = np.mean(z)
    return float(np.sqrt(np.mean(np.abs(z - mean_vec) ** 2)))


def dft_slope_to_speed(dphi, delta_f=DELTA_F, delta_t=DELTA_T, k_path=K_PATH, nfft=NFFT,
                        v_search_max=V_SEARCH_MAX, v_resolution=V_RESOLUTION):
    """对 dphi(k) 做 DFT(補零), 在已确认的 +-5m/s 搜索范围内取幅度谱峰值位置,
    换算出速度估计, 并按已确认的 0.05m/s 分辨率量化输出 —— 与现网DFT测速的
    搜索范围/分辨率设定对齐, 而非在整个(未限幅的)DFT不模糊范围内搜索峰值。"""
    s = np.exp(1j * dphi)
    spec = np.fft.fftshift(np.fft.fft(s, n=nfft))
    freq_cyc_per_sample = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0))  # cycles/sample, range (-0.5,0.5]
    v_grid = -freq_cyc_per_sample * C_LIGHT / (delta_f * k_path) / delta_t
    mask = np.abs(v_grid) <= v_search_max
    mag = np.abs(spec)
    local_idx = np.argmax(mag[mask])
    v_hat_raw = v_grid[mask][local_idx]
    v_hat = float(np.round(v_hat_raw / v_resolution) * v_resolution)
    delta_d_hat = v_hat * delta_t
    return v_hat, delta_d_hat


def measure_round_pair(d_prev, d_curr, snr_db, generator=rng):
    """完整走一遍: 生成两轮IQ -> 相位差 -> DFT测速 -> 复数标准差."""
    iq1 = gen_iq(d_prev, snr_db, generator=generator)
    iq2 = gen_iq(d_curr, snr_db, generator=generator)
    dphi = phase_diff(iq1, iq2)
    std_dphi = circular_std(dphi)
    v_hat, _ = dft_slope_to_speed(dphi)
    return v_hat, std_dphi, dphi


def classify_state(std_dphi, v_hat):
    """两档静止判据. 返回 'static' / 'low_speed' / 'moving'."""
    if std_dphi < STD_TH_STATIC or abs(v_hat) < V_TH_STATIC:
        return "static"
    if std_dphi < STD_TH_LOW:
        return "low_speed"
    return "moving"


def clamp_range(state, v_hat, delta_t=DELTA_T):
    """依据状态返回限幅半宽度; 'moving' 返回 None 表示不限幅."""
    if state == "static":
        return STATIC_RANGE_SCALE * abs(v_hat) * delta_t
    if state == "low_speed":
        return abs(v_hat) * delta_t
    return None


# --------------------------------------------------------------------------
# 实验0: 静止状态下, 复数标准差 std 随 SNR 的变化
# --------------------------------------------------------------------------
def experiment_0_std_vs_snr():
    """v_true=0 (严格静止), 扫描单频点SNR, 观察 std 的变化. 用修正后的
    "相位相减+复数标准差"公式, std 应随SNR提升而单调下降(噪声主导), 不再像
    旧的"复数直接相减"公式那样退化为与SNR无关的常数 —— 用于验证公式修正的效果,
    并标注当前0.3/0.6两个阈值对应的隐含SNR工作点。
    """
    snr_grid_db = np.arange(-5, 41, 2.5)
    n_trials = 500

    std_mean, std_p10, std_p90 = [], [], []
    for snr in snr_grid_db:
        vals = []
        for _ in range(n_trials):
            d0 = rng.uniform(*D0_RANGE)
            _, s, _ = measure_round_pair(d0, d0, snr)
            vals.append(s)
        vals = np.array(vals)
        std_mean.append(vals.mean())
        std_p10.append(np.percentile(vals, 10))
        std_p90.append(np.percentile(vals, 90))
    std_mean = np.array(std_mean)

    snr_at_static_th = float(np.interp(STD_TH_STATIC, std_mean[::-1], snr_grid_db[::-1]))
    snr_at_low_th = float(np.interp(STD_TH_LOW, std_mean[::-1], snr_grid_db[::-1]))

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(snr_grid_db, std_mean, color=C_BLUE, linewidth=2, marker="o", markersize=3.5,
            label="std 均值 (严格静止 v_true=0)")
    ax.fill_between(snr_grid_db, std_p10, std_p90, color=C_BLUE, alpha=0.15, linewidth=0, label="10~90百分位")
    ax.axhline(STD_TH_STATIC, color=C_GOOD, linewidth=1.2, linestyle=":", label=f"静止门限 {STD_TH_STATIC}")
    ax.axhline(STD_TH_LOW, color=C_CRIT, linewidth=1.0, linestyle=":", label=f"低速门限 {STD_TH_LOW}")
    ax.axvline(snr_at_static_th, color=C_GOOD, linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axvline(snr_at_low_th, color=C_CRIT, linewidth=0.8, linestyle="--", alpha=0.7)
    ax.annotate(f"≈{snr_at_static_th:.1f}dB", (snr_at_static_th, 0.02), fontsize=8, color=C_GOOD, ha="center")
    ax.annotate(f"≈{snr_at_low_th:.1f}dB", (snr_at_low_th, 0.02), fontsize=8, color=C_CRIT, ha="center")
    ax.set_xlabel("单频点 SNR (dB)")
    ax.set_ylabel("std (无量纲, [0,1])")
    ax.set_title("实验0: 修正公式后, 严格静止目标的 std 随SNR单调下降\n(标注当前两档阈值对应的隐含SNR工作点)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "exp0_std_vs_snr.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {"snr_at_static_threshold_db": snr_at_static_th, "snr_at_low_speed_threshold_db": snr_at_low_th}


# --------------------------------------------------------------------------
# 实验0b: std 对运动量(位移 Delta_d = v*dt)的响应, 与达到阈值所需的运动量
# --------------------------------------------------------------------------
def experiment_0b_std_vs_motion():
    """固定SNR(取三档代表值), 扫描相邻两轮间的位移 Delta_d (对数坐标, 从亚毫米
    级到数十米级), 观察 std 何时开始明显抬升、何时越过两档阈值 —— 用于回答
    "在当前假设的Δt/扫频带宽下, 需要多大的运动量才能被std判据感知到"。
    """
    snr_list = [10, 20, 30]
    colors = {10: C_ORANGE, 20: C_BLUE, 30: C_AQUA}
    delta_d_grid = np.logspace(-4, 1.5, 36)  # 0.0001 m ~ 31.6 m
    n_trials = 300

    results = {}
    for snr in snr_list:
        means = []
        for dd in delta_d_grid:
            vals = []
            for _ in range(n_trials):
                d0 = rng.uniform(*D0_RANGE)
                _, s, _ = measure_round_pair(d0, d0 + dd, snr)
                vals.append(s)
            means.append(np.mean(vals))
        results[snr] = np.array(means)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for snr in snr_list:
        ax.plot(delta_d_grid, results[snr], color=colors[snr], linewidth=2, label=f"SNR={snr}dB")
    ax.axhline(STD_TH_STATIC, color=C_GOOD, linewidth=1.2, linestyle=":", label=f"静止门限 {STD_TH_STATIC}")
    ax.axhline(STD_TH_LOW, color=C_CRIT, linewidth=1.0, linestyle=":", label=f"低速门限 {STD_TH_LOW}")
    v_walk_max = 2.0 * DELTA_T
    v_fast_max = 5.0 * DELTA_T
    ax.axvline(v_walk_max, color=MUTED, linewidth=1.2, linestyle="--",
               label=f"步行上限2m/s×Δt={v_walk_max*1000:.0f}mm")
    ax.axvline(v_fast_max, color=INK, linewidth=1.2, linestyle="--",
               label=f"快速上限5m/s×Δt={v_fast_max*1000:.0f}mm")
    ax.set_xscale("log")
    ax.set_xlabel("相邻两轮间的位移 Δd = v·Δt (m, 对数坐标)")
    ax.set_ylabel("std (无量纲, [0,1])")
    ax.set_title(f"实验0b: std 随单轮位移 Δd 的响应曲线\n(标注已确认Δt={DELTA_T*1000:.0f}ms下, 步行/快速上限对应的Δd)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "exp0b_std_vs_motion.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    dd_needed = {}
    for snr in snr_list:
        dd_needed[snr] = float(np.interp(STD_TH_LOW, results[snr], delta_d_grid))
    return {"delta_d_needed_for_low_speed_threshold_m": dd_needed,
            "walk_max_delta_d_m": float(v_walk_max),
            "fast_max_delta_d_m": float(v_fast_max)}


# --------------------------------------------------------------------------
# 实验1: v_hat 与 std 随真实速度 / SNR 的变化关系
# --------------------------------------------------------------------------
def experiment_1_speed_response():
    n_trials = 300

    # (a)(b) 已确认的现实速度范围: 静止~慢速~步行~快速运动, 覆盖到DFT测速的
    #        确认搜索边界 +-5 m/s
    v_true_grid_realistic = np.linspace(-V_SEARCH_MAX, V_SEARCH_MAX, 41)
    snr_list = [10, 20, 30]
    colors = {10: C_ORANGE, 20: C_BLUE, 30: C_AQUA}

    res_realistic = {snr: {"v_mean": [], "v_std": [], "std_mean": []} for snr in snr_list}
    for snr in snr_list:
        for v_true in v_true_grid_realistic:
            v_hats, stds = [], []
            for _ in range(n_trials):
                d_prev = rng.uniform(*D0_RANGE)
                d_curr = d_prev + v_true * DELTA_T
                v_hat, std_dphi, _ = measure_round_pair(d_prev, d_curr, snr)
                v_hats.append(v_hat)
                stds.append(std_dphi)
            res_realistic[snr]["v_mean"].append(np.mean(v_hats))
            res_realistic[snr]["v_std"].append(np.std(v_hats))
            res_realistic[snr]["std_mean"].append(np.mean(stds))

    # (c) 探查DFT测速在已确认的 +-5m/s 搜索边界附近/之外的行为(超出搜索范围时,
    #     搜索被限定在窗口内, 无法跟踪真实值, 只能看到窗口内的伪峰) —— 高SNR下
    snr_wide = 40
    v_true_grid_edge = np.linspace(0, 8.0, 33)  # 覆盖到超出确认搜索边界60%
    v_hat_edge_mean, v_hat_edge_std = [], []
    for v_true in v_true_grid_edge:
        v_hats = []
        for _ in range(n_trials):
            d_prev = rng.uniform(*D0_RANGE)
            d_curr = d_prev + v_true * DELTA_T
            v_hat, _, _ = measure_round_pair(d_prev, d_curr, snr_wide)
            v_hats.append(v_hat)
        v_hat_edge_mean.append(np.mean(v_hats))
        v_hat_edge_std.append(np.std(v_hats))
    v_hat_edge_mean = np.array(v_hat_edge_mean)
    v_hat_edge_std = np.array(v_hat_edge_std)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

    ax = axes[0]
    ax.plot([-V_SEARCH_MAX, V_SEARCH_MAX], [-V_SEARCH_MAX, V_SEARCH_MAX], color=MUTED, linewidth=1.2,
            linestyle="--", label="理想 v_hat=v_true")
    for snr in snr_list:
        r = res_realistic[snr]
        ax.plot(v_true_grid_realistic, r["v_mean"], color=colors[snr], linewidth=2, label=f"SNR={snr}dB")
    ax.set_xlabel("真实速度 v_true (m/s)")
    ax.set_ylabel("v_hat (m/s)")
    ax.set_title(f"(a) 确认速度范围(±{V_SEARCH_MAX:.0f}m/s搜索窗): v_hat vs v_true\n(DFT相干增益下均值可跟踪, 单次噪声随SNR下降而增大)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    for snr in snr_list:
        r = res_realistic[snr]
        ax.plot(v_true_grid_realistic, r["std_mean"], color=colors[snr], linewidth=2, label=f"SNR={snr}dB")
    ax.axhline(STD_TH_STATIC, color=C_GOOD, linewidth=1.2, linestyle=":", label=f"静止门限 {STD_TH_STATIC}")
    ax.axhline(STD_TH_LOW, color=C_CRIT, linewidth=1.0, linestyle=":", label=f"低速门限 {STD_TH_LOW}")
    ax.set_xlabel("真实速度 v_true (m/s)")
    ax.set_ylabel("std (无量纲)")
    ax.set_title("(b) 确认速度范围: std vs v_true\n(仅在接近±5m/s边界时越过静止门限)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    ax.plot([0, V_SEARCH_MAX], [0, V_SEARCH_MAX], color=MUTED, linewidth=1.2, linestyle="--",
            label="理想 v_hat=v_true")
    ax.axvline(V_SEARCH_MAX, color=C_RED, linewidth=1.2, linestyle=":", label=f"确认搜索边界 {V_SEARCH_MAX:.0f}m/s")
    ax.plot(v_true_grid_edge, v_hat_edge_mean, color=C_VIOLET, linewidth=2, marker="o", markersize=4,
            label=f"v_hat 均值 (SNR={snr_wide}dB)")
    ax.fill_between(v_true_grid_edge, v_hat_edge_mean - v_hat_edge_std, v_hat_edge_mean + v_hat_edge_std,
                     color=C_VIOLET, alpha=0.15, linewidth=0)
    ax.set_xlabel("真实速度 v_true (m/s, 含超出确认搜索边界的场景)")
    ax.set_ylabel("v_hat (m/s)")
    ax.set_title(f"(c) 搜索边界附近/之外的行为 (SNR={snr_wide}dB)\n(超过±5m/s后v_hat不再跟踪, 被限定在搜索窗内)")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("实验1: 修正公式 + 已确认DFT搜索范围(±5m/s,0.05m/s分辨率)后的速度响应特性", y=1.03, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "exp1_speed_response.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {
        "realistic_v_true_grid": v_true_grid_realistic.tolist(),
        "snr_list": snr_list,
        "std_mean_at_snr20_v0": float(res_realistic[20]["std_mean"][20]),
        "std_mean_at_snr20_vmax": float(res_realistic[20]["std_mean"][-1]),
    }


# --------------------------------------------------------------------------
# 实验2: 三种运动状态下 std 的分布 (箱线图)
# --------------------------------------------------------------------------
def experiment_2_std_distribution():
    snr_db = 20
    n_trials = 1000

    # 与需求方确认的三级速度定义对齐: 慢速<0.3, 步行<2, 快速<5 m/s
    states = {
        "静止\n(v=0)": (0.0, 0.0),
        "慢速\n(0.05~0.3 m/s)": (0.05, 0.3),
        "步行\n(0.3~2 m/s)": (0.3, 2.0),
        "快速\n(2~5 m/s)": (2.0, 5.0),
    }
    box_colors = [C_AQUA, C_YELLOW, C_ORANGE, C_RED]

    data = {}
    for label, (vlo, vhi) in states.items():
        vals = []
        for _ in range(n_trials):
            v_true = 0.0 if vlo == vhi == 0.0 else rng.uniform(vlo, vhi) * rng.choice([-1, 1])
            d_prev = rng.uniform(*D0_RANGE)
            d_curr = d_prev + v_true * DELTA_T
            _, std_dphi, _ = measure_round_pair(d_prev, d_curr, snr_db)
            vals.append(std_dphi)
        data[label] = np.array(vals)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    positions = np.arange(len(data))
    bp = ax.boxplot(list(data.values()), positions=positions, widths=0.5, patch_artist=True,
                     showfliers=False, medianprops=dict(color=INK, linewidth=1.5))
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor(color)
    ax.axhline(STD_TH_STATIC, color=C_GOOD, linewidth=1.2, linestyle=":", label=f"静止门限 {STD_TH_STATIC}")
    ax.axhline(STD_TH_LOW, color=C_CRIT, linewidth=1.2, linestyle=":", label=f"低速门限 {STD_TH_LOW}")
    ax.set_xticks(positions)
    ax.set_xticklabels(list(data.keys()))
    ax.set_ylabel("std (无量纲)")
    ax.set_title(f"实验2: 四级运动状态下 std 分布 (SNR={snr_db}dB, N={n_trials}次/组, 已确认速度分级)\n(步行/快速已明显抬升, 但静止/慢速仍高度重叠 —— 见实验0b成因分析)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "exp2_std_distribution.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {k: {"mean": float(v.mean()), "std": float(v.std()), "p10": float(np.percentile(v, 10)),
                "p90": float(np.percentile(v, 90))} for k, v in data.items()}


# --------------------------------------------------------------------------
# 实验3: 两档判据的分类性能 (混淆矩阵 + 阈值扫描)
# --------------------------------------------------------------------------
def experiment_3_classifier_performance():
    snr_db = 20
    n_trials = 600

    # 与需求方确认的三级速度定义对齐: 慢速<0.3, 步行<2, 快速<5 m/s
    truth_states = {"static": (0.0, 0.0), "slow": (0.05, 0.3), "walk": (0.3, 2.0), "fast": (2.0, 5.0)}
    order = ["static", "slow", "walk", "fast"]

    confusion = {t: {p: 0 for p in ["static", "low_speed", "moving"]} for t in order}
    for truth, (vlo, vhi) in truth_states.items():
        for _ in range(n_trials):
            v_true = 0.0 if vlo == vhi == 0.0 else rng.uniform(vlo, vhi) * rng.choice([-1, 1])
            d_prev = rng.uniform(*D0_RANGE)
            d_curr = d_prev + v_true * DELTA_T
            v_hat, std_dphi, _ = measure_round_pair(d_prev, d_curr, snr_db)
            pred = classify_state(std_dphi, v_hat)
            confusion[truth][pred] += 1

    std_th_grid = np.linspace(0.02, 1.0, 50)
    static_stds, fast_stds = [], []
    for _ in range(n_trials):
        d_prev = rng.uniform(*D0_RANGE)
        _, s, _ = measure_round_pair(d_prev, d_prev, snr_db)
        static_stds.append(s)
    for _ in range(n_trials):
        v_true = rng.uniform(2.0, 5.0) * rng.choice([-1, 1])
        d_prev = rng.uniform(*D0_RANGE)
        d_curr = d_prev + v_true * DELTA_T
        _, s, _ = measure_round_pair(d_prev, d_curr, snr_db)
        fast_stds.append(s)
    static_stds, fast_stds = np.array(static_stds), np.array(fast_stds)
    det_rate = [float(np.mean(static_stds < th)) for th in std_th_grid]
    far_rate = [float(np.mean(fast_stds < th)) for th in std_th_grid]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))

    ax = axes[0]
    pred_order = ["static", "low_speed", "moving"]
    mat = np.array([[confusion[t][p] for p in pred_order] for t in order], dtype=float)
    mat_norm = mat / mat.sum(axis=1, keepdims=True)
    ax.imshow(mat_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(pred_order)))
    ax.set_yticks(range(len(order)))
    label_map = {"static": "判静止", "low_speed": "判低速", "moving": "判运动"}
    label_map_t = {"static": "真实静止(v=0)", "slow": "真实慢速(<0.3)", "walk": "真实步行(0.3~2)",
                   "fast": "真实快速(2~5)"}
    ax.set_xticklabels([label_map[p] for p in pred_order])
    ax.set_yticklabels([label_map_t[t] for t in order])
    for i in range(len(order)):
        for j in range(len(pred_order)):
            ax.text(j, i, f"{mat_norm[i, j]*100:.0f}%\n({int(mat[i, j])})", ha="center", va="center",
                     color=INK if mat_norm[i, j] < 0.6 else "white", fontsize=9)
    ax.set_title(f"(a) 混淆矩阵 (SNR={snr_db}dB, 现网阈值 0.6/0.3/0.1,\n已确认速度分级)")

    ax = axes[1]
    ax.plot(std_th_grid, det_rate, color=C_BLUE, linewidth=2, label="真实静止 -> std<阈值 比例 (检出率)")
    ax.plot(std_th_grid, far_rate, color=C_RED, linewidth=2, label="真实快速(2~5m/s) -> std<阈值 比例 (虚警率)")
    ax.axvline(STD_TH_STATIC, color=C_GOOD, linewidth=1.2, linestyle=":", label=f"现网静止门限 {STD_TH_STATIC}")
    ax.set_xlabel("std 判决门限")
    ax.set_ylabel("比例")
    ax.set_title("(b) 静止门限扫描: 检出率 / 虚警率")
    ax.legend(frameon=False, fontsize=8.5, loc="center right")

    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "exp3_classifier_performance.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    det_at_th = float(np.mean(static_stds < STD_TH_STATIC))
    far_at_th = float(np.mean(fast_stds < STD_TH_STATIC))
    return {"confusion": {t: confusion[t] for t in order},
            "static_detect_rate_at_threshold": det_at_th,
            "fast_false_alarm_rate_at_threshold": far_at_th}


# --------------------------------------------------------------------------
# 实验4/5: 限幅前后测距输出轨迹对比
# --------------------------------------------------------------------------
def simulate_baseline_distance(d_true_seq, jitter_std=0.03, glitch_prob=0.06, glitch_scale=0.6, generator=rng):
    """基线测距模块的仿真替身: 在真实距离上叠加正常抖动 + 偶发多径野值.
    该模块本身不在本报告设计范围内, 此处仅为演示限幅效果构造的简化黑盒模型。
    """
    n = len(d_true_seq)
    d_raw = d_true_seq + generator.normal(0, jitter_std, n)
    glitch_mask = generator.random(n) < glitch_prob
    d_raw[glitch_mask] += generator.normal(0, glitch_scale, glitch_mask.sum())
    return d_raw


def apply_speed_gate(d_raw, v_hat_seq, std_seq, delta_t=DELTA_T):
    n = len(d_raw)
    d_out = np.zeros(n)
    d_out[0] = d_raw[0]
    states = []
    for i in range(1, n):
        state = classify_state(std_seq[i], v_hat_seq[i])
        states.append(state)
        rng_half = clamp_range(state, v_hat_seq[i], delta_t)
        if rng_half is None:
            d_out[i] = d_raw[i]
        else:
            lo, hi = d_out[i - 1] - rng_half, d_out[i - 1] + rng_half
            d_out[i] = min(max(d_raw[i], lo), hi)
    return d_out, states


def build_scenario(kind, n_rounds=60, snr_db=20):
    t = np.arange(n_rounds) * DELTA_T
    if kind == "static":
        d_true = np.full(n_rounds, 3.0) + rng.normal(0, 0.001, n_rounds)  # mm级静态微抖动
    elif kind == "walk":
        d_true = 3.0 + 1.2 * t  # 恒速 1.2 m/s, 典型步行速度("步行"档 0.3~2m/s 内)
    elif kind == "fast":
        d_true = 3.0 + 3.0 * t  # 恒速 3.0 m/s, "快速"档(2~5m/s)代表值
    elif kind == "transition":
        d_true = np.empty(n_rounds)
        seg1, seg2 = n_rounds // 3, 2 * n_rounds // 3
        d_true[:seg1] = 3.0
        ramp_len = seg2 - seg1
        d_true[seg1:seg2] = 3.0 + 0.5 * (t[:ramp_len] - t[0])
        d_true[seg2:] = d_true[seg2 - 1]
    else:
        raise ValueError(kind)

    v_hat_seq = np.zeros(n_rounds)
    std_seq = np.zeros(n_rounds)
    for i in range(1, n_rounds):
        v_hat, std_dphi, _ = measure_round_pair(d_true[i - 1], d_true[i], snr_db)
        v_hat_seq[i] = v_hat
        std_seq[i] = std_dphi

    d_raw = simulate_baseline_distance(d_true)
    d_out, states = apply_speed_gate(d_raw, v_hat_seq, std_seq)
    return t, d_true, d_raw, d_out, v_hat_seq, std_seq, states


def experiment_4_gating_traces():
    scenarios = [("static", "静止场景(叠加多径野值)"), ("walk", "步行场景 (1.2 m/s, 步行档 0.3~2m/s 代表值)"),
                 ("fast", "快速场景 (3.0 m/s, 快速档 2~5m/s 代表值)"),
                 ("transition", "静止->运动->静止 过渡场景")]

    fig, axes = plt.subplots(len(scenarios), 1, figsize=(9, 12.5), sharex=False)
    metrics = {}
    for ax, (kind, title) in zip(axes, scenarios):
        t, d_true, d_raw, d_out, v_hat_seq, std_seq, states = build_scenario(kind)
        ax.plot(t, d_true, color=MUTED, linewidth=1.4, linestyle="--", label="真实距离 d_true")
        ax.plot(t, d_raw, color=C_ORANGE, linewidth=1.0, alpha=0.8, label="基线测距(未限幅) d_raw")
        ax.plot(t, d_out, color=C_BLUE, linewidth=1.6, label="速度卡限后输出 d_out")
        ax.set_ylabel("距离 (m)")
        ax.set_title(title, fontsize=10.5, loc="left")
        ax.legend(frameon=False, fontsize=8, loc="upper left")

        rmse_raw = float(np.sqrt(np.mean((d_raw - d_true) ** 2)))
        rmse_out = float(np.sqrt(np.mean((d_out - d_true) ** 2)))
        max_step_raw = float(np.max(np.abs(np.diff(d_raw))))
        max_step_out = float(np.max(np.abs(np.diff(d_out))))
        metrics[kind] = {"rmse_raw": rmse_raw, "rmse_out": rmse_out,
                          "max_step_raw": max_step_raw, "max_step_out": max_step_out}

    axes[-1].set_xlabel("时间 (s)")
    fig.suptitle("实验4: 速度卡限对测距输出轨迹的影响 (修正公式后的 v_hat/std 驱动)", y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "exp4_gating_traces.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    return metrics


def classify_state_oracle(v_true, low_speed_bound=2.0):
    """假设速度/状态判别完全准确(直接用真实速度 v_true 代入), 用于展示限幅逻辑
    本身在"上游判据准确"前提下的效果, 与实验0~3揭示的"当前占位参数下判据不够
    灵敏"问题相互独立地评估。"""
    if abs(v_true) < V_TH_STATIC:
        return "static"
    if abs(v_true) < low_speed_bound:
        return "low_speed"
    return "moving"


def apply_speed_gate_oracle(d_raw, v_true_seq, delta_t=DELTA_T):
    n = len(d_raw)
    d_out = np.zeros(n)
    d_out[0] = d_raw[0]
    for i in range(1, n):
        state = classify_state_oracle(v_true_seq[i])
        rng_half = clamp_range(state, v_true_seq[i], delta_t)
        if rng_half is None:
            d_out[i] = d_raw[i]
        else:
            lo, hi = d_out[i - 1] - rng_half, d_out[i - 1] + rng_half
            d_out[i] = min(max(d_raw[i], lo), hi)
    return d_out


def experiment_5_oracle_gating():
    scenarios = [("static", "静止场景(叠加多径野值)"), ("walk", "步行场景 (1.2 m/s, 步行档 0.3~2m/s 代表值)"),
                 ("fast", "快速场景 (3.0 m/s, 快速档 2~5m/s 代表值)"),
                 ("transition", "静止->运动->静止 过渡场景")]

    fig, axes = plt.subplots(len(scenarios), 1, figsize=(9, 12.5), sharex=False)
    metrics = {}
    for ax, (kind, title) in zip(axes, scenarios):
        t, d_true, d_raw, _, _, _, _ = build_scenario(kind)
        v_true_seq = np.concatenate([[0.0], np.diff(d_true) / DELTA_T])
        d_out = apply_speed_gate_oracle(d_raw, v_true_seq)

        ax.plot(t, d_true, color=MUTED, linewidth=1.4, linestyle="--", label="真实距离 d_true")
        ax.plot(t, d_raw, color=C_ORANGE, linewidth=1.0, alpha=0.8, label="基线测距(未限幅) d_raw")
        ax.plot(t, d_out, color=C_VIOLET, linewidth=1.6, label="理想判据限幅后输出 d_out (oracle)")
        ax.set_ylabel("距离 (m)")
        ax.set_title(title, fontsize=10.5, loc="left")
        ax.legend(frameon=False, fontsize=8, loc="upper left")

        rmse_raw = float(np.sqrt(np.mean((d_raw - d_true) ** 2)))
        rmse_out = float(np.sqrt(np.mean((d_out - d_true) ** 2)))
        max_step_raw = float(np.max(np.abs(np.diff(d_raw))))
        max_step_out = float(np.max(np.abs(np.diff(d_out))))
        metrics[kind] = {"rmse_raw": rmse_raw, "rmse_out": rmse_out,
                          "max_step_raw": max_step_raw, "max_step_out": max_step_out}

    axes[-1].set_xlabel("时间 (s)")
    fig.suptitle("实验5: 假设速度/状态判据100%准确(oracle)时, 限幅逻辑本身的效果", y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "exp5_oracle_gating.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    return metrics


# --------------------------------------------------------------------------
# 实验6: 验证 std 对"相邻两轮公共相位偏移(CFO)"的不变性
# --------------------------------------------------------------------------
def experiment_6_cfo_invariance():
    """修正后的 std 基于"到均值向量的偏离", 对整体旋转(即所有频点叠加同一个
    常数相位偏移, 如CFO/本振漂移)天然不变。本实验用仿真验证这一性质,
    说明修正公式相比旧的"复数直接相减"公式在工程上更鲁棒(不需要额外担心
    CFO残留的影响)。"""
    snr_db = 20
    n_trials = 400
    cfo_grid = np.array([0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0])

    def run(cfo):
        vals = []
        for _ in range(n_trials):
            d0 = rng.uniform(*D0_RANGE)
            iq1 = gen_iq(d0, snr_db)
            iq2 = gen_iq(d0, snr_db) * np.exp(1j * cfo)  # 人为叠加公共相位偏移
            dphi = phase_diff(iq1, iq2)
            vals.append(circular_std(dphi))
        return float(np.mean(vals)), float(np.std(vals))

    means, stds = [], []
    for cfo in cfo_grid:
        m, s = run(cfo)
        means.append(m)
        stds.append(s)
    means, stds = np.array(means), np.array(stds)

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.plot(cfo_grid, means, color=C_BLUE, linewidth=2, marker="o", markersize=5)
    ax.fill_between(cfo_grid, means - stds, means + stds, color=C_BLUE, alpha=0.15)
    ax.set_xlabel("人为叠加的公共相位偏移 (rad, 代表CFO/本振漂移)")
    ax.set_ylabel("std (无量纲)")
    ax.set_title(f"实验6: std 对公共相位偏移(CFO)的不变性验证 (SNR={snr_db}dB)\n(曲线应基本水平 —— std只反映\"相位差的一致性\", 与整体旋转无关)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "exp6_cfo_invariance.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {"cfo_grid": cfo_grid.tolist(), "std_mean": means.tolist(),
            "max_relative_variation": float((means.max() - means.min()) / means.mean())}


# --------------------------------------------------------------------------
# 实验7: 用已确认RSSI实测点换算出的真实SNR, 重新评估"步行"误判问题
# --------------------------------------------------------------------------
def experiment_7_real_snr_classification():
    """现网已确认RSSI实测点(1m:-60dBm, 20m:-80dBm), 结合占位的接收机噪声系数(NF)
    换算出各工作距离下的单频点SNR, 用真实(而非占位)SNR重新评估四级速度状态的
    分类性能, 直接检验"步行误判静止"问题在真实SNR水平下是否依然存在。"""
    distances = [1, 5, 10, 20, 50, 100]
    n_trials = 1000

    snr_by_distance = {d: float(snr_from_distance(d)) for d in distances}

    walk_static_rate = {}
    fast_static_rate = {}
    std_walk_mean = {}
    for d in distances:
        snr_db = snr_by_distance[d]
        walk_states, fast_states, walk_stds = [], [], []
        for _ in range(n_trials):
            v = rng.uniform(0.3, 2.0) * rng.choice([-1, 1])
            v_hat, std_dphi, _ = measure_round_pair(d, d + v * DELTA_T, snr_db)
            walk_states.append(classify_state(std_dphi, v_hat))
            walk_stds.append(std_dphi)
        for _ in range(n_trials):
            v = rng.uniform(2.0, 5.0) * rng.choice([-1, 1])
            v_hat, std_dphi, _ = measure_round_pair(d, d + v * DELTA_T, snr_db)
            fast_states.append(classify_state(std_dphi, v_hat))
        walk_static_rate[d] = float(np.mean([s == "static" for s in walk_states]))
        fast_static_rate[d] = float(np.mean([s == "static" for s in fast_states]))
        std_walk_mean[d] = float(np.mean(walk_stds))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    ax.plot(distances, [snr_by_distance[d] for d in distances], color=C_VIOLET, linewidth=2,
            marker="o", markersize=6)
    ax.scatter([1, 20], [RSSI_AT_1M_DBM - NOISE_FLOOR_DBM, RSSI_AT_20M_DBM - NOISE_FLOOR_DBM],
               color=C_RED, zorder=5, s=60, label="已确认RSSI实测点换算值")
    ax.set_xscale("log")
    ax.set_xlabel("工作距离 (m, 对数坐标)")
    ax.set_ylabel("换算SNR (dB)")
    ax.set_title(f"(a) 由已确认RSSI换算的单频点SNR随距离变化\n(NF={RX_NOISE_FIGURE_DB:.0f}dB占位假设, 1m/20m为实测点, 其余为外推)")
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[1]
    ax.plot(distances, [walk_static_rate[d] * 100 for d in distances], color=C_ORANGE, linewidth=2,
            marker="o", markersize=6, label="步行(0.3~2m/s)误判静止率")
    ax.plot(distances, [fast_static_rate[d] * 100 for d in distances], color=C_AQUA, linewidth=2,
            marker="s", markersize=6, label="快速(2~5m/s)误判静止率")
    ax.set_xscale("log")
    ax.set_xlabel("工作距离 (m, 对数坐标)")
    ax.set_ylabel("误判为静止的比例 (%)")
    ax.set_ylim(-5, 105)
    ax.set_title("(b) 真实RSSI换算SNR下, 步行/快速运动的\n静止误判率随距离的变化")
    ax.legend(frameon=False, fontsize=8.5)

    fig.suptitle("实验7: 用已确认RSSI实测点换算的真实SNR重新评估\"步行误判\"问题", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "exp7_real_snr_classification.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {"snr_by_distance_db": snr_by_distance,
            "walk_static_misclassify_rate": walk_static_rate,
            "fast_static_misclassify_rate": fast_static_rate,
            "std_walk_mean_by_distance": std_walk_mean,
            "rx_noise_figure_db_assumed": RX_NOISE_FIGURE_DB,
            "path_loss_exponent_fitted": float(_PATH_LOSS_EXP)}


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------
def main():
    print("运行实验0: std 随SNR的变化(静止) ...")
    exp0 = experiment_0_std_vs_snr()
    print("运行实验0b: std 随运动量(位移)的响应 ...")
    exp0b = experiment_0b_std_vs_motion()
    print("运行实验1: 速度响应特性 ...")
    exp1 = experiment_1_speed_response()
    print("运行实验2: std 分布 ...")
    exp2 = experiment_2_std_distribution()
    print("运行实验3: 分类器性能 ...")
    exp3 = experiment_3_classifier_performance()
    print("运行实验4: 限幅轨迹对比 ...")
    exp4 = experiment_4_gating_traces()
    print("运行实验5: 理想判据下的限幅效果 ...")
    exp5 = experiment_5_oracle_gating()
    print("运行实验6: CFO不变性验证 ...")
    exp6 = experiment_6_cfo_invariance()
    print("运行实验7: 真实RSSI换算SNR下的步行误判评估 ...")
    exp7 = experiment_7_real_snr_classification()

    summary = {
        "params": {
            "K_PATH": K_PATH,
            "N_F": N_F,
            "DELTA_F": DELTA_F,
            "DELTA_T": DELTA_T,
            "STD_TH_LOW": STD_TH_LOW,
            "STD_TH_STATIC": STD_TH_STATIC,
            "V_TH_STATIC": V_TH_STATIC,
            "STATIC_RANGE_SCALE": STATIC_RANGE_SCALE,
            "D0_RANGE": D0_RANGE,
            "RSSI_AT_1M_DBM": RSSI_AT_1M_DBM,
            "RSSI_AT_20M_DBM": RSSI_AT_20M_DBM,
            "RX_NOISE_FIGURE_DB": RX_NOISE_FIGURE_DB,
            "NOISE_FLOOR_DBM": NOISE_FLOOR_DBM,
        },
        "exp0_std_vs_snr": exp0,
        "exp0b_std_vs_motion": exp0b,
        "exp1_speed_response": exp1,
        "exp2_std_distribution": exp2,
        "exp3_classifier_performance": exp3,
        "exp4_gating_traces": exp4,
        "exp5_oracle_gating": exp5,
        "exp6_cfo_invariance": exp6,
        "exp7_real_snr_classification": exp7,
    }
    with open(os.path.join(RESULT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("完成. 结果保存在:", RESULT_DIR)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
