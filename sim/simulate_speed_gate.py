#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
星闪(类蓝牙CS)测距速度卡限算法 - 仿真脚本

复现"设计报告"中描述的完整链路:
  1) 多频点(信道)IQ信号模型
  2) 相邻两次测量IQ做差求相位: dphi(k) = angle(IQ2(k) - IQ1(k))          [已与用户确认为现网实现方式]
  3) 对 dphi(k) 关于频点索引 k 做 DFT/IFFT 峰值搜索, 拟合出相位斜率, 换算为
     两次测量间的位移 Delta_d_hat, 进而得到速度估计 v_hat = Delta_d_hat / Delta_t
  4) 对 dphi(k) 在所有频点上求标准差 std_dphi (rad), 作为动/静判据
  5) 两档限幅:
        档位1(低速): std_dphi < STD_TH_LOW                     -> range = |v_hat| * dt
        档位2(静止): std_dphi < STD_TH_STATIC or |v_hat| < V_TH -> range = 0.1 * |v_hat| * dt
        否则: 不限幅, 直接采用原始测距值

本脚本所有物理/协议层参数 (频点数, 频点间隔, K系数, 测量间隔dt, 噪声与多径模型)
均为仿真占位假设, 已在代码中以 "TODO(实测)" / 注释标出, 需要用实际星闪 CS 协议参数
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
# 1. 仿真参数 (占位假设, 需与实际星闪测距协议核对 —— TODO(实测))
# --------------------------------------------------------------------------
C_LIGHT = 3e8          # 光速 m/s
K_PATH = 2             # 单程=1 / 双程=2, 占位取双程                         TODO(实测)
N_F = 21               # 每次测量的频点(信道)数量, 占位                       TODO(实测)
DELTA_F = 1e6          # 频点间隔 Hz, 占位 1MHz                              TODO(实测)
DELTA_T = 0.01         # 相邻两次测量的时间间隔 s, 占位 10ms                  TODO(实测)
NFFT = 4096            # DFT/IFFT 补零点数, 提高斜率(峰值)搜索分辨率

# 两档限幅阈值 (现网取值, 与用户确认)
STD_TH_LOW = 0.6       # rad, 低速档标准差门限
STD_TH_STATIC = 0.3    # rad, 静止档标准差门限
V_TH_STATIC = 0.1      # m/s, 静止档速度门限
STATIC_RANGE_SCALE = 0.1  # 静止档限幅收紧系数

# 单轮相位斜率(绝对测距)在 N_F 个频点上不发生 2*pi 缠绕的最大距离, 即"非模糊测距范围"
# d_unambig = c / (K * (N_F-1) * delta_f)。 这是本仿真中发现的关键系统约束: 见实验0。
D_UNAMBIG = C_LIGHT / (K_PATH * (N_F - 1) * DELTA_F)

# 两组工作距离区间, 用于对照"在非模糊范围内" vs "超出非模糊范围"两种情形     TODO(实测: 实际部署的典型工作距离)
D0_RANGE_NEAR = (0.3, 0.5 * D_UNAMBIG)   # 近场: 远小于非模糊范围, 单轮相位斜率不缠绕
D0_RANGE_FAR = (0.5 * D_UNAMBIG, 2.0 * D_UNAMBIG)  # 远场: 跨越/超出非模糊范围


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


def phase_diff_raw(iq_prev, iq_curr):
    """现网实现: 复数直接相减后取相角, Delta_phi(k) = angle(IQ2(k) - IQ1(k))."""
    return np.angle(iq_curr - iq_prev)


def dft_slope_to_speed(dphi, delta_f=DELTA_F, delta_t=DELTA_T, k_path=K_PATH, nfft=NFFT):
    """对 dphi(k) 做 DFT(補零), 取幅度谱峰值位置换算出等效位移/速度估计."""
    s = np.exp(1j * dphi)
    spec = np.fft.fftshift(np.fft.fft(s, n=nfft))
    freq_cyc_per_sample = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0))  # cycles/sample, range (-0.5,0.5]
    peak_idx = np.argmax(np.abs(spec))
    f_peak = freq_cyc_per_sample[peak_idx]
    delta_d_hat = -f_peak * C_LIGHT / (delta_f * k_path)
    v_hat = delta_d_hat / delta_t
    return v_hat, delta_d_hat


def measure_round_pair(d_prev, d_curr, snr_db, generator=rng):
    """完整走一遍: 生成两轮IQ -> 相位差 -> 标准差 -> DFT测速."""
    iq1 = gen_iq(d_prev, snr_db, generator=generator)
    iq2 = gen_iq(d_curr, snr_db, generator=generator)
    dphi = phase_diff_raw(iq1, iq2)
    std_dphi = float(np.std(dphi))
    v_hat, delta_d_hat = dft_slope_to_speed(dphi)
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
# 实验0: 理想2径(LOS)+AWGN模型下, 严格静止目标的 std(Δφ) 退化现象
# --------------------------------------------------------------------------
def gen_iq_cfo(d, snr_db, cfo_phase=0.0, n_f=N_F, generator=rng):
    """在 gen_iq 基础上额外叠加一个"公共相位偏移"(代表相邻两轮之间的本振/CFO相位漂移),
    该偏移对同一轮内所有频点相同, 不随频点k变化。 仅用于实验0b 的敏感性分析。"""
    theta = true_phase(d, n_f=n_f) + cfo_phase
    signal = np.exp(1j * theta)
    snr_lin = 10 ** (snr_db / 10.0)
    noise_std = 1.0 / np.sqrt(2.0 * snr_lin)
    noise = generator.normal(0, noise_std, n_f) + 1j * generator.normal(0, noise_std, n_f)
    return signal + noise


def experiment_0_static_degeneracy():
    """核心发现: 对严格静止目标(Delta_d = 0, 无CFO), Delta_phi(k)=angle(IQ2(k)-IQ1(k))
    在理想2径模型下退化为纯噪声差 n2(k)-n1(k) —— 因为信号项 2A*sin(Delta_theta/2) 在
    Delta_theta=0 处恒为0。 圆对称复高斯噪声的相角是严格均匀分布, 其标准差恒为
    理论值 pi/sqrt(3) ≈ 1.814 rad, 与SNR/工作距离都无关。 本实验用 std vs SNR 扫描验证这一点:
    如果现网算法在真实条件下确实对静止有效, 说明真实信道中必然存在某个本模型未包含的
    "非退化"机制(如本振/CFO相位漂移、丰富多径引起的干涉敏感性等), 需要用实测IQ数据核实。
    """
    snr_grid_db = np.arange(0, 81, 5)
    n_trials = 500
    d0_range = D0_RANGE_NEAR

    std_mean, std_std = [], []
    for snr in snr_grid_db:
        stds = []
        for _ in range(n_trials):
            d0 = rng.uniform(*d0_range)
            _, s, _ = measure_round_pair(d0, d0, snr)
            stds.append(s)
        std_mean.append(np.mean(stds))
        std_std.append(np.std(stds))
    std_mean, std_std = np.array(std_mean), np.array(std_std)
    theoretical_uniform_std = float(np.pi / np.sqrt(3))

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(snr_grid_db, std_mean, color=C_BLUE, linewidth=2, marker="o", markersize=4,
            label="std(Δφ) 仿真均值 (严格静止 v_true=0, 无CFO)")
    ax.fill_between(snr_grid_db, std_mean - std_std, std_mean + std_std, color=C_BLUE, alpha=0.15, linewidth=0)
    ax.axhline(theoretical_uniform_std, color=MUTED, linewidth=1.2, linestyle="--",
               label=f"均匀分布理论值 π/√3≈{theoretical_uniform_std:.3f}")
    ax.axhline(STD_TH_STATIC, color=C_GOOD, linewidth=1.2, linestyle=":", label=f"静止门限 {STD_TH_STATIC}")
    ax.axhline(STD_TH_LOW, color=C_CRIT, linewidth=1.0, linestyle=":", label=f"低速门限 {STD_TH_LOW}")
    ax.set_xlabel("单频点 SNR (dB)")
    ax.set_ylabel("std(Δφ) (rad)")
    ax.set_title("实验0: 理想2径模型下, 严格静止目标的 std(Δφ) 不随SNR改善\n(信号项恒为0, 退化为纯噪声相角, 均匀分布)")
    ax.legend(frameon=False, fontsize=8.5, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "exp0_static_degeneracy.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {"theoretical_uniform_std": theoretical_uniform_std,
            "std_at_snr_20db": float(np.interp(20, snr_grid_db, std_mean)),
            "std_at_snr_60db": float(np.interp(60, snr_grid_db, std_mean))}


# --------------------------------------------------------------------------
# 实验0b: 引入"公共相位偏移"(CFO/本振相位漂移占位)后, 能否恢复静止/运动的可分性
# --------------------------------------------------------------------------
def experiment_0b_cfo_sensitivity():
    """在两轮之间叠加一个随机公共相位 cfo_phase ~ N(0, CFO_STD) (代表本振/CFO相位漂移,
    对全部频点相同, 与运动无关), 扫描 CFO_STD 幅度, 对比"严格静止"和"快速运动"两组
    std(Δφ) 的均值差异。 目的: 检验"公共相位偏移"这一现网未明确建模的机制,是否足以
    单独解释std对静止/运动的区分能力。
    """
    snr_db = 20
    n_trials = 400
    d0_range = D0_RANGE_NEAR
    cfo_std_grid = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0])

    def run(v_lo, v_hi, cfo_std):
        stds = []
        for _ in range(n_trials):
            d0 = rng.uniform(*d0_range)
            v_true = 0.0 if v_hi == 0.0 else rng.uniform(v_lo, v_hi) * rng.choice([-1, 1])
            cfo_phase = rng.normal(0, cfo_std) if cfo_std > 0 else 0.0
            iq1 = gen_iq_cfo(d0, snr_db, cfo_phase=0.0)
            iq2 = gen_iq_cfo(d0 + v_true * DELTA_T, snr_db, cfo_phase=cfo_phase)
            stds.append(float(np.std(np.angle(iq2 - iq1))))
        return float(np.mean(stds)), float(np.std(stds))

    static_mean, static_std, moving_mean, moving_std = [], [], [], []
    for cfo_std in cfo_std_grid:
        m, s = run(0.0, 0.0, cfo_std)
        static_mean.append(m)
        static_std.append(s)
        m, s = run(0.5, 2.0, cfo_std)
        moving_mean.append(m)
        moving_std.append(s)
    static_mean, static_std = np.array(static_mean), np.array(static_std)
    moving_mean, moving_std = np.array(moving_mean), np.array(moving_std)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(cfo_std_grid, static_mean, color=C_AQUA, linewidth=2, marker="o", markersize=4, label="静止 (v=0)")
    ax.fill_between(cfo_std_grid, static_mean - static_std, static_mean + static_std, color=C_AQUA, alpha=0.15)
    ax.plot(cfo_std_grid, moving_mean, color=C_ORANGE, linewidth=2, marker="s", markersize=4, label="快速运动 (0.5~2 m/s)")
    ax.fill_between(cfo_std_grid, moving_mean - moving_std, moving_mean + moving_std, color=C_ORANGE, alpha=0.15)
    ax.axhline(STD_TH_STATIC, color=C_GOOD, linewidth=1.2, linestyle=":", label=f"静止门限 {STD_TH_STATIC}")
    ax.axhline(STD_TH_LOW, color=C_CRIT, linewidth=1.0, linestyle=":", label=f"低速门限 {STD_TH_LOW}")
    ax.set_xlabel("相邻两轮公共相位偏移标准差 CFO_STD (rad, 占位假设)")
    ax.set_ylabel("std(Δφ) (rad)")
    ax.set_title(f"实验0b: 引入公共相位偏移(CFO占位)后, 静止 vs 运动的 std 仍高度重合\n(SNR={snr_db}dB, 说明现实速度下相位增量远小于CFO/噪声量级)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "exp0b_cfo_sensitivity.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {"cfo_std_grid": cfo_std_grid.tolist(),
            "static_mean": static_mean.tolist(),
            "moving_mean": moving_mean.tolist(),
            "max_abs_gap": float(np.max(np.abs(static_mean - moving_mean)))}


# --------------------------------------------------------------------------
# 实验1: v_hat 与 std_dphi 随真实速度 / SNR 的变化关系 (工作距离限定在非模糊范围内)
# --------------------------------------------------------------------------
def experiment_1_speed_response():
    v_true_grid = np.linspace(-2.0, 2.0, 41)  # m/s
    snr_list = [10, 20, 30]
    n_trials = 300
    d0_range = D0_RANGE_NEAR  # 近场: 工作距离限定在非模糊范围内, 见实验0结论   TODO(实测:实际测距范围)

    results = {snr: {"v_mean": [], "v_std": [], "std_mean": [], "std_std": []} for snr in snr_list}

    for snr in snr_list:
        for v_true in v_true_grid:
            v_hats, stds = [], []
            for _ in range(n_trials):
                d_prev = rng.uniform(*d0_range)
                d_curr = d_prev + v_true * DELTA_T
                v_hat, std_dphi, _ = measure_round_pair(d_prev, d_curr, snr)
                v_hats.append(v_hat)
                stds.append(std_dphi)
            results[snr]["v_mean"].append(np.mean(v_hats))
            results[snr]["v_std"].append(np.std(v_hats))
            results[snr]["std_mean"].append(np.mean(stds))
            results[snr]["std_std"].append(np.std(stds))

    colors = {10: C_ORANGE, 20: C_BLUE, 30: C_AQUA}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    ax.plot([-2, 2], [-2, 2], color=MUTED, linewidth=1.2, linestyle="--", label="理想 v_hat = v_true")
    for snr in snr_list:
        r = results[snr]
        ax.plot(v_true_grid, r["v_mean"], color=colors[snr], linewidth=2, label=f"SNR={snr}dB")
        ax.fill_between(
            v_true_grid,
            np.array(r["v_mean"]) - np.array(r["v_std"]),
            np.array(r["v_mean"]) + np.array(r["v_std"]),
            color=colors[snr],
            alpha=0.15,
            linewidth=0,
        )
    ax.set_xlabel("真实速度 v_true (m/s)")
    ax.set_ylabel("DFT 斜率法速度估计 v_hat (m/s)")
    ax.set_title("(a) 速度估计 v_hat vs 真实速度")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    for snr in snr_list:
        r = results[snr]
        ax.plot(v_true_grid, r["std_mean"], color=colors[snr], linewidth=2, label=f"SNR={snr}dB")
        ax.fill_between(
            v_true_grid,
            np.array(r["std_mean"]) - np.array(r["std_std"]),
            np.array(r["std_mean"]) + np.array(r["std_std"]),
            color=colors[snr],
            alpha=0.15,
            linewidth=0,
        )
    ax.axhline(STD_TH_STATIC, color=C_GOOD, linewidth=1.2, linestyle=":", label=f"静止门限 {STD_TH_STATIC}")
    ax.axhline(STD_TH_LOW, color=C_CRIT, linewidth=1.2, linestyle=":", label=f"低速门限 {STD_TH_LOW}")
    ax.set_xlabel("真实速度 v_true (m/s)")
    ax.set_ylabel("相位差标准差 std(Δφ) (rad)")
    ax.set_title("(b) 相位差标准差 vs 真实速度")
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle("实验1: 现网\"复数直接相减取相角\"公式的速度响应特性", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "exp1_speed_response.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {
        "v_true_grid": v_true_grid.tolist(),
        "snr_list": snr_list,
        "results": {str(k): v for k, v in results.items()},
    }


# --------------------------------------------------------------------------
# 实验2: 三种运动状态下 std(Δφ) 的分布 (直方图/箱线图), 近场 vs 远场对比
# --------------------------------------------------------------------------
def experiment_2_std_distribution():
    snr_db = 20
    n_trials = 1000

    states = {
        "静止\n(v=0)": (0.0, 0.0),
        "低速\n(0.05~0.3 m/s)": (0.05, 0.3),
        "快速运动\n(0.5~2 m/s)": (0.5, 2.0),
    }
    box_colors = [C_AQUA, C_YELLOW, C_ORANGE]

    def collect(d0_range):
        data = {}
        for label, (vlo, vhi) in states.items():
            stds = []
            for _ in range(n_trials):
                v_true = 0.0 if vlo == vhi == 0.0 else rng.uniform(vlo, vhi) * rng.choice([-1, 1])
                d_prev = rng.uniform(*d0_range)
                d_curr = d_prev + v_true * DELTA_T
                _, std_dphi, _ = measure_round_pair(d_prev, d_curr, snr_db)
                stds.append(std_dphi)
            data[label] = np.array(stds)
        return data

    data_near = collect(D0_RANGE_NEAR)
    data_far = collect(D0_RANGE_FAR)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, data, title in [
        (axes[0], data_near, f"(a) 近场 d0∈[{D0_RANGE_NEAR[0]:.2f},{D0_RANGE_NEAR[1]:.2f}]m (未超非模糊范围)"),
        (axes[1], data_far, f"(b) 远场 d0∈[{D0_RANGE_FAR[0]:.2f},{D0_RANGE_FAR[1]:.2f}]m (跨越非模糊范围)"),
    ]:
        positions = np.arange(len(data))
        bp = ax.boxplot(
            list(data.values()), positions=positions, widths=0.5, patch_artist=True,
            showfliers=False, medianprops=dict(color=INK, linewidth=1.5),
        )
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
            patch.set_edgecolor(color)
        ax.axhline(STD_TH_STATIC, color=C_GOOD, linewidth=1.2, linestyle=":", label=f"静止门限 {STD_TH_STATIC}")
        ax.axhline(STD_TH_LOW, color=C_CRIT, linewidth=1.2, linestyle=":", label=f"低速门限 {STD_TH_LOW}")
        ax.set_xticks(positions)
        ax.set_xticklabels(list(data.keys()))
        ax.set_title(title, fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].set_ylabel("相位差标准差 std(Δφ) (rad)")

    fig.suptitle(f"实验2: 三种运动状态下 std(Δφ) 分布 —— 工作距离是否在非模糊范围内的对比 (SNR={snr_db}dB, N={n_trials}次/组)",
                 y=1.03, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "exp2_std_distribution.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    def summarize(data):
        return {k: {"mean": float(v.mean()), "std": float(v.std()), "p10": float(np.percentile(v, 10)),
                     "p90": float(np.percentile(v, 90))} for k, v in data.items()}

    return {"near_range": summarize(data_near), "far_range": summarize(data_far)}


# --------------------------------------------------------------------------
# 实验3: 两档判据的分类性能 (混淆矩阵 + 阈值扫描)
# --------------------------------------------------------------------------
def experiment_3_classifier_performance():
    snr_db = 20
    n_trials = 600
    d0_range = D0_RANGE_NEAR  # 近场: 工作距离限定在非模糊范围内, 见实验0结论

    truth_states = {
        "static": (0.0, 0.0),
        "low_speed": (0.05, 0.3),
        "moving": (0.5, 2.0),
    }
    order = ["static", "low_speed", "moving"]

    confusion = {t: {p: 0 for p in order} for t in order}
    for truth, (vlo, vhi) in truth_states.items():
        for _ in range(n_trials):
            v_true = 0.0 if vlo == vhi == 0.0 else rng.uniform(vlo, vhi) * rng.choice([-1, 1])
            d_prev = rng.uniform(*d0_range)
            d_curr = d_prev + v_true * DELTA_T
            v_hat, std_dphi, _ = measure_round_pair(d_prev, d_curr, snr_db)
            pred = classify_state(std_dphi, v_hat)
            confusion[truth][pred] += 1

    # 阈值扫描: 静止门限 std_th 从 0.05 到 1.5, 统计"真实静止"被正确判为静止的比例(检出率)
    # 以及"真实快速运动"被误判为静止的比例(虚警率)
    std_th_grid = np.linspace(0.05, 1.5, 60)
    det_rate, far_rate = [], []
    static_stds, moving_stds = [], []
    for _ in range(n_trials):
        d_prev = rng.uniform(*d0_range)
        _, s, _ = measure_round_pair(d_prev, d_prev, snr_db)
        static_stds.append(s)
    for _ in range(n_trials):
        v_true = rng.uniform(0.5, 2.0) * rng.choice([-1, 1])
        d_prev = rng.uniform(*d0_range)
        d_curr = d_prev + v_true * DELTA_T
        _, s, _ = measure_round_pair(d_prev, d_curr, snr_db)
        moving_stds.append(s)
    static_stds = np.array(static_stds)
    moving_stds = np.array(moving_stds)
    for th in std_th_grid:
        det_rate.append(float(np.mean(static_stds < th)))
        far_rate.append(float(np.mean(moving_stds < th)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    mat = np.array([[confusion[t][p] for p in order] for t in order], dtype=float)
    mat_norm = mat / mat.sum(axis=1, keepdims=True)
    im = ax.imshow(mat_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    label_map = {"static": "判静止", "low_speed": "判低速", "moving": "判运动"}
    label_map_t = {"static": "真实静止", "low_speed": "真实低速", "moving": "真实运动"}
    ax.set_xticklabels([label_map[p] for p in order])
    ax.set_yticklabels([label_map_t[t] for t in order])
    for i in range(len(order)):
        for j in range(len(order)):
            ax.text(j, i, f"{mat_norm[i, j]*100:.0f}%\n({int(mat[i, j])})", ha="center", va="center",
                     color=INK if mat_norm[i, j] < 0.6 else "white", fontsize=9)
    ax.set_title(f"(a) 混淆矩阵 (SNR={snr_db}dB, 现网阈值 0.6/0.3/0.1)")

    ax = axes[1]
    ax.plot(std_th_grid, det_rate, color=C_BLUE, linewidth=2, label="真实静止 -> std<阈值 比例 (检出率)")
    ax.plot(std_th_grid, far_rate, color=C_RED, linewidth=2, label="真实快速运动 -> std<阈值 比例 (虚警率)")
    ax.axvline(STD_TH_STATIC, color=C_GOOD, linewidth=1.2, linestyle=":", label=f"现网静止门限 {STD_TH_STATIC}")
    ax.set_xlabel("std 判决门限 (rad)")
    ax.set_ylabel("比例")
    ax.set_title("(b) 静止门限扫描: 检出率 / 虚警率")
    ax.legend(frameon=False, fontsize=8.5, loc="center right")

    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "exp3_classifier_performance.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    det_at_th = float(np.mean(static_stds < STD_TH_STATIC))
    far_at_th = float(np.mean(moving_stds < STD_TH_STATIC))
    return {"confusion": {t: confusion[t] for t in order},
            "static_detect_rate_at_threshold": det_at_th,
            "moving_false_alarm_rate_at_threshold": far_at_th}


# --------------------------------------------------------------------------
# 实验4: 限幅前后测距输出轨迹对比 (静止抖动抑制 / 运动跟踪 / 状态切换过渡)
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


def build_scenario(kind, n_rounds=200, snr_db=20):
    t = np.arange(n_rounds) * DELTA_T
    if kind == "static":
        d_true = np.full(n_rounds, 3.0) + rng.normal(0, 0.001, n_rounds)  # mm级静态微抖动
    elif kind == "moving":
        d_true = 3.0 + 0.3 * t  # 恒速 0.3 m/s
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
    scenarios = [("static", "静止场景(叠加多径野值)"), ("moving", "恒速运动场景 (0.3 m/s)"),
                 ("transition", "静止->运动->静止 过渡场景")]

    fig, axes = plt.subplots(len(scenarios), 1, figsize=(9, 10), sharex=False)
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
        metrics[kind] = {
            "rmse_raw": rmse_raw,
            "rmse_out": rmse_out,
            "max_step_raw": max_step_raw,
            "max_step_out": max_step_out,
        }

    axes[-1].set_xlabel("时间 (s)")
    fig.suptitle("实验4: 速度卡限对测距输出轨迹的影响", y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "exp4_gating_traces.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    return metrics


# --------------------------------------------------------------------------
# 实验5: 理想(oracle)判据下的限幅效果 —— 与卡限"逻辑"本身的价值解耦
# --------------------------------------------------------------------------
def classify_state_oracle(v_true, low_speed_bound=0.5):
    """假设速度/状态判别完全准确(直接用真实速度 v_true 代入), 用于展示限幅逻辑
    本身在"上游判据准确"前提下的效果, 与实验0/0b/1/2/3揭示的"当前公式在假设参数下
    判据失效"问题相互独立地评估。"""
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
    scenarios = [("static", "静止场景(叠加多径野值)"), ("moving", "恒速运动场景 (0.3 m/s)"),
                 ("transition", "静止->运动->静止 过渡场景")]

    fig, axes = plt.subplots(len(scenarios), 1, figsize=(9, 10), sharex=False)
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
        metrics[kind] = {
            "rmse_raw": rmse_raw,
            "rmse_out": rmse_out,
            "max_step_raw": max_step_raw,
            "max_step_out": max_step_out,
        }

    axes[-1].set_xlabel("时间 (s)")
    fig.suptitle("实验5: 假设速度/状态判据100%准确(oracle)时, 限幅逻辑本身的效果", y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "exp5_oracle_gating.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    return metrics


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------
def main():
    print("运行实验0: 理想模型下静止目标的std退化现象 ...")
    exp0 = experiment_0_static_degeneracy()
    print("运行实验0b: 公共相位偏移(CFO占位)敏感性分析 ...")
    exp0b = experiment_0b_cfo_sensitivity()
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
            "D_UNAMBIG": D_UNAMBIG,
            "D0_RANGE_NEAR": D0_RANGE_NEAR,
            "D0_RANGE_FAR": D0_RANGE_FAR,
        },
        "exp0_static_degeneracy": exp0,
        "exp0b_cfo_sensitivity": exp0b,
        "exp2_std_distribution": exp2,
        "exp3_classifier_performance": exp3,
        "exp4_gating_traces": exp4,
        "exp5_oracle_gating": exp5,
    }
    with open(os.path.join(RESULT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("完成. 结果保存在:", RESULT_DIR)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
