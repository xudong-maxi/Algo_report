#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成算法总体框图(替代Word中渲染错位的ASCII art)"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

C_BLUE = "#2a78d6"
C_ORANGE = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "WenQuanYi Zen Hei",
    "axes.unicode_minus": False,
})

fig, ax = plt.subplots(figsize=(9.5, 4.8))
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ax.set_xlim(0, 10.3)
ax.set_ylim(0, 5.1)
ax.axis("off")


def box(cx, cy, w, h, text, color, fontsize=10, fontweight="normal", text_color=INK):
    b = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                        boxstyle="round,pad=0.06,rounding_size=0.08",
                        linewidth=1.6, edgecolor=color, facecolor="white", zorder=2)
    ax.add_patch(b)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight=fontweight, zorder=3, linespacing=1.5)


def arrow(x1, y1, x2, y2, color=INK2, lw=1.6, connectionstyle="arc3,rad=0.0"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                         linewidth=lw, color=color, zorder=1, connectionstyle=connectionstyle,
                         shrinkA=2, shrinkB=2)
    ax.add_patch(a)


# 基线测距模块 (黑盒)
box(2.3, 4.15, 3.2, 1.0, "基线测距模块(黑盒)\n本报告范围外,现网已有", MUTED, fontsize=9.5)

# 速度/标准差估计模块
box(2.5, 2.15, 3.8, 2.0,
    "速度/标准差估计模块\n(本报告设计范围)\n\nθ=angle(IQ), Δφ=θ2-θ1\nDFT斜率拟合 → v_hat\nIQ_diff=exp(jΔφ) → 复数std",
    C_BLUE, fontsize=9)

# 两档限幅模块
box(8.3, 3.1, 2.8, 1.6, "两档限幅模块\n(本报告设计范围)", C_ORANGE, fontsize=10, fontweight="bold")

# 输入标注与箭头
ax.text(0.1, 4.4, "IQ(k, n)\nIQ(k, n-1)", fontsize=9, va="center", color=INK2, linespacing=1.4)
arrow(1.05, 4.15, 0.68, 4.15)

ax.text(0.1, 2.45, "IQ(k, n)\nIQ(k, n-1)", fontsize=9, va="center", color=INK2, linespacing=1.4)
arrow(1.05, 2.15, 0.68, 2.15)

# 基线测距模块 -> d_raw(n) -> 两档限幅模块
arrow(3.9, 4.15, 6.95, 3.55, connectionstyle="arc3,rad=-0.15")
ax.text(5.4, 4.25, "d_raw(n)", fontsize=9.5, color=INK, ha="center")

# 速度/标准差估计模块 -> v_hat/std -> 两档限幅模块
arrow(4.4, 2.55, 6.95, 2.95, connectionstyle="arc3,rad=0.15")
ax.text(5.65, 2.2, "v_hat(n), std(n)", fontsize=9.5, color=INK, ha="center")

# d_out(n-1) 状态反馈
arrow(8.3, 2.3, 8.3, 0.55, connectionstyle="arc3,rad=0")
arrow(8.3, 0.55, 3.6, 0.55, connectionstyle="arc3,rad=0")
arrow(3.6, 0.55, 2.5, 1.15, connectionstyle="arc3,rad=-0.2")
ax.text(5.8, 0.28, "d_out(n-1)  (状态反馈至限幅模块)", fontsize=9, color=MUTED, ha="center")

# 输出
arrow(9.7, 3.1, 10.15, 3.1)
ax.text(10.2, 3.1, "d_out(n)", fontsize=10.5, color=INK, fontweight="bold", va="center")

ax.text(5.1, 4.95, "速度/标准差估计模块与限幅模块是本报告的设计范围;基线测距模块视为已有的上游黑盒",
        fontsize=8.5, color=MUTED, ha="center", style="italic")

fig.tight_layout()
fig.savefig("/home/user/Algo_report/report/figures/block_diagram.png", dpi=200, bbox_inches="tight",
            facecolor=SURFACE)
print("saved")
