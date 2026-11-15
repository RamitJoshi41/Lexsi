"""Headline figure: the gap (left) and the fix (right). Numbers from results/final_v2.json + bootstrap_ci.json (held-out)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib as mpl

mpl.rcParams.update({"font.family": "Inter", "font.size": 11, "axes.edgecolor": "#c9c8c3", "axes.linewidth": 0.8,
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "text.color": "#0b0b0b"})
BLUE, LIGHT, SURF, INK2, MUTED = "#2a78d6", "#86b6ef", "#fcfcfb", "#52514e", "#8a8984"

fig = plt.figure(figsize=(12, 6.3), dpi=200, facecolor=SURF)
fig.text(0.04, 0.93, "Knows, But Doesn't Refuse", fontsize=22, weight="bold")
fig.text(0.04, 0.875, "Tiny Aya Global on held-out harmful prompts: the same request, typed the way India chats, is refused far less, "
         "and a gated refusal steer fixes it.", fontsize=11.5, color=INK2)


def rbar(ax, x, h, w, color, z=2):
    # bar with 4px-ish rounded top, anchored to baseline
    ax.add_patch(FancyBboxPatch((x - w / 2, 0), w, h, boxstyle="round,pad=0,rounding_size=1.2", linewidth=0,
                                facecolor=color, zorder=z, mutation_aspect=1 / 0.35 * 0.35))


# ---- left: the gap
ax = fig.add_axes([0.05, 0.25, 0.40, 0.53], facecolor=SURF)
scripts = ["English", "Hindi\nDevanagari", "Hindi\nromanized", "Punjabi\nGurmukhi", "Punjabi\nromanized"]
vals = [74, 59, 30, 52, 20]
for i, v in enumerate(vals):
    c = BLUE if i not in (2, 4) else "#eb6834"
    ax.bar(i, v, 0.62, color=c, zorder=2)
    ax.text(i, v + 2, f"{v}%", ha="center", va="bottom", fontsize=12, weight="bold" if i in (2, 4) else "normal")
ax.set_xticks(range(5), scripts, fontsize=10)
ax.set_ylim(0, 100); ax.set_yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
ax.grid(axis="y", color="#ecebe7", zorder=0); ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.set_title("1  The gap: refused by plain Tiny Aya", loc="left", fontsize=13, weight="bold", pad=26)
ax.text(0, -0.17, "Orange = romanized (Latin-script) Hindi / Punjabi.\n46 held-out harmful prompts per script; judge: Aya Expanse 8B.",
        transform=ax.transAxes, fontsize=8.5, color=MUTED, va="top")

# ---- right: the fix
ax2 = fig.add_axes([0.54, 0.25, 0.42, 0.53], facecolor=SURF)
meth = ["Plain\nTiny Aya", "SafeTune CAST\n(all-script calib.)", "Ours: gated\nsteer (v3)"]
judge = [40, 88, 86]; usable = [38, 40, 76]; lo = [31, 32, 69]; hi = [46, 48, 83]
w = 0.34
for i in range(3):
    ax2.bar(i - w / 2 - 0.01, judge[i], w, color=LIGHT, zorder=2)
    ax2.bar(i + w / 2 + 0.01, usable[i], w, color=BLUE, zorder=2)
    ax2.errorbar(i + w / 2 + 0.01, usable[i], yerr=[[usable[i] - lo[i]], [hi[i] - usable[i]]], fmt="none",
                 ecolor="#0b0b0b", elinewidth=1, capsize=3, zorder=3)
    ax2.text(i - w / 2 - 0.01, judge[i] + 2, f"{judge[i]}%", ha="center", va="bottom", fontsize=10.5, color=INK2)
    ax2.text(i + w / 2 + 0.01, hi[i] + 2, f"{usable[i]}%", ha="center", va="bottom", fontsize=12.5, weight="bold")
ax2.text(1 - w / 2 - 0.01, 44, "46%\nare\nloops", ha="center", va="center", fontsize=9.5, color="#0b0b0b")
ax2.set_xticks(range(3), meth, fontsize=10)
ax2.set_ylim(0, 100); ax2.set_yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
ax2.grid(axis="y", color="#ecebe7", zorder=0); ax2.spines[["top", "right", "left"]].set_visible(False)
ax2.tick_params(axis="y", length=0)
ax2.set_title("2  The fix: non-English harmful prompts", loc="left", fontsize=13, weight="bold", pad=26)
from matplotlib.patches import Patch
ax2.legend(handles=[Patch(color=LIGHT, label="judge says \"refused\""),
                    Patch(color=BLUE, label="usable refusal ± 95% CI")],
           loc="lower left", frameon=False, fontsize=9, bbox_to_anchor=(-0.01, 0.99), ncol=2, handlelength=1.2, columnspacing=1.2)
ax2.text(0, -0.17, "Usable = refused, not a loop, in a language the user reads. 184 held-out prompts.\nOurs vs CAST: +36 pts [+25, +46], McNemar p < 1e-10.",
         transform=ax2.transAxes, fontsize=8.5, color=MUTED, va="top")

fig.text(0.04, 0.025, "No retraining  ·  MMLU 50.5 → 50.5  ·  guard fires on 2 / 600 everyday requests  ·  no measurable latency  "
         "·  FOOTER4", fontsize=10.5, color=INK2)
import sys
footer = sys.argv[1] if len(sys.argv) > 1 else "works on the 4-bit model too"
for t in fig.texts:
    if "FOOTER4" in t.get_text(): t.set_text(t.get_text().replace("FOOTER4", footer))
fig.savefig("fig_headline.png", facecolor=SURF)
print("saved")
