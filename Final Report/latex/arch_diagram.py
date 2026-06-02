import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, axes = plt.subplots(1, 3, figsize=(7, 3.2))
fig.subplots_adjust(wspace=0.35, left=0.02, right=0.98, top=0.88, bottom=0.05)


def draw_block(ax, x, y, w, h, label, color, fontsize=5):
    rect = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02", linewidth=0.6,
        edgecolor="#555555", facecolor=color,
    )
    ax.add_patch(rect)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize)


def draw_arrow(ax, x1, y1, x2, y2, dashed=False, color="#555555"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8,
                        linestyle="dashed" if dashed else "solid"),
    )


# ---- Panel 1: PlainCNN ----
ax = axes[0]
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.set_title("PlainCNN", fontsize=7, fontweight="bold", pad=3)

BLUE = "#AED6F1"
DILS = [1, 1, 2, 2, 4, 4, 8, 8]
n = len(DILS)
ys = [0.92 - i * (0.82 / n) for i in range(n)]
for i, (y, d) in enumerate(zip(ys, DILS)):
    draw_block(ax, 0.5, y, 0.72, 0.085, f"DoubleConv\nd={d}", BLUE, fontsize=4.5)
    if i < n - 1:
        draw_arrow(ax, 0.5, y - 0.042, 0.5, ys[i + 1] + 0.042)
out_y = ys[-1] - 0.11
draw_block(ax, 0.5, out_y, 0.55, 0.07, "Conv 1×1 → Output", "#D5DBDB", fontsize=4.5)
draw_arrow(ax, 0.5, ys[-1] - 0.042, 0.5, out_y + 0.035)
ax.text(0.5, 0.045, "b=64, ~600K params", ha="center", fontsize=4, color="#555555")

# ---- Panel 2: EncoderDecoder ----
ax = axes[1]
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.set_title("EncoderDecoder", fontsize=7, fontweight="bold", pad=3)

ORANGE = "#FAD7A0"
GREEN = "#A9DFBF"
enc_labels = ["Enc1\n256px", "Enc2\n128px", "Enc3\n64px", "Enc4\n32px"]
enc_ys = [0.90, 0.74, 0.58, 0.42]
enc_ws = [0.60, 0.48, 0.38, 0.30]
dec_labels = ["Dec4\n32px", "Dec3\n64px", "Dec2\n128px", "Dec1\n256px"]
dec_ys = [0.42, 0.58, 0.74, 0.90]
dec_ws = enc_ws[::-1]

for i, (y, w, lbl) in enumerate(zip(enc_ys, enc_ws, enc_labels)):
    draw_block(ax, 0.3, y, w, 0.10, lbl, ORANGE, fontsize=4.5)
    if i < len(enc_ys) - 1:
        draw_arrow(ax, 0.3, y - 0.05, 0.3, enc_ys[i + 1] + 0.05)

bot_y = 0.26
draw_block(ax, 0.5, bot_y, 0.24, 0.10, "Bottleneck\n16px", "#D2B4DE", fontsize=4.5)
draw_arrow(ax, 0.3, enc_ys[-1] - 0.05, 0.3, bot_y + 0.05)
draw_arrow(ax, 0.38, bot_y, 0.62, bot_y)

for i, (y, w, lbl) in enumerate(zip(dec_ys, dec_ws, dec_labels)):
    draw_block(ax, 0.7, y, w, 0.10, lbl, GREEN, fontsize=4.5)
    if i < len(dec_ys) - 1:
        draw_arrow(ax, 0.7, dec_ys[i] + 0.05, 0.7, dec_ys[i + 1] - 0.05)

draw_arrow(ax, 0.62, bot_y, 0.7, dec_ys[0] - 0.05)
ax.text(0.5, 0.045, "b=64, ~28M params", ha="center", fontsize=4, color="#555555")

# ---- Panel 3: U-Net ----
ax = axes[2]
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.set_title("U-Net", fontsize=7, fontweight="bold", pad=3)

for i, (y, w, lbl) in enumerate(zip(enc_ys, enc_ws, enc_labels)):
    draw_block(ax, 0.3, y, w, 0.10, lbl, ORANGE, fontsize=4.5)
    if i < len(enc_ys) - 1:
        draw_arrow(ax, 0.3, y - 0.05, 0.3, enc_ys[i + 1] + 0.05)

draw_block(ax, 0.5, bot_y, 0.24, 0.10, "Bottleneck\n16px", "#D2B4DE", fontsize=4.5)
draw_arrow(ax, 0.3, enc_ys[-1] - 0.05, 0.3, bot_y + 0.05)
draw_arrow(ax, 0.38, bot_y, 0.62, bot_y)

for i, (y, w, lbl) in enumerate(zip(dec_ys, dec_ws, dec_labels)):
    draw_block(ax, 0.7, y, w, 0.10, lbl, GREEN, fontsize=4.5)
    if i < len(dec_ys) - 1:
        draw_arrow(ax, 0.7, dec_ys[i] + 0.05, 0.7, dec_ys[i + 1] - 0.05)

draw_arrow(ax, 0.62, bot_y, 0.7, dec_ys[0] - 0.05)

# Skip connection dashed arrows
skip_labeled = False
for ey, dw in zip(enc_ys, dec_ws):
    draw_arrow(ax, 0.3 + enc_ws[enc_ys.index(ey)] / 2, ey,
               0.7 - dw / 2, ey, dashed=True, color="#888888")
    if not skip_labeled:
        ax.text(0.50, ey + 0.04, "skip", ha="center", fontsize=3.5,
                color="#888888", style="italic")
        skip_labeled = True

ax.text(0.5, 0.045, "b=32, ~2M params", ha="center", fontsize=4, color="#555555")

plt.savefig("arch_diagram.pdf", bbox_inches="tight", dpi=150)
plt.savefig("arch_diagram.png", bbox_inches="tight", dpi=150)
print("Saved arch_diagram.pdf and arch_diagram.png")
