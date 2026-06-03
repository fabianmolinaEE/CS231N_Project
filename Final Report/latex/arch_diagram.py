"""Generate three separate architecture diagram PDFs.

Outputs:
  arch_plaincnn.pdf     — PlainCNN (vertical dilated stack)
  arch_encoderdecoder.pdf — EncoderDecoder (U-shape, no skip connections)
  arch_unet.pdf           — U-Net (U-shape, with skip connections)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BLUE   = "#AED6F1"
ORANGE = "#FAD7A0"
GREEN  = "#A9DFBF"
PURPLE = "#D2B4DE"
GRAY   = "#D5DBDB"
SKIP_COLOR = "#888888"


def draw_block(ax, x, y, w, h, label, color, fontsize=6):
    rect = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02", linewidth=0.7,
        edgecolor="#444444", facecolor=color, zorder=3,
    )
    ax.add_patch(rect)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, zorder=4)


def arrow(ax, x1, y1, x2, y2, dashed=False, color="#444444", lw=1.0):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=lw,
            linestyle="dashed" if dashed else "solid",
            mutation_scale=8,
        ),
        zorder=5,
    )


# ---------------------------------------------------------------------------
# PlainCNN
# ---------------------------------------------------------------------------
def make_plaincnn():
    fig, ax = plt.subplots(figsize=(2.4, 4.2))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.04)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    DILS = [1, 1, 2, 2, 4, 4, 8, 8]
    n = len(DILS)
    h = 0.076
    gap = 0.012
    step = h + gap
    y_top = 0.945
    ys = [y_top - i * step for i in range(n)]

    for i, (y, d) in enumerate(zip(ys, DILS)):
        draw_block(ax, 0.5, y, 0.82, h, f"DoubleConv  d={d}", BLUE, fontsize=6.5)
        if i < n - 1:
            arrow(ax, 0.5, y - h / 2, 0.5, ys[i + 1] + h / 2)

    out_y = ys[-1] - h - gap
    draw_block(ax, 0.5, out_y, 0.65, h * 0.85, "Conv 1×1 → Output", GRAY, fontsize=6)
    arrow(ax, 0.5, ys[-1] - h / 2, 0.5, out_y + h * 0.425)

    ax.text(0.5, out_y - 0.045, "b=64  ≈600K params",
            ha="center", fontsize=5.5, color="#555555")

    fig.savefig("arch_plaincnn.pdf", bbox_inches="tight", dpi=150)
    fig.savefig("arch_plaincnn.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved arch_plaincnn.pdf/png")


# ---------------------------------------------------------------------------
# Shared encoder/decoder geometry for EncDec + U-Net
# ---------------------------------------------------------------------------
ENC_X   = 0.23   # encoder column centre
DEC_X   = 0.77   # decoder column centre
BOT_X   = 0.50   # bottleneck centre
H       = 0.11   # block height

# enc_ws[i] is width of encoder stage i (256,128,64,32 px)
ENC_WS  = [0.27, 0.22, 0.17, 0.13]
DEC_WS  = ENC_WS[::-1]           # [0.13, 0.17, 0.22, 0.27]
BOT_W   = 0.12

ENC_YS  = [0.88, 0.72, 0.56, 0.40]   # top → bottom
DEC_YS  = [0.40, 0.56, 0.72, 0.88]   # bottom → top  (same y-levels as encoder)
BOT_Y   = 0.22

ENC_LBLS = ["Enc1\n256px", "Enc2\n128px", "Enc3\n64px", "Enc4\n32px"]
DEC_LBLS = ["Dec4\n32px",  "Dec3\n64px",  "Dec2\n128px", "Dec1\n256px"]


def _draw_encoder(ax):
    for i, (y, w, lbl) in enumerate(zip(ENC_YS, ENC_WS, ENC_LBLS)):
        draw_block(ax, ENC_X, y, w, H, lbl, ORANGE, fontsize=5.5)
        if i < len(ENC_YS) - 1:
            arrow(ax, ENC_X, y - H / 2, ENC_X, ENC_YS[i + 1] + H / 2)
    # enc4 → bottleneck (diagonal down-right)
    draw_block(ax, BOT_X, BOT_Y, BOT_W, H, "Bottleneck\n16px", PURPLE, fontsize=5.5)
    arrow(ax, ENC_X, ENC_YS[-1] - H / 2, BOT_X - BOT_W / 2, BOT_Y)


def _draw_decoder(ax):
    for i, (y, w, lbl) in enumerate(zip(DEC_YS, DEC_WS, DEC_LBLS)):
        draw_block(ax, DEC_X, y, w, H, lbl, GREEN, fontsize=5.5)
        if i < len(DEC_YS) - 1:
            arrow(ax, DEC_X, DEC_YS[i] + H / 2, DEC_X, DEC_YS[i + 1] - H / 2)
    # bottleneck → dec4 (diagonal up-right)
    arrow(ax, BOT_X + BOT_W / 2, BOT_Y, DEC_X, DEC_YS[0] - H / 2)


def _add_param_label(ax, text):
    ax.text(0.5, BOT_Y - 0.08, text, ha="center", fontsize=5.5, color="#555555")


# ---------------------------------------------------------------------------
# EncoderDecoder
# ---------------------------------------------------------------------------
def make_encoderdecoder():
    fig, ax = plt.subplots(figsize=(3.4, 3.8))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.04)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _draw_encoder(ax)
    _draw_decoder(ax)
    _add_param_label(ax, "b=64  ≈28M params")

    fig.savefig("arch_encoderdecoder.pdf", bbox_inches="tight", dpi=150)
    fig.savefig("arch_encoderdecoder.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved arch_encoderdecoder.pdf/png")


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------
def make_unet():
    fig, ax = plt.subplots(figsize=(3.4, 3.8))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.04)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _draw_encoder(ax)
    _draw_decoder(ax)

    # Skip connections: enc stage i → dec stage at same resolution (same y-level)
    # enc_ys[i] pairs with dec_ys[3-i] (same y by construction).
    # Arrow goes from right edge of encoder block to left edge of matching decoder block.
    skip_labeled = False
    for i in range(len(ENC_YS)):
        src_x = ENC_X + ENC_WS[i] / 2          # right edge of encoder block i
        j     = len(DEC_YS) - 1 - i             # matching decoder index (same resolution)
        dst_x = DEC_X - DEC_WS[j] / 2           # left edge of that decoder block
        y     = ENC_YS[i]                        # same y on both sides

        arrow(ax, src_x, y, dst_x, y,
              dashed=True, color=SKIP_COLOR, lw=0.9)

        if not skip_labeled:
            mid_x = (src_x + dst_x) / 2
            ax.text(mid_x, y + 0.045, "skip connection",
                    ha="center", fontsize=4.5, color=SKIP_COLOR, style="italic")
            skip_labeled = True

    _add_param_label(ax, "b=32  ≈2M params")

    fig.savefig("arch_unet.pdf", bbox_inches="tight", dpi=150)
    fig.savefig("arch_unet.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved arch_unet.pdf/png")


if __name__ == "__main__":
    make_plaincnn()
    make_encoderdecoder()
    make_unet()
