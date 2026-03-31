#!/usr/bin/env python3
"""
Graphical Abstract for Technovation submission.
Generates a clean, Nature-style visual flow (left → right) in three panels.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# ── Color palette (matches other figures) ────────────────────────────────
PRIMARY   = '#2C3E50'   # navy
SECONDARY = '#E74C3C'   # red
TERTIARY  = '#3498DB'   # blue
ACCENT    = '#27AE60'   # green
GRAY      = '#95A5A6'   # gray
WHITE     = '#FFFFFF'
LIGHT_BG  = '#F8F9FA'

# Lighter tints for panel backgrounds
BLUE_LIGHT  = '#D6EAF8'
RED_LIGHT   = '#FADBD8'
GREEN_LIGHT = '#D5F5E3'
NAVY_LIGHT  = '#D5D8DC'

# ── Canvas ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')
fig.patch.set_facecolor(WHITE)

# ── Helper: rounded box ──────────────────────────────────────────────────
def rounded_box(ax, xy, width, height, facecolor, edgecolor, linewidth=2, alpha=1.0):
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.15",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=linewidth, alpha=alpha,
        transform=ax.transData, zorder=2
    )
    ax.add_patch(box)
    return box

def small_box(ax, xy, width, height, facecolor, edgecolor, linewidth=1.5, alpha=0.95):
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.08",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=linewidth, alpha=alpha,
        transform=ax.transData, zorder=3
    )
    ax.add_patch(box)
    return box

# ── Big flow arrow between panels ────────────────────────────────────────
def flow_arrow(ax, x_start, x_end, y, color, label=None):
    ax.annotate(
        '', xy=(x_end, y), xytext=(x_start, y),
        arrowprops=dict(
            arrowstyle='->,head_width=0.4,head_length=0.25',
            color=color, lw=3.5, shrinkA=0, shrinkB=0
        ),
        zorder=5
    )
    if label:
        mid = (x_start + x_end) / 2
        ax.text(mid, y + 0.2, label, ha='center', va='bottom',
                fontsize=12, color=color, fontfamily='Helvetica',
                fontweight='bold', fontstyle='italic', zorder=6)

# ═══════════════════════════════════════════════════════════════════════════
# PANEL 1 — Standard Theory (left)
# ═══════════════════════════════════════════════════════════════════════════
p1_x, p1_y, p1_w, p1_h = 0.3, 0.5, 3.6, 5.0
rounded_box(ax, (p1_x, p1_y), p1_w, p1_h, BLUE_LIGHT, TERTIARY, linewidth=2.5)

# Panel title
ax.text(p1_x + p1_w/2, p1_y + p1_h - 0.45, 'Standard Theory',
        ha='center', va='center', fontsize=16, fontweight='bold',
        color=TERTIARY, fontfamily='Helvetica', zorder=4)

# Mechanism box
small_box(ax, (p1_x + 0.35, 2.8), 2.9, 1.6, WHITE, TERTIARY, linewidth=1.5)

ax.text(p1_x + p1_w/2, 4.0, r'$\uparrow$ Labor Costs',
        ha='center', va='center', fontsize=15, fontweight='bold',
        color=PRIMARY, fontfamily='Helvetica', zorder=4)

# Down-arrow inside box
ax.annotate(
    '', xy=(p1_x + p1_w/2, 3.25), xytext=(p1_x + p1_w/2, 3.65),
    arrowprops=dict(arrowstyle='->,head_width=0.3,head_length=0.2',
                    color=TERTIARY, lw=2.5),
    zorder=5
)

ax.text(p1_x + p1_w/2, 3.0, r'$\uparrow$ Automation',
        ha='center', va='center', fontsize=15, fontweight='bold',
        color=TERTIARY, fontfamily='Helvetica', zorder=4)

# Citation
ax.text(p1_x + p1_w/2, 1.8, 'Acemoglu & Restrepo',
        ha='center', va='center', fontsize=11, color=GRAY,
        fontfamily='Helvetica', fontstyle='italic', zorder=4)
ax.text(p1_x + p1_w/2, 1.4, '(2019, 2020)',
        ha='center', va='center', fontsize=11, color=GRAY,
        fontfamily='Helvetica', fontstyle='italic', zorder=4)

# Small caption
ax.text(p1_x + p1_w/2, 0.85, '"Substitution logic"',
        ha='center', va='center', fontsize=10, color=GRAY,
        fontfamily='Helvetica', zorder=4)


# ═══════════════════════════════════════════════════════════════════════════
# PANEL 2 — Dual Economy Reality (center)
# ═══════════════════════════════════════════════════════════════════════════
p2_x, p2_y, p2_w, p2_h = 4.7, 0.5, 4.6, 5.0
rounded_box(ax, (p2_x, p2_y), p2_w, p2_h, '#FEF5F5', SECONDARY, linewidth=2.5)

# Panel title
ax.text(p2_x + p2_w/2, p2_y + p2_h - 0.45, 'Dual Economy Reality',
        ha='center', va='center', fontsize=16, fontweight='bold',
        color=PRIMARY, fontfamily='Helvetica', zorder=4)

# Mechanism box — the OPPOSITE finding
small_box(ax, (p2_x + 0.35, 3.15), 3.9, 1.45, WHITE, SECONDARY, linewidth=2)

ax.text(p2_x + p2_w/2, 4.2, r'$\uparrow$ Labor Costs',
        ha='center', va='center', fontsize=15, fontweight='bold',
        color=PRIMARY, fontfamily='Helvetica', zorder=4)

ax.annotate(
    '', xy=(p2_x + p2_w/2, 3.55), xytext=(p2_x + p2_w/2, 3.95),
    arrowprops=dict(arrowstyle='->,head_width=0.3,head_length=0.2',
                    color=SECONDARY, lw=2.5),
    zorder=5
)

ax.text(p2_x + p2_w/2, 3.35, r'$\downarrow$ Investment',
        ha='center', va='center', fontsize=15, fontweight='bold',
        color=SECONDARY, fontfamily='Helvetica', zorder=4)

# Key stat
ax.text(p2_x + p2_w/2, 2.65, r'$\beta$ = −16.7%,  p = 0.045',
        ha='center', va='center', fontsize=13, fontweight='bold',
        color=SECONDARY, fontfamily='Helvetica', zorder=4)

# "The Dual Economy Trap" label
small_box(ax, (p2_x + 0.7, 2.1), 3.2, 0.4, SECONDARY, SECONDARY, linewidth=0, alpha=0.9)
ax.text(p2_x + p2_w/2, 2.3, 'THE DUAL ECONOMY TRAP',
        ha='center', va='center', fontsize=12, fontweight='bold',
        color=WHITE, fontfamily='Helvetica', zorder=4)

# Two channels below
# Channel 1
small_box(ax, (p2_x + 0.25, 1.25), 1.8, 0.65, WHITE, PRIMARY, linewidth=1.2)
ax.text(p2_x + 1.15, 1.72, 'Cash', ha='center', va='center',
        fontsize=11, fontweight='bold', color=PRIMARY, fontfamily='Helvetica', zorder=4)
ax.text(p2_x + 1.15, 1.42, 'constraint', ha='center', va='center',
        fontsize=10, color=PRIMARY, fontfamily='Helvetica', zorder=4)

# Channel 2
small_box(ax, (p2_x + 2.55, 1.25), 1.8, 0.65, WHITE, PRIMARY, linewidth=1.2)
ax.text(p2_x + 3.45, 1.72, 'Informality', ha='center', va='center',
        fontsize=11, fontweight='bold', color=PRIMARY, fontfamily='Helvetica', zorder=4)
ax.text(p2_x + 3.45, 1.42, 'escape valve', ha='center', va='center',
        fontsize=10, color=PRIMARY, fontfamily='Helvetica', zorder=4)

# Sample note
ax.text(p2_x + p2_w/2, 0.82, '66,775 firm-year obs · Colombia 2016–2024',
        ha='center', va='center', fontsize=9, color=GRAY,
        fontfamily='Helvetica', zorder=4)


# ═══════════════════════════════════════════════════════════════════════════
# PANEL 3 — Policy Implication (right)
# ═══════════════════════════════════════════════════════════════════════════
p3_x, p3_y, p3_w, p3_h = 10.1, 0.5, 3.6, 5.0
rounded_box(ax, (p3_x, p3_y), p3_w, p3_h, GREEN_LIGHT, ACCENT, linewidth=2.5)

# Panel title
ax.text(p3_x + p3_w/2, p3_y + p3_h - 0.45, 'Policy Implication',
        ha='center', va='center', fontsize=16, fontweight='bold',
        color=ACCENT, fontfamily='Helvetica', zorder=4)

# Three-step flow inside panel
steps = [
    ('Parafiscal reform', 4.0),
    ('Free capital', 3.2),
    ('Enable automation', 2.4),
]

for i, (label, y_pos) in enumerate(steps):
    small_box(ax, (p3_x + 0.35, y_pos - 0.22), 2.9, 0.55, WHITE, ACCENT, linewidth=1.5)
    ax.text(p3_x + p3_w/2, y_pos + 0.05, label,
            ha='center', va='center', fontsize=13, fontweight='bold',
            color=PRIMARY, fontfamily='Helvetica', zorder=4)

    # Arrow between steps (pointing downward to next step)
    if i < len(steps) - 1:
        next_y = steps[i+1][1]
        ax.annotate(
            '', xy=(p3_x + p3_w/2, next_y + 0.35), xytext=(p3_x + p3_w/2, y_pos - 0.25),
            arrowprops=dict(arrowstyle='->,head_width=0.2,head_length=0.15',
                            color=ACCENT, lw=2),
            zorder=5
        )

# Key stat highlight
small_box(ax, (p3_x + 0.55, 1.25), 2.5, 0.65, ACCENT, ACCENT, linewidth=0, alpha=0.9)
ax.text(p3_x + p3_w/2, 1.57, '540K jobs saved',
        ha='center', va='center', fontsize=14, fontweight='bold',
        color=WHITE, fontfamily='Helvetica', zorder=4)

# Caption
ax.text(p3_x + p3_w/2, 0.82, 'Monte Carlo simulation',
        ha='center', va='center', fontsize=9, color=GRAY,
        fontfamily='Helvetica', fontstyle='italic', zorder=4)


# ═══════════════════════════════════════════════════════════════════════════
# FLOW ARROWS between panels
# ═══════════════════════════════════════════════════════════════════════════
flow_arrow(ax, p1_x + p1_w + 0.05, p2_x - 0.05, 3.0, PRIMARY, 'but in Colombia...')
flow_arrow(ax, p2_x + p2_w + 0.05, p3_x - 0.05, 3.0, ACCENT, 'therefore...')


# ═══════════════════════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════════════════════
plt.tight_layout(pad=0.3)

# Output paths
base = '/Users/cristianespinal/Claude Code/Projects/Research/automatizacion_colombia'
paths_png = [
    os.path.join(base, 'images', 'en', 'graphical_abstract.png'),
    os.path.join(base, 'submission', 'graphical_abstract.png'),
]
path_pdf = os.path.join(base, 'images', 'en', 'graphical_abstract.pdf')

for p in paths_png:
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fig.savefig(p, dpi=300, bbox_inches='tight', facecolor=WHITE, pad_inches=0.15)
    print(f'Saved PNG → {p}')

os.makedirs(os.path.dirname(path_pdf), exist_ok=True)
fig.savefig(path_pdf, bbox_inches='tight', facecolor=WHITE, pad_inches=0.15)
print(f'Saved PDF → {path_pdf}')

plt.close()

# Verify dimensions
from PIL import Image
for p in paths_png:
    img = Image.open(p)
    w, h = img.size
    print(f'  {os.path.basename(p)}: {w} × {h} px  (min required: 1328 × 531)')
    assert w >= 1328, f'Width {w} < 1328'
    assert h >= 531, f'Height {h} < 531'

print('\nDone — graphical abstract generated successfully.')
