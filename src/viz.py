"""Shared plotting style for the evaluation scripts.

Colour is assigned by the job it does, not by taste:

  * Categorical (identity: which strategy / which series) uses the first three
    slots of the validated reference palette, in fixed order. Those three are
    documented as clearing the all-pairs colour-vision gates in both modes
    (CVD dE 9.2, normal-vision dE 24.0 on the light surface), which matters
    because some of these figures are scatters rather than lines.
  * Sequential (magnitude: a probability heatmap) uses ONE hue, light to dark.
    Not a rainbow -- a rainbow invents ordering that the data does not have and
    hides real structure behind hue changes.
  * Reference lines (break-even, perfect calibration) are neutral grey. They are
    annotation, not a series, and must not compete for identity.

Every figure carries a legend when it draws two or more series, and values are
printed in text ink rather than in the series colour.
"""

import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots 1-3 of the reference palette, in fixed order.
BLUE = '#2a78d6'
ORANGE = '#eb6834'
AQUA = '#1baf7a'

# Roles. Colour follows the entity, so these never get reassigned.
C_ACTOR = BLUE
C_ORACLE = ORANGE
C_MARKET = AQUA

INK = '#0b0b0b'
INK_SOFT = '#52514e'
GRID = '#d9d8d4'
REFERENCE = '#8a8984'
SURFACE = '#fcfcfb'

# Single-hue sequential ramp, light -> dark, ending on the categorical blue.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    'mono_blue', ['#f4f8fd', '#c3ddf5', '#7fb4e8', '#4a90dc', BLUE, '#1b4f8f']
)


def use_style():
    """Recessive axes and grid; the data should be the only assertive thing."""
    plt.rcParams.update({
        'figure.facecolor': SURFACE,
        'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE,
        'axes.edgecolor': GRID,
        'axes.labelcolor': INK_SOFT,
        'axes.titlecolor': INK,
        'axes.titlesize': 12,
        # DejaVu Sans (the matplotlib default) has no 'medium' weight; asking for
        # one logs a findfont warning on every figure and falls back anyway.
        'axes.titleweight': 'normal',
        'axes.labelsize': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.color': GRID,
        'grid.linewidth': 0.7,
        'grid.alpha': 0.7,
        'xtick.color': INK_SOFT,
        'ytick.color': INK_SOFT,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'legend.frameon': False,
        'lines.linewidth': 2.0,
        'font.size': 10,
        'figure.dpi': 110,
    })


def band(ax, x, lo, mid, hi, color, label, linestyle='-'):
    """Median line with an interquartile band.

    A mean line alone hides how wide the outcome distribution is, and with
    hundreds of sequences the spread is the finding.
    """
    ax.fill_between(x, lo, hi, color=color, alpha=0.16, linewidth=0)
    ax.plot(x, mid, color=color, label=label, linestyle=linestyle)


def annotate_last(ax, x, y, text, color):
    """Direct label at the end of a line, so identity is not colour-alone."""
    ax.annotate(text, xy=(x[-1], y[-1]), xytext=(4, 0), textcoords='offset points',
                color=INK_SOFT, fontsize=9, va='center')


def heatmap(ax, matrix, extent, label, vmax=None):
    im = ax.imshow(matrix, aspect='auto', cmap=SEQUENTIAL, origin='lower',
                   extent=extent, vmin=0.0, vmax=vmax)
    ax.grid(False)
    cbar = ax.figure.colorbar(im, ax=ax, orientation='vertical', pad=0.008)
    cbar.set_label(label, fontsize=9, color=INK_SOFT)
    cbar.ax.tick_params(labelsize=8, colors=INK_SOFT)
    cbar.outline.set_visible(False)
    return im


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def iqr_bands(curves):
    """(p25, median, p75) down the sequence axis."""
    return (np.quantile(curves, 0.25, axis=0),
            np.median(curves, axis=0),
            np.quantile(curves, 0.75, axis=0))
