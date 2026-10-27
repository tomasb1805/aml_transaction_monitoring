"""
Shared chart styling: one palette and one set of chrome rules for every plot.
Figures are written to reports/output_plots/ (non-interactive backend).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config import PLOTS_DIR


# chart chrome & ink
SURFACE        = '#fcfcfb'
TEXT_PRIMARY   = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
TEXT_MUTED     = '#898781'
GRID           = '#e1e0d9'
AXIS           = '#c3c2b7'

# categorical series, assigned in fixed order
MODEL_COLOUR = '#2a78d6'   # slot 1 — statistical model
RULES_COLOUR = '#eb6834'   # slot 2 — TM rules engine

# status colours, reserved for the triage tiers (always shown with a text label)
TIER_COLOURS = {
    'CLEAR':    '#0ca30c',
    'REVIEW':   '#fab219',
    'ESCALATE': '#d03b3b',
}


def apply_style() -> None:
    plt.rcParams.update({
        'figure.facecolor':   SURFACE,
        'axes.facecolor':     SURFACE,
        'savefig.facecolor':  SURFACE,
        'axes.edgecolor':     AXIS,
        'axes.linewidth':     0.8,
        'axes.labelcolor':    TEXT_SECONDARY,
        'axes.titlecolor':    TEXT_PRIMARY,
        'axes.titlesize':     12,
        'axes.titleweight':   'bold',
        'axes.titlelocation': 'left',
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'axes.grid':          True,
        'axes.axisbelow':     True,
        'grid.color':         GRID,
        'grid.linewidth':     0.6,
        'grid.linestyle':     '-',
        'xtick.color':        TEXT_MUTED,
        'ytick.color':        TEXT_MUTED,
        'text.color':         TEXT_PRIMARY,
        'legend.frameon':     False,
        'legend.labelcolor':  TEXT_SECONDARY,
        'font.family':        'sans-serif',
        'font.size':          10,
        'lines.linewidth':    2,
    })


def save_figure(fig, filename: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot → reports/output_plots/{filename}")
