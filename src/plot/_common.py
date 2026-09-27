"""Shared styles and result loading for experiment and paper figures."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.lines import Line2D
import numpy as np

from ..data import load_dataset
from ..protocol import stable_seed
from .reporting import ALGORITHM_ORDER as ALGORITHM_PANEL_ORDER, ALGORITHM_LABELS
from ..run_config import (
    PrivacyRegime,
    RunConfig,
    load_configs,
    execution_settings,
    task_is_supported,
    tasks,
)
from ..policies import fastcb_supports_loss


ALGORITHM_COLORS = {
    "supervised": "#2ca02c",
    "bpo": "#e377c2",
    "vpo": "#d62728",
    "linucb": "#17becf",
    "regcb": "#8c564b",
    "squarecb": "#1f77b4",
    "adacb": "#9467bd",
    "fastcb": "#ff7f0e",
}
ALGORITHM_MARKERS = {
    "supervised": "*", "bpo": "o", "vpo": "s", "linucb": "^",
    "regcb": "d", "squarecb": "v", "adacb": "P", "fastcb": "X",
}
MARKER_SIZE = 3.4
MARKER_EDGE_WIDTH = 0.25
MARKER_CLEARANCE = 0.9
ALGORITHM_PANEL_HEIGHT_SCALE = 0.7
LEGEND_FONT_SIZE = 8.8
ALGORITHM_LEGEND_HANDLELENGTH = 2.2
PRIVACY_LEGEND_HANDLELENGTH = 4.07
# Match the marker silhouettes in the 3x3 paper figure.
MARKER_SIZE_SCALE = {"*": 1.45, "^": 1.18, "d": 1.24, "v": 1.18, "P": 1.21, "X": 1.18}


def _last_power_of_two_indices(horizon: int) -> list[int]:
    """Return indices for the last four power-of-two rounds within the horizon."""
    last_exponent = int(horizon).bit_length() - 1
    return [(1 << exponent) - 1
            for exponent in range(max(0, last_exponent - 3), last_exponent + 1)]


def _algorithm_marker_style(algorithm: str, background="white") -> dict:
    marker = ALGORITHM_MARKERS[algorithm]
    color = ALGORITHM_COLORS[algorithm]
    return {
        "marker": marker,
        "markersize": MARKER_SIZE * MARKER_SIZE_SCALE.get(marker, 1.0),
        "markerfacecolor": color, "markeredgecolor": color,
        "markeredgewidth": MARKER_EDGE_WIDTH,
        "path_effects": [
            path_effects.Stroke(linewidth=MARKER_EDGE_WIDTH + 2 * MARKER_CLEARANCE,
                                foreground=background),
            path_effects.Normal(),
        ],
    }


def _plot_algorithm_markers(axis, rounds, mean, algorithm, indices):
    """Overlay symbols so their white clearance does not outline the curves."""
    axis.plot(
        rounds[indices], mean[indices], linestyle="None",
        color=ALGORITHM_COLORS[algorithm], label="_nolegend_",
        zorder={"vpo": 31, "bpo": 32}.get(algorithm, 22),
        **_algorithm_marker_style(algorithm, axis.get_facecolor()),
    )

# Non-private, epsilon 8, 4, 2, 1: match the 3x3 paper figure.
PRIVACY_LINESTYLES = (
    (0, (1, 2)),
    (0, (6, 2, 1, 2, 1, 2)),
    (0, (6, 2, 1, 2)),
    (0, (6, 2)),
    "-",
)


class _PrivacyExampleHandler(HandlerLine2D):
    """Use the same physical privacy-line samples as the 3x3 paper figure."""

    def create_artists(self, legend, original_handle, xdescent, ydescent,
                       width, height, fontsize, transform):
        fraction = float(getattr(original_handle, "_example_fraction", 0.6825))
        sample_width = width * fraction
        start = xdescent + (width - sample_width) / 2
        line = Line2D([start, start + sample_width], [height / 2, height / 2])
        self.update_prop(line, original_handle, legend)
        line.set_transform(transform)
        return [line]


def _privacy_styles(regimes):
    """Keep standard epsilon styles stable even when only a subset is present."""
    if len(regimes) > len(PRIVACY_LINESTYLES):
        raise ValueError("at most five privacy regimes are supported")
    standard = dict(zip((None, 8.0, 4.0, 2.0, 1.0), PRIVACY_LINESTYLES))
    styles = {regime: standard[regime.epsilon] for regime in regimes
              if regime.epsilon in standard}
    available = iter(style for style in PRIVACY_LINESTYLES if style not in styles.values())
    for regime in regimes:
        if regime not in styles:
            styles[regime] = next(available)
    return styles

def _entry_regime(entry: dict[str, object]) -> PrivacyRegime:
    return PrivacyRegime(entry["epsilon"], float(entry["delta"]))


def _ordered_regimes(regimes: tuple[PrivacyRegime, ...]) -> list[PrivacyRegime]:
    return sorted(
        regimes,
        key=lambda value: (
            value.epsilon is not None,
            -(value.epsilon or 0),
            value.delta,
        ),
    )


def _ordered_algorithms(algorithms: object) -> list[str]:
    available = set(algorithms)
    unknown = available - set(ALGORITHM_PANEL_ORDER)
    if unknown:
        raise ValueError(f"unknown algorithms: {sorted(unknown)}")
    return [name for name in ALGORITHM_PANEL_ORDER if name in available]


def _legend_in_row_order(handles, columns):
    """Compensate for Matplotlib filling multirow legends down each column."""
    return [handle for column in range(columns) for handle in handles[column::columns]]


def _delta_math(delta: float) -> str:
    exponent = int(round(math.log10(delta)))
    if math.isclose(delta, 10.0**exponent, rel_tol=1e-12, abs_tol=0.0):
        return rf"10^{{{exponent}}}"
    return f"{delta:g}"


def _regime_title(regime: PrivacyRegime) -> str:
    if regime.epsilon is None:
        return "Non-Private"
    return rf"$(\varepsilon,\delta) = ({regime.epsilon:g}, {_delta_math(regime.delta)})$"


def _regime_legend(regime: PrivacyRegime) -> str:
    if regime.epsilon is None:
        return "Non-Private"
    return rf"$\varepsilon={regime.epsilon:g}$"


def _regime_short(regime: PrivacyRegime) -> str:
    return "np" if regime.epsilon is None else f"{regime.epsilon:g}"


def _title_lines(
    config: RunConfig,
    selection: dict[str, object],
) -> tuple[str, str]:
    run_count = len(selection["seeds"])
    first = (
        f"OpenML {config.dataset_id} ({run_count} runs) — "
        "Latest-batch oracle (SSP)"
    )
    lambda_text = ", ".join(f"{value:g}" for value in config.ridge_lambdas)
    second = (
        rf"$T={int(selection['horizon'])}$, $q=2$, "
        rf"$\lambda\in\{{{lambda_text}\}}$; without replacement"
    )
    if len(config.losses) > 1:
        second += "; best oracle loss per algorithm/privacy setting"
    return first, second


def _curves(
    root: Path,
    selected: dict[str, object],
    horizon: int,
) -> np.ndarray:
    curves = []
    for item in selected["seed_runs"]:
        path = root / item["archive"]
        with np.load(path, allow_pickle=False) as archive:
            losses = np.asarray(
                archive["losses"][item["row_index"]], dtype=np.float64
            )
        if losses.shape != (horizon,) or not np.all(np.isin(losses, (0, 1))):
            raise ValueError(f"wrong curve length in {path}")
        curves.append(np.cumsum(losses) / np.arange(1, horizon + 1))
    if not curves:
        raise ValueError("a selected setting has no seed runs")
    return np.stack(curves)


def _best_action_loss(
    config: RunConfig,
    selection: dict[str, object],
) -> float:
    dataset = load_dataset(
        config.dataset_id,
        datasets_dir=config.datasets_dir,
        input_path=config.input_npz,
        use_bias=config.use_bias,
        project_unitball=config.project_unitball,
    )
    horizon = int(selection["horizon"])
    counts = np.zeros(dataset.num_actions, dtype=np.int64)
    for seed in selection["seeds"]:
        permutation_seed = stable_seed(int(seed), "permutation", config.dataset_id)
        indices = np.random.default_rng(permutation_seed).permutation(
            len(dataset.labels)
        )[:horizon]
        counts += np.bincount(
            dataset.labels[indices], minlength=dataset.num_actions
        )
    total = horizon * len(selection["seeds"])
    return 1.0 - float(counts.max()) / float(total)


def _meaningful_common_ylim(
    curves: list[np.ndarray],
    reference_values: tuple[float, ...],
    horizon: int,
) -> tuple[float, float]:
    """Choose common bounds without allowing startup spikes to dominate."""

    first_scale_round = min(max(1, horizon // 20), horizon - 1)
    values = [np.asarray(curve)[first_scale_round:] for curve in curves]
    values.append(np.asarray(reference_values, dtype=np.float64))
    finite = np.concatenate(values)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("cannot choose y-axis bounds without finite values")
    value_min = float(finite.min())
    value_max = float(finite.max())
    span = max(0.05, value_max - value_min)
    padding = max(0.015, 0.08 * span)
    tick = 0.05
    lower = max(0.0, tick * math.floor((value_min - padding) / tick))
    upper = min(1.0, tick * math.ceil((value_max + padding) / tick))
    if upper <= lower:
        upper = min(1.0, lower + tick)
    return lower, upper


def _policy_parameters(algorithm: str, policy: dict[str, object]) -> str:
    if algorithm == "supervised":
        return "none"
    if algorithm == "vpo":
        return f"eta0={float(policy['learning_rate_scale']):g}"
    if algorithm == "bpo":
        return (
            f"eta={float(policy['learning_rate_scale']):g}; "
            f"gamma={float(policy['gamma_scale']):g}"
        )
    if algorithm in {"squarecb", "fastcb"}:
        return (
            f"gamma={float(policy['gamma_scale']):g}; "
            f"rho={float(policy['gamma_exponent']):g}"
        )
    if algorithm == "adacb":
        return (
            f"gamma={float(policy['gamma_scale']):g}; "
            f"rho={float(policy['gamma_exponent']):g}; "
            f"width_scale={float(policy['width_scale']):g}"
        )
    if algorithm == "regcb":
        return f"width_scale={float(policy['width_scale']):g}"
    if algorithm == "linucb":
        return f"ucb_scale={float(policy['ucb_scale']):g}"
    raise ValueError(f"unknown algorithm: {algorithm}")


def _box_subplot(axis: plt.Axes) -> None:
    """Give every plotted panel the same thin rectangular border."""
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#606060")
        spine.set_linewidth(0.7)


def _save_figure(figure: plt.Figure, output: Path) -> Path:
    """Save a 300-dpi PNG and a compact PDF with 150-dpi rasterized artists.

    Curves, markers, and text remain vector graphics in the PDF; only artists
    explicitly marked rasterized (such as uncertainty bands) use pixels.
    """

    output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42,
                         "pdf.compression": 9}):
        for suffix, dpi in ((".png", 300), (".pdf", 150)):
            figure.savefig(output.with_suffix(suffix), dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_units(config_path):
    for config in load_configs(Path(config_path)):
        path = config.output_dir / "selection.json"
        selection = json.loads(path.read_text())
        if (selection.get("settings", execution_settings(config)) != execution_settings(config)
                or selection.get("dataset_id") != config.dataset_id
                or set(selection.get("seeds", ())) != set(config.seeds)
                or (config.horizon is not None and selection.get("horizon") != config.horizon)):
            raise ValueError(f"selection does not match config: {path}")
        entries = [row for row in selection["selected"]
                   if row["algorithm"] != "fastcb" or fastcb_supports_loss(row["loss"])]
        groups = [(row["epsilon"], row["delta"], row["algorithm"]) for row in entries]
        expected = {(task.epsilon, task.delta, task.algorithm) for task in tasks(config)
                    if task_is_supported(task)}
        if len(groups) != len(set(groups)) or set(groups) != expected:
            raise ValueError("selection must have one winner per algorithm/privacy setting")
        for row in entries:
            if not 0 <= float(row["mean_final_pv_loss"]) <= 1:
                raise ValueError("selected PV loss must lie in [0, 1]")
            if row["loss"] not in config.losses:
                raise ValueError("selected oracle loss is not in the config")
            seeds = [run["seed"] for run in row["seed_runs"]]
            if len(seeds) != len(config.seeds) or set(seeds) != set(config.seeds):
                raise ValueError("selection has missing or duplicate seeds")
            for run in row["seed_runs"]:
                archive = Path(run["archive"])
                if archive.is_absolute() or ".." in archive.parts:
                    raise ValueError("trajectory references must stay inside the result directory")
        if not entries:
            continue
        yield config, selection, entries


def plot_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("config", type=Path)
    return parser
