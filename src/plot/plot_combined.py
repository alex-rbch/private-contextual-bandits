"""Plot privacy-level comparisons above algorithm comparisons in one PNG/PDF.

Run: python -m src.plot configs/paper_grid.json
Only saved selected trajectories are read; no experiments are rerun.
"""

from pathlib import Path
import math

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from ._common import (
    plt, ALGORITHM_COLORS, ALGORITHM_LABELS, ALGORITHM_PANEL_HEIGHT_SCALE,
    LEGEND_FONT_SIZE, ALGORITHM_LEGEND_HANDLELENGTH, PRIVACY_LEGEND_HANDLELENGTH,
    _PrivacyExampleHandler, _algorithm_marker_style, _box_subplot, _curves,
    _entry_regime, _last_power_of_two_indices, _legend_in_row_order,
    _meaningful_common_ylim, _ordered_algorithms, _ordered_regimes,
    _plot_algorithm_markers, _privacy_styles, _regime_legend, _regime_short,
    _regime_title, _save_figure, plot_units,
)


def make_figure(config, selection, output=None, *, entries=None, curve_reader=None):
    """Draw both comparisons from the same selected seed trajectories."""
    entries = selection["selected"] if entries is None else entries
    algorithms = _ordered_algorithms({row["algorithm"] for row in entries})
    regimes = _ordered_regimes(tuple({_entry_regime(row) for row in entries}))
    if not algorithms or not regimes:
        raise ValueError("no supported results to plot")
    styles = _privacy_styles(regimes)
    horizon = int(selection["horizon"])
    if horizon < 1:
        raise ValueError("the horizon must be positive")
    rounds = np.arange(1, horizon + 1)
    marker_indices = _last_power_of_two_indices(horizon)
    uniform_loss = 1.0 - 1.0 / int(selection["num_actions"])
    read_curves = _curves if curve_reader is None else curve_reader
    means, stds = {}, {}
    for entry in entries:
        key = (str(entry["algorithm"]), _entry_regime(entry))
        if key in means:
            raise ValueError("duplicate selected algorithm/privacy setting")
        seed_curves = read_curves(config.output_dir, entry, horizon)
        mean = seed_curves.mean(axis=0)
        if not math.isclose(float(mean[-1]), float(entry["mean_final_pv_loss"]),
                            rel_tol=0, abs_tol=1e-12):
            raise ValueError("curve endpoint disagrees with the selected PV loss")
        means[key] = mean
        stds[key] = (seed_curves.std(axis=0, ddof=1) if len(seed_curves) > 1
                     else np.zeros(horizon))
    common_ylim = _meaningful_common_ylim(list(means.values()), (uniform_loss,), horizon)
    best_non_supervised = {
        regime: min((algorithm for algorithm in algorithms
                     if algorithm != "supervised" and (algorithm, regime) in means),
                    key=lambda algorithm: means[algorithm, regime][-1], default=None)
        for regime in regimes
    }

    # Preserve the paper layout's physical panel heights and legend/row gaps.
    columns = min(4, len(algorithms))
    rows = math.ceil(len(algorithms) / columns)
    original_panel_height = 12.8 * (0.565 - 0.105) / (2 + 0.29)
    algorithm_height = original_panel_height * ALGORITHM_PANEL_HEIGHT_SCALE
    row_gap = original_panel_height * 0.29
    header_space, privacy_height = 12.8 * 0.075, 12.8 * 0.23
    section_gap, bottom_space = 12.8 * 0.13, 12.8 * 0.105
    algorithm_section = rows * algorithm_height + (rows - 1) * row_gap
    height = header_space + privacy_height + section_gap + algorithm_section + bottom_space
    width = max(7.0, 4.6 * len(regimes), 5.3 * columns)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11}):
        figure = plt.figure(figsize=(width, height))
        privacy_top = 1 - header_space / height
        privacy_bottom = privacy_top - privacy_height / height
        algorithm_top = privacy_bottom - section_gap / height
        top_grid = figure.add_gridspec(
            1, len(regimes), left=0.055, right=0.99,
            top=privacy_top, bottom=privacy_bottom, wspace=0.08,
        )
        bottom_grid = figure.add_gridspec(
            rows, columns, left=0.055, right=0.99,
            top=algorithm_top, bottom=bottom_space / height,
            hspace=row_gap / algorithm_height, wspace=0.10,
        )

        def format_axis(axis, show_ylabels, show_xlabels=True):
            axis.set_xlim(1, max(2, horizon))
            axis.set_ylim(*common_ylim)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            if show_xlabels:
                axis.set_xlabel("Round")
            else:
                axis.tick_params(labelbottom=False)
            if show_ylabels:
                axis.set_ylabel("Progressive-validation loss")
            else:
                axis.tick_params(labelleft=False)
            axis.grid(alpha=0.2)
            _box_subplot(axis)
            axis.axhline(uniform_loss, color="black", linestyle="--", linewidth=1.1,
                         label=f"Uniform ({uniform_loss:.3f})")

        for column, regime in enumerate(regimes):
            axis = figure.add_subplot(top_grid[0, column])
            for algorithm in algorithms:
                key = (algorithm, regime)
                if key not in means:
                    continue
                mean, std = means[key], stds[key]
                color = ALGORITHM_COLORS[algorithm]
                axis.plot(rounds, mean, color=color, linestyle="-", linewidth=1.9,
                          label=f"{ALGORITHM_LABELS[algorithm]} ({mean[-1]:.3f})")
                axis.fill_between(rounds, np.maximum(mean - std, 0),
                                  np.minimum(mean + std, 1), color=color,
                                  alpha=0.055, linewidth=0, rasterized=True)
                _plot_algorithm_markers(axis, rounds, mean, algorithm, marker_indices)
            axis.set_title(_regime_title(regime), fontsize=12)
            format_axis(axis, show_ylabels=column == 0)

        color_handles = [
            Line2D([0], [0], color=ALGORITHM_COLORS[algorithm], linestyle="-",
                   linewidth=1.9, label=ALGORITHM_LABELS[algorithm],
                   **_algorithm_marker_style(algorithm))
            for algorithm in algorithms
        ]
        color_handles.append(Line2D([0], [0], color="black", linestyle="--",
                                   linewidth=1.1, label=f"Uniform ({uniform_loss:.3f})"))
        legend_fontsize = 11
        length_scale = LEGEND_FONT_SIZE / legend_fontsize
        legend_columns = min(len(color_handles), max(1, int(width / 2.3)))
        figure.legend(
            handles=_legend_in_row_order(color_handles, legend_columns),
            loc="center", bbox_to_anchor=(0.5, privacy_bottom - 12.8 * 0.07 / height),
            ncol=legend_columns, frameon=False, fontsize=legend_fontsize,
            handlelength=ALGORITHM_LEGEND_HANDLELENGTH * length_scale, columnspacing=1.8,
        )

        for index, algorithm in enumerate(algorithms):
            row, column = divmod(index, columns)
            axis = figure.add_subplot(bottom_grid[row, column])
            terminal_parts = []
            for regime in regimes:
                if (algorithm, regime) not in means:
                    continue
                mean = means[algorithm, regime]
                axis.plot(rounds, mean, color=ALGORITHM_COLORS[algorithm],
                          linestyle=styles[regime], linewidth=2.1,
                          label=_regime_legend(regime))
                _plot_algorithm_markers(axis, rounds, mean, algorithm, marker_indices)
                value_text = f"{mean[-1]:.3f}"
                if best_non_supervised[regime] == algorithm:
                    value_text = rf"$\mathbf{{{value_text}}}$"
                terminal_parts.append(f"{_regime_short(regime)}: {value_text}")
            terminal_text = "{" + ", ".join(reversed(terminal_parts)) + "}"
            axis.set_title(f"{ALGORITHM_LABELS[algorithm]}\n{terminal_text}", fontsize=11)
            format_axis(axis, show_ylabels=column == 0,
                        show_xlabels=index + columns >= len(algorithms))

        privacy_handles = [
            Line2D([0], [0], color="black", linestyle=styles[regime], linewidth=1.4,
                   label=_regime_legend(regime))
            for regime in reversed(regimes)
        ]
        figure.legend(
            handles=privacy_handles, loc="center", bbox_to_anchor=(0.5, 12.8 * 0.035 / height),
            handler_map={Line2D: _PrivacyExampleHandler()}, ncol=len(privacy_handles),
            frameon=False, fontsize=legend_fontsize,
            handlelength=PRIVACY_LEGEND_HANDLELENGTH * length_scale, columnspacing=2.0,
        )
        figure.suptitle(
            f"OpenML {config.dataset_id} | T = {horizon:,} | "
            f"A = {int(selection['num_actions'])} | {len(selection['seeds'])} seeds",
            fontsize=17, y=1 - 12.8 * 0.015 / height,
        )
        destination = (Path(output) if output is not None else
                       config.output_dir / "plots" / f"privacy_alg_{config.dataset_id}.png")
        return _save_figure(figure, destination)


def make_plots(config_path):
    return [make_figure(config, selection, entries=entries)
            for config, selection, entries in plot_units(config_path)]
