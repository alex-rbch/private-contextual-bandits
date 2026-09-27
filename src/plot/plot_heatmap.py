"""Plot pairwise win rates from a CSV, saving PNG and PDF beside it.

Run: python -m src.plot.plot_heatmap
Rebuild from a completed campaign: python -m src.plot.plot_heatmap --from-results results
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from ._common import plt, _box_subplot, _save_figure
from .reporting import ALGORITHM_ORDER, ALGORITHM_LABELS
from matplotlib.colors import LinearSegmentedColormap, Normalize

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / "results/precomputed/win_rates.csv"
ALGORITHMS = ("bpo",) + tuple(
    name for name in ALGORITHM_ORDER if name not in {"bpo", "supervised"}
)
REGIMES = ((math.inf, 0.0), (8.0, 1e-5), (4.0, 1e-5), (2.0, 1e-5), (1.0, 1e-5))
TIE_ATOL = 1e-12
FIELDS = ("epsilon", "delta", "row_algorithm", "column_algorithm", "datasets",
          "wins", "ties", "losses", "win_rate_percent")


def load_rates(path: Path) -> np.ndarray:
    """Validate a complete five-panel CSV and restore its ordered matrices."""
    shape = (len(REGIMES), len(ALGORITHMS), len(ALGORITHMS))
    rates = np.full(shape, np.nan)
    wins, ties, losses = (np.zeros(shape, dtype=int) for _ in range(3))
    dataset_count = None
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(FIELDS).issubset(reader.fieldnames or ()):
            raise ValueError(f"Missing win-rate columns in {path}")
        for row in reader:
            p = REGIMES.index((float(row["epsilon"]), float(row["delta"])))
            i = ALGORITHMS.index(row["row_algorithm"])
            j = ALGORITHMS.index(row["column_algorithm"])
            count = int(row["datasets"])
            w, t, lost = (int(row[name]) for name in ("wins", "ties", "losses"))
            rate = float(row["win_rate_percent"])
            if dataset_count is None:
                dataset_count = count
            if (count <= 0 or count != dataset_count or min(w, t, lost) < 0
                    or w + t + lost != count or not math.isfinite(rate)
                    or not 0 <= rate <= 100
                    or not math.isclose(rate, 100 * (w + 0.5 * t) / count,
                                        rel_tol=0, abs_tol=1e-10)
                    or not np.isnan(rates[p, i, j])):
                raise ValueError(f"Invalid or duplicate win-rate cell in {path}: {row}")
            rates[p, i, j] = rate
            wins[p, i, j], ties[p, i, j], losses[p, i, j] = w, t, lost
    if np.isnan(rates).any():
        raise ValueError(f"Expected all 245 algorithm/privacy cells in {path}")
    if (not np.array_equal(wins, losses.transpose(0, 2, 1))
            or not np.array_equal(ties, ties.transpose(0, 2, 1))
            or not np.all(np.diagonal(ties, axis1=1, axis2=2) == dataset_count)):
        raise ValueError(f"Inconsistent pairwise counts in {path}")
    return rates


def campaign_scores(results_dir: Path) -> np.ndarray:
    """Select loss-specific winners from all 100 datasets' saved summaries.

    Each summary already selects hyperparameters using the mean final PV loss
    over ten seeds. FastCB uses logistic only; other methods take the lower
    score across logistic and quadratic. No datasets or raw runs are loaded.
    """
    groups = {}
    for index in range(1, 6):
        group = f"campaign100_{index:02d}"
        config = json.loads((ROOT / "configs" / f"{group}.json").read_text())
        groups[group] = config["datasets"]
    ids = [dataset_id for values in groups.values() for dataset_id in values]
    if len(ids) != 100 or len(set(ids)) != 100:
        raise ValueError("Campaign configs must contain exactly 100 distinct datasets")
    expected = {(eps, delta, alg) for eps, delta in REGIMES for alg in ALGORITHMS}
    scores = []
    for group, dataset_ids in groups.items():
        for dataset_id in dataset_ids:
            tables = {}
            for loss in ("logistic", "quadratic"):
                path = (results_dir / group / f"openml_{dataset_id}" / "tables"
                        / f"final_losses_{loss}.csv")
                table = {}
                with path.open(newline="") as handle:
                    for row in csv.DictReader(handle):
                        if row["algorithm"] == "supervised":
                            continue
                        key = (float(row["epsilon"]), float(row["delta"]), row["algorithm"])
                        score = float(row["mean_final_pv_loss"])
                        unsupported = (row["algorithm"] == "fastcb"
                                       and loss == "quadratic" and score == -1)
                        if (int(row["dataset_id"]) != dataset_id or row["loss"] != loss
                                or key not in expected or key in table
                                or not math.isfinite(score)
                                or (not unsupported and (not 0 <= score <= 1
                                    or row.get("status", "completed") != "completed"))):
                            raise ValueError(f"Invalid or duplicate campaign score in {path}: {row}")
                        table[key] = score
                if set(table) != expected:
                    raise ValueError(f"Incomplete campaign summary: {path}")
                tables[loss] = table
            scores.append([
                [min(tables[loss][(eps, delta, algorithm)]
                     for loss in (("logistic",) if algorithm == "fastcb"
                                  else ("logistic", "quadratic")))
                 for algorithm in ALGORITHMS]
                for eps, delta in REGIMES
            ])
    return np.asarray(scores).transpose(1, 0, 2)


def write_campaign_rates(results_dir: Path, output: Path) -> Path:
    """Aggregate a complete campaign into the same CSV format as the bundle."""
    scores = campaign_scores(results_dir)
    difference = scores[:, :, :, None] - scores[:, :, None, :]
    ties = (np.abs(difference) <= TIE_ATOL).sum(axis=1)
    wins = (difference < -TIE_ATOL).sum(axis=1)
    count = scores.shape[1]
    rates = 100 * (wins + 0.5 * ties) / count
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        for p, (eps, delta) in enumerate(REGIMES):
            for i, row_algorithm in enumerate(ALGORITHMS):
                for j, column_algorithm in enumerate(ALGORITHMS):
                    writer.writerow((eps, delta, row_algorithm, column_algorithm, count,
                                     wins[p, i, j], ties[p, i, j],
                                     count - wins[p, i, j] - ties[p, i, j],
                                     f"{rates[p, i, j]:.1f}"))
    return output


def make_plot(csv_path: Path = DEFAULT_CSV) -> Path:
    """Read only the win-rate CSV and write matching PNG/PDF files beside it."""
    csv_path = Path(csv_path)
    rates = load_rates(csv_path)
    labels = tuple(ALGORITHM_LABELS[name] for name in ALGORITHMS)
    n = len(ALGORITHMS)
    cmap = LinearSegmentedColormap.from_list(
        "pairwise", ["#9e3d23", "#e5a57e", "#faf7f2", "#85bbb8", "#136b70"], N=256)
    cmap.set_bad("#eeeeed")
    norm = Normalize(0, 100)
    titles = ["Non-Private"] + [rf"$\varepsilon={eps:g}$" for eps, _ in REGIMES[1:]]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "savefig.facecolor": "white"}):
        figure, axes = plt.subplots(3, 2, figsize=(9.6, 14.4), sharey=True)
        figure.subplots_adjust(left=0.13, right=0.98, bottom=0.07, top=0.96,
                               wspace=0.20, hspace=0.46)
        # Non-Private and the color scale, then 8/4 and 2/1.
        panel_axes = [axes[0, 0], axes[1, 0], axes[1, 1], axes[2, 0], axes[2, 1]]
        for index, (axis, matrix, title) in enumerate(zip(panel_axes, rates, titles)):
            masked = np.ma.array(matrix, mask=np.eye(n, dtype=bool))
            mesh = axis.pcolormesh(masked, cmap=cmap, norm=norm,
                                   edgecolors="white", linewidth=0.9)
            axis.set(xlim=(0, n), ylim=(n, 0), aspect="equal")
            axis.set_title(title, fontsize=15, fontweight="semibold", pad=14)
            axis.set_xticks(np.arange(n) + 0.5, labels, rotation=45,
                           ha="right", rotation_mode="anchor")
            axis.set_yticks(np.arange(n) + 0.5, labels)
            axis.tick_params(axis="both", length=0, pad=6, labelsize=10)
            axis.tick_params(axis="y", labelleft=index in (0, 1, 3))
            _box_subplot(axis)
            for i in range(n):
                for j in range(n):
                    value = matrix[i, j]
                    color = ("#82888b" if i == j else
                             "white" if value < 18 or value > 82 else "#14292c")
                    axis.text(j + 0.5, i + 0.5, "-" if i == j else f"{value:.1f}",
                              ha="center", va="center", fontsize=9.6, color=color)
        axes[0, 1].set_box_aspect(1)
        axes[0, 1].set_axis_off()
        cax = axes[0, 1].inset_axes([0.4325, 0.05, 0.11, 0.90])
        colorbar = figure.colorbar(mesh, cax=cax, orientation="vertical",
                                  ticks=[0, 25, 50, 75, 100])
        colorbar.set_label("Win rate (%)", fontsize=15, labelpad=10)
        colorbar.outline.set_visible(False)
        colorbar.ax.tick_params(length=0, labelsize=9)
        return _save_figure(figure, csv_path.with_suffix(".png"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path,
                        help="Input CSV (default: results/precomputed/win_rates.csv); "
                             "with --from-results, the output CSV instead")
    parser.add_argument("--from-results", type=Path, metavar="DIR",
                        help="Rebuild from DIR/campaign100_01 through campaign100_05; "
                             "default output: DIR/win_rates.csv")
    args = parser.parse_args()
    csv_path = args.csv or (args.from_results / "win_rates.csv"
                            if args.from_results is not None else DEFAULT_CSV)
    try:
        if args.from_results is not None:
            print(write_campaign_rates(args.from_results, csv_path))
        output = make_plot(csv_path)
        print(output)
        print(output.with_suffix(".pdf"))
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
