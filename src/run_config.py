"""Run any experiment config through the unchanged src.run.run_setting function.

Each task is one dataset, loss, privacy regime, algorithm, and seed, including
one fixed choice or the complete regularization/policy grid.
Scores and loss trajectories share one atomic NPZ archive. Completed configs
select the best loss and parameters per privacy setting/algorithm by mean final
classification loss over seeds (equivalently, maximum classification accuracy).
Only data and summaries are saved; plotting is a separate command.
Selection and reporting use the same seeds, so these are benchmark results,
not held-out estimates or DP releases.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import fcntl
import multiprocessing
import csv
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

if __name__ in {"__main__", "__mp_main__"}:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"

import numpy as np

from .data import LoadedDataset, load_dataset
from .data.loader import DEFAULT_DATASETS_DIR
from .data.prepare import prepare_input
from .policies import PolicyConfiguration, fastcb_supports_loss
from .run import run_setting
from .plot.reporting import summary_row_order


POLICY_FIELDS = {
    "supervised": frozenset(),
    "linucb": frozenset({"ucb_scale"}),
    "regcb": frozenset({"width_scale"}),
    "adacb": frozenset({"gamma_scale", "gamma_exponent", "width_scale"}),
    "squarecb": frozenset({"gamma_scale", "gamma_exponent"}),
    "fastcb": frozenset({"gamma_scale", "gamma_exponent"}),
    "vpo": frozenset({"learning_rate_scale"}),
    "bpo": frozenset({"learning_rate_scale", "gamma_scale"}),
}


@dataclass(frozen=True)
class PrivacyRegime:
    epsilon: float | None
    delta: float


def privacy_stem(regime: PrivacyRegime) -> str:
    """Stable filename component for one privacy pair."""

    if regime.epsilon is None:
        return "nonprivate"
    return f"eps{regime.epsilon:g}_delta{regime.delta:g}"


def results_base(path: Path | None = None) -> Path:
    """Use the same result location in direct Python commands and job scripts."""

    configured_root = os.environ.get("DP_CMAB_RESULTS_ROOT")
    if configured_root:
        return Path(configured_root).expanduser().resolve()
    scratch = os.environ.get("SCRATCH")
    if scratch:
        return (Path(scratch).expanduser() / "dp-cmab" / "results").resolve()
    if path is None:
        return Path(__file__).resolve().parents[1] / "results"
    source = Path(path).resolve()
    parent = source.parent.parent if source.parent.name == "configs" else source.parent
    return (parent / "results").resolve()


def study_results_root(path: Path) -> Path:
    """Return the active output directory for one named study config."""

    return results_base(path) / Path(path).stem


@dataclass(frozen=True)
class RunConfig:
    dataset_id: int
    input_npz: Path | None
    datasets_dir: Path
    output_dir: Path
    horizon: int | None
    losses: tuple[str, ...]
    privacy: tuple[PrivacyRegime, ...]
    seeds: tuple[int, ...]
    ridge_lambdas: tuple[float, ...]
    policies: dict[str, tuple[PolicyConfiguration, ...]]
    use_bias: bool
    project_unitball: bool
    fixed_settings: tuple[dict, ...] = ()


@dataclass(frozen=True)
class Task:
    index: int
    loss: str
    epsilon: float | None
    delta: float
    seed: int
    algorithm: str


_WORKER_CONFIG: RunConfig | None = None
_WORKER_DATASET: LoadedDataset | None = None


def _initialize_worker(config: RunConfig) -> None:
    global _WORKER_CONFIG, _WORKER_DATASET
    _WORKER_CONFIG = config
    _WORKER_DATASET = load_dataset(
        config.dataset_id,
        datasets_dir=config.datasets_dir,
        input_path=config.input_npz,
        use_bias=config.use_bias,
        project_unitball=config.project_unitball,
    )


def _run_worker(task: Task) -> str:
    if _WORKER_CONFIG is None or _WORKER_DATASET is None:
        raise RuntimeError("config worker was not initialized")
    return str(run_task(_WORKER_CONFIG, task, _WORKER_DATASET))


def _unique(values: list[Any], name: str) -> None:
    if not values or len(values) != len({json.dumps(v, sort_keys=True) for v in values}):
        raise ValueError(f"{name} must be nonempty and contain no duplicates")


def _parse_privacy(raw: dict[str, Any]) -> tuple[PrivacyRegime, ...]:
    rows = raw["privacy"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("privacy must be a nonempty JSON list")
    parsed = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"epsilon", "delta"}:
            raise ValueError("each privacy row needs epsilon and delta")
        epsilon_value = row["epsilon"]
        delta_value = row["delta"]
        if isinstance(epsilon_value, str) and epsilon_value.lower() in {"inf", "infinity"}:
            epsilon = None
        elif isinstance(epsilon_value, bool):
            raise ValueError("epsilon must be positive or 'inf'")
        else:
            epsilon = float(epsilon_value)
            if not math.isfinite(epsilon) or epsilon <= 0:
                raise ValueError("epsilon must be positive and finite or 'inf'")
        if isinstance(delta_value, bool):
            raise ValueError("delta must be numeric")
        delta = float(delta_value)
        if epsilon is None and delta != 0:
            raise ValueError("non-private privacy pair must be (inf, 0)")
        if epsilon is not None and (not math.isfinite(delta) or not 0 < delta < 1):
            raise ValueError("private delta must lie in (0, 1)")
        parsed.append(PrivacyRegime(epsilon, delta))
    if len(set(parsed)) != len(parsed):
        raise ValueError("privacy pairs must be unique")
    return tuple(parsed)


def _parse_config(source: Path, raw: dict[str, Any]) -> RunConfig:
    """Validate one dataset's grid; paths are relative to the source JSON."""

    if not isinstance(raw, dict):
        raise ValueError("grid config must be a JSON object")
    allowed = {
        "dataset_id", "input_npz", "datasets_dir", "output_dir", "horizon", "losses",
        "privacy", "seeds", "use_bias", "project_unitball", "grid",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown grid config keys: {sorted(unknown)}")
    dataset_id = int(raw["dataset_id"])
    if dataset_id <= 0:
        raise ValueError("dataset_id must be positive")
    horizon = raw.get("horizon")
    if horizon is not None:
        horizon = int(horizon)
        if horizon <= 0:
            raise ValueError("horizon must be positive")
    losses = raw.get("losses", ["quadratic"])
    privacy = _parse_privacy(raw)
    seeds = raw.get("seeds", [1])
    grid = raw["grid"]
    if not isinstance(grid, dict) or set(grid) != {"ridge_lambda", "policies"}:
        raise ValueError("grid must have ridge_lambda and policies")
    for name, values in (
        ("losses", losses), ("seeds", seeds),
        ("grid.ridge_lambda", grid["ridge_lambda"]),
    ):
        if not isinstance(values, list):
            raise ValueError(f"{name} must be a JSON list")
        _unique(values, name)
    if any(loss not in {"quadratic", "logistic"} for loss in losses):
        raise ValueError("losses must be quadratic or logistic")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("seeds must be nonnegative integers")
    ridge_lambdas = tuple(float(value) for value in grid["ridge_lambda"])
    if any(not math.isfinite(value) or value <= 0 for value in ridge_lambdas):
        raise ValueError("ridge_lambda values must be finite and positive")
    policy_grid = grid["policies"]
    if not isinstance(policy_grid, dict) or not policy_grid:
        raise ValueError("grid.policies must be a nonempty object")
    policies: dict[str, tuple[PolicyConfiguration, ...]] = {}
    for algorithm, parameters in policy_grid.items():
        if algorithm not in POLICY_FIELDS:
            raise ValueError(f"unsupported algorithm {algorithm!r}")
        if not isinstance(parameters, dict):
            raise ValueError(f"policy grid for {algorithm} must be an object")
        unexpected = set(parameters) - POLICY_FIELDS[algorithm]
        if unexpected:
            raise ValueError(f"unused parameters for {algorithm}: {sorted(unexpected)}")
        for field, values in parameters.items():
            if not isinstance(values, list):
                raise ValueError(f"{algorithm}.{field} must be a JSON list")
            _unique(values, f"{algorithm}.{field}")
        names = tuple(sorted(parameters))
        combinations = itertools.product(*(parameters[name] for name in names))
        policies[algorithm] = tuple(
            PolicyConfiguration(algorithm, **dict(zip(names, values)))
            for values in combinations
        )
    base = source.parent
    input_npz = raw.get("input_npz")
    datasets_dir = raw.get("datasets_dir")
    output_dir = raw.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError("output_dir must be a path string")
    if input_npz is not None and not isinstance(input_npz, str):
        raise ValueError("input_npz must be a path string")
    if datasets_dir is not None and not isinstance(datasets_dir, str):
        raise ValueError("datasets_dir must be a path string")
    if not isinstance(raw.get("use_bias", True), bool) or not isinstance(
        raw.get("project_unitball", True), bool
    ):
        raise ValueError("preprocessing switches must be booleans")
    return RunConfig(
        dataset_id=dataset_id,
        input_npz=None if input_npz is None else (base / input_npz).resolve(),
        datasets_dir=DEFAULT_DATASETS_DIR if datasets_dir is None else (base / datasets_dir).resolve(),
        output_dir=(base / output_dir).resolve(),
        horizon=horizon,
        losses=tuple(losses),
        privacy=privacy,
        seeds=tuple(seeds),
        ridge_lambdas=ridge_lambdas,
        policies=policies,
        use_bias=raw.get("use_bias", True),
        project_unitball=raw.get("project_unitball", True),
    )


def study_payload(path: Path) -> dict:
    """Read a self-contained study config."""

    source = Path(path).resolve()
    raw = json.loads(source.read_text())
    if not isinstance(raw, dict):
        raise ValueError("study config must be a JSON object")
    return raw


def load_configs(path: Path) -> tuple[RunConfig, ...]:
    """Normalize a fixed-setting or grid config into dataset units."""

    source = Path(path).resolve()
    raw = study_payload(source)
    if raw.get("format") == "paper-fixed-v1":
        return _fixed_configs(source, raw)
    allowed = {
        "datasets", "datasets_dir", "horizon", "losses", "privacy",
        "seeds", "grid", "use_bias", "project_unitball",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown study config keys: {sorted(unknown)}")
    missing = {"datasets", "privacy", "seeds", "grid"} - set(raw)
    if missing:
        raise ValueError(f"missing study config keys: {sorted(missing)}")
    datasets = raw["datasets"]
    if not isinstance(datasets, list):
        raise ValueError("datasets must be a JSON list")
    _unique(datasets, "datasets")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in datasets):
        raise ValueError("dataset IDs must be positive integers")
    losses = raw.get("losses", ["quadratic", "logistic"])
    if not isinstance(losses, list):
        raise ValueError("losses must be a JSON list")
    _unique(losses, "losses")
    if any(value not in {"quadratic", "logistic"} for value in losses):
        raise ValueError("losses must be quadratic or logistic")
    root = study_results_root(source)
    configs = []
    for selected_id in datasets:
        child = {
            "dataset_id": selected_id,
            "output_dir": str(root / f"openml_{selected_id}"),
            "losses": losses,
            "privacy": raw["privacy"],
            "seeds": raw["seeds"],
            "grid": raw["grid"],
        }
        for optional in ("datasets_dir", "horizon", "use_bias", "project_unitball"):
            if optional in raw:
                child[optional] = raw[optional]
        configs.append(_parse_config(source, child))
    return tuple(configs)


def _validate_fixed(raw):
    if raw.get("format") != "paper-fixed-v1":
        raise ValueError("expected a paper-fixed-v1 configuration")
    seeds = raw["seeds"]
    if (not seeds or len(seeds) != len(set(seeds))
            or any(type(seed) is not int or seed < 0 for seed in seeds)):
        raise ValueError("seeds must be distinct nonnegative integers")
    if set(raw["preprocessing"]) != {"use_bias", "project_unitball"} or any(
        type(value) is not bool for value in raw["preprocessing"].values()
    ):
        raise ValueError("preprocessing requires two boolean switches")
    seen_datasets = set()
    for dataset in raw["datasets"]:
        unknown = set(dataset) - {"id", "settings", "horizon"}
        if unknown:
            raise ValueError(f"unknown dataset config keys: {sorted(unknown)}")
        dataset_id = dataset["id"]
        if type(dataset_id) is not int or dataset_id <= 0 or dataset_id in seen_datasets:
            raise ValueError("dataset IDs must be distinct positive integers")
        seen_datasets.add(dataset_id)
        horizon = dataset.get("horizon")
        if horizon is not None and (type(horizon) is not int or horizon <= 0):
            raise ValueError("horizon must be a positive integer")
        seen_settings = set()
        for setting in dataset["settings"]:
            policy = PolicyConfiguration(**setting["policy"])
            if policy.algorithm != setting["algorithm"]:
                raise ValueError("policy/algorithm mismatch")
            if setting["loss"] not in ("quadratic", "logistic"):
                raise ValueError("unknown loss")
            ridge = setting["ridge_lambda"]
            if not math.isfinite(ridge) or ridge <= 0:
                raise ValueError("ridge must be positive and finite")
            epsilon, delta = setting["epsilon"], setting["delta"]
            if epsilon is None:
                if delta != 0:
                    raise ValueError("nonprivate delta must be zero")
            elif not math.isfinite(epsilon) or epsilon <= 0 or not 0 < delta < 1:
                raise ValueError("invalid privacy pair")
            key = (setting["loss"], epsilon, delta, policy.algorithm)
            if key in seen_settings:
                raise ValueError("only ONE fixed setting per dataset/loss/privacy/algorithm")
            seen_settings.add(key)
        if not seen_settings:
            raise ValueError("dataset has no settings")
    if not seen_datasets:
        raise ValueError("configuration has no datasets")
    return raw


def _fixed_configs(source, raw):
    _validate_fixed(raw)
    base = study_results_root(source)
    directory = (source.parent / raw["datasets_dir"]).resolve() if "datasets_dir" in raw else DEFAULT_DATASETS_DIR
    configs = []
    for dataset in raw["datasets"]:
        settings = tuple(dataset["settings"])
        losses = tuple(sorted({row["loss"] for row in settings}))
        policies = {}
        for row in settings:
            policy = PolicyConfiguration(**row["policy"])
            policies.setdefault(row["algorithm"], [])
            if policy not in policies[row["algorithm"]]:
                policies[row["algorithm"]].append(policy)
        output = base / f"openml_{dataset['id']}"
        configs.append(RunConfig(
            dataset_id=dataset["id"], input_npz=None,
            datasets_dir=directory, output_dir=output,
            horizon=dataset.get("horizon"), losses=losses,
            privacy=tuple(dict.fromkeys(PrivacyRegime(row["epsilon"], row["delta"]) for row in settings)),
            seeds=tuple(raw["seeds"]),
            ridge_lambdas=tuple(sorted({row["ridge_lambda"] for row in settings})),
            policies={name: tuple(values) for name, values in policies.items()},
            **raw["preprocessing"], fixed_settings=settings,
        ))
    return tuple(configs)


def candidate_settings(config, task):
    if config.fixed_settings:
        return tuple((row["ridge_lambda"], PolicyConfiguration(**row["policy"]))
                     for row in config.fixed_settings
                     if (row["loss"], row["epsilon"], row["delta"], row["algorithm"])
                     == (task.loss, task.epsilon, task.delta, task.algorithm))
    return tuple(itertools.product(config.ridge_lambdas, config.policies[task.algorithm]))


def tasks(config: RunConfig) -> tuple[Task, ...]:
    combinations = itertools.product(config.losses, config.privacy, config.policies, config.seeds)
    planned = []
    for loss, regime, algorithm, seed in combinations:
        task = Task(len(planned), loss, regime.epsilon, regime.delta, seed, algorithm)
        if not config.fixed_settings or candidate_settings(config, task):
            planned.append(task)
    return tuple(planned)


def task_path(config: RunConfig, task: Task) -> Path:
    regime = privacy_stem(PrivacyRegime(task.epsilon, task.delta))
    directory = config.output_dir / "data" / task.loss
    return directory / regime / task.algorithm / f"seed_{task.seed}.npz"


def task_is_supported(task: Task) -> bool:
    return task.algorithm != "fastcb" or fastcb_supports_loss(task.loss)


def read_task(config: RunConfig, task: Task, *, verify_losses: bool = False) -> dict:
    """Read the current storage format, rejecting incompatible or damaged output."""

    path = task_path(config, task)
    try:
        with np.load(path, allow_pickle=False) as archive:
            payload = json.loads(str(archive["metadata"].item()))
            if (payload.get("format") != "sweep_v3"
                    or payload.get("task") != asdict(task)
                    or payload.get("dataset_id") != config.dataset_id):
                raise ValueError("task identity mismatch")
            preprocessing = {"use_bias": config.use_bias, "project_unitball": config.project_unitball}
            if payload.get("preprocessing", preprocessing) != preprocessing:
                raise ValueError("preprocessing settings mismatch")
            expected = [
                (ridge, asdict(policy))
                for ridge, policy in candidate_settings(config, task)
            ]
            actual = [(row["ridge_lambda"], row["policy"]) for row in payload["candidates"]]
            if actual != expected or "losses" not in archive:
                raise ValueError("incomplete tuning grid")
            unsupported = payload.get("status") == "unsupported"
            if payload.get("status", "completed") not in {"completed", "unsupported"}:
                raise ValueError("unknown task status")
            if unsupported and task_is_supported(task):
                raise ValueError("unsupported status on a supported algorithm/loss")
            for row in payload["candidates"]:
                if unsupported:
                    if row["final_pv_loss"] != -1 or row["cumulative_loss"] != -payload["horizon"]:
                        raise ValueError("invalid unsupported loss sentinel")
                elif not 0 <= row["final_pv_loss"] <= 1:
                    raise ValueError("PV loss must lie in [0, 1]")
            if verify_losses:
                losses = archive["losses"]
                if (losses.dtype != (np.int8 if unsupported else np.uint8)
                        or losses.shape != (len(expected), payload["horizon"])
                        or (np.any(losses != -1) if unsupported else np.any(losses > 1))):
                    raise ValueError("invalid loss trajectories")
                for row, curve in zip(payload["candidates"], losses):
                    total = int(curve.sum())
                    if (row["cumulative_loss"] != total
                            or row["final_pv_loss"] != total / payload["horizon"]):
                        raise ValueError("scores disagree with trajectories")
        return payload
    except (OSError, EOFError, BadZipFile, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid task archive {path}: {error}") from error


def run_task(
    config: RunConfig,
    task: Task,
    dataset: LoadedDataset,
) -> Path:
    output = task_path(config, task)
    horizon = len(dataset.labels) if config.horizon is None else config.horizon
    if output.exists():
        saved = read_task(config, task, verify_losses=True)
        if saved["horizon"] != horizon:
            raise ValueError(f"horizon mismatch at {output}")
        return output
    candidates: list[dict[str, object]] = []
    curves: list[np.ndarray] = []
    unsupported = not task_is_supported(task)
    for ridge_lambda, policy in candidate_settings(config, task):
        result = run_setting(
            dataset,
            policy=policy,
            loss=task.loss,
            ridge_lambda=ridge_lambda,
            epsilon=task.epsilon,
            delta=task.delta,
            seed=task.seed,
            horizon=config.horizon,
        )
        run = result["algorithms"][task.algorithm]
        losses = np.asarray(run["observed_losses"], dtype=np.float64)
        if ((run.get("status") == "unsupported") != unsupported
                or losses.shape != (horizon,)
                or not np.all(np.isin(losses, (-1.0,) if unsupported else (0.0, 1.0)))):
            raise AssertionError("expected one binary loss per round")
        candidates.append({
            "ridge_lambda": ridge_lambda,
            "policy": asdict(policy),
            "cumulative_loss": float(run["cumulative_loss"]),
            "final_pv_loss": float(run["cumulative_loss"]) / horizon,
        })
        curves.append(losses.astype(np.int8 if unsupported else np.uint8))
    payload = {
        "format": "sweep_v3",
        "preprocessing": {"use_bias": config.use_bias, "project_unitball": config.project_unitball},
        "dataset_id": config.dataset_id,
        "horizon": horizon,
        "task": asdict(task),
        "candidates": candidates,
        "benchmark_only_not_dp_release": task.epsilon is not None,
    }
    if unsupported:
        payload.update(status="unsupported", reason=run["reason"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle, losses=np.stack(curves),
            metadata=np.asarray(json.dumps(payload, sort_keys=True)),
        )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output)
    return output


def select(config: RunConfig, dataset: LoadedDataset) -> Path:
    """Jointly select oracle loss and parameters by mean final classification loss."""

    grouped: dict[tuple[float | None, float, str, str], list[dict[str, object]]] = {}
    for task in tasks(config):
        path = task_path(config, task)
        payload = read_task(config, task)
        for row_index, candidate in enumerate(payload["candidates"]):
            setting = {
                "loss": task.loss,
                "ridge_lambda": candidate["ridge_lambda"],
                "policy": candidate["policy"],
            }
            setting_key = json.dumps(setting, sort_keys=True, separators=(",", ":"))
            key = (task.epsilon, task.delta, task.algorithm, setting_key)
            grouped.setdefault(key, []).append({
                "seed": task.seed,
                "archive": path.relative_to(config.output_dir).as_posix(),
                "row_index": row_index,
                # Historical quadratic FastCB archives remain readable, but
                # this unavailable combination cannot enter model selection.
                "final_pv_loss": candidate["final_pv_loss"] if task_is_supported(task) else -1.0,
            })
    best = {scope: {} for scope in (*config.losses, "best")}
    for (epsilon, delta, algorithm, setting_key), rows in grouped.items():
        if {row["seed"] for row in rows} != set(config.seeds) or len(rows) != len(config.seeds):
            raise ValueError("a candidate is missing one or more seed runs")
        mean_loss = float(np.mean([row["final_pv_loss"] for row in rows]))
        group = (epsilon, delta, algorithm)
        choice = (mean_loss, setting_key, sorted(rows, key=lambda row: row["seed"]))
        for scope in (json.loads(setting_key)["loss"], "best"):
            if scope == "best" and not 0 <= mean_loss <= 1:
                continue
            if group not in best[scope] or choice[:2] < best[scope][group][:2]:
                best[scope][group] = choice
    selections = {}
    for scope, choices in best.items():
        selected = []
        for (epsilon, delta, algorithm), (mean_loss, setting_key, rows) in choices.items():
            setting = json.loads(setting_key)
            selected.append({
                "loss": setting["loss"],
                "epsilon": epsilon,
                "delta": delta,
                "algorithm": algorithm,
                "ridge_lambda": setting["ridge_lambda"],
                "policy": setting["policy"],
                "mean_final_pv_loss": mean_loss,
                "seed_runs": rows,
            })
            if mean_loss == -1:
                selected[-1].update(status="unsupported", reason="FastCB only supports logistic loss")
        selected.sort(key=lambda row: (row["loss"], row["epsilon"] is not None,
                                       -(row["epsilon"] or 0), row["delta"], row["algorithm"]))
        selections[scope] = selected
    unsupported_selections = [row for scope, rows in selections.items() if scope != "best"
                              for row in rows if row.get("status") == "unsupported"]
    output = config.output_dir / "selection.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "settings": execution_settings(config),
        "dataset_id": config.dataset_id,
        "num_actions": dataset.num_actions,
        "horizon": len(dataset.labels) if config.horizon is None else config.horizon,
        "seeds": list(config.seeds),
        "reporting_source": ("frozen_settings_replayed_without_retuning" if config.fixed_settings and len(config.fixed_settings) == len(selections["best"])
                             else "same_seed_grid_selection_not_held_out"),
        "selected": selections["best"],
        "unsupported": unsupported_selections,
    }, indent=2, sort_keys=True) + "\n")
    tables_dir = config.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    parameter_names = sorted(set().union(*(POLICY_FIELDS[name] for name in config.policies)))
    columns = ["dataset_id", "algorithm", "loss", "epsilon", "delta", "ridge_lambda", *parameter_names,
               "mean_final_pv_loss", "status"]
    for scope, selected in selections.items():
        if scope == "best":
            available = {(row["epsilon"], row["delta"], row["algorithm"]) for row in selected}
            selected = selected + [row for row in unsupported_selections
                                  if (row["epsilon"], row["delta"], row["algorithm"]) not in available]
        table_path = tables_dir / f"final_losses_{scope}.csv"
        temporary = table_path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in sorted(selected, key=summary_row_order):
                policy = row["policy"]
                writer.writerow({
                    "dataset_id": config.dataset_id,
                    "algorithm": row["algorithm"],
                    "loss": row["loss"],
                    "epsilon": "inf" if row["epsilon"] is None else row["epsilon"],
                    "delta": row["delta"],
                    "ridge_lambda": row["ridge_lambda"],
                    **{name: policy[name] for name in POLICY_FIELDS[row["algorithm"]]},
                    "mean_final_pv_loss": row["mean_final_pv_loss"],
                    "status": row.get("status", "completed"),
                })
        temporary.replace(table_path)
    return output



def execution_settings(config):
    """Plain experiment settings for resuming and plotting saved results."""
    settings = {
        "dataset_id": config.dataset_id, "horizon": config.horizon,
        "seeds": list(config.seeds),
        "preprocessing": {"use_bias": config.use_bias, "project_unitball": config.project_unitball},
    }
    if config.fixed_settings:
        settings["settings"] = list(config.fixed_settings)
    else:
        settings.update(
            losses=list(config.losses), privacy=[asdict(regime) for regime in config.privacy],
            ridge_lambdas=list(config.ridge_lambdas),
            policies={name: [asdict(policy) for policy in policies]
                      for name, policies in config.policies.items()},
        )
    return settings


def _check_execution(config):
    """Keep different experiment settings out of the same result directory."""
    identity = execution_settings(config)
    path = config.output_dir / "execution.json"
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError("existing results use different experiment settings; use a new config name")
    else:
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)


def run_config(config_path, workers=1):
    """Run all configured settings and save data and summaries, without plotting."""
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    outputs = []
    for config in load_configs(Path(config_path)):
        planned = tasks(config)
        prepare_input(config.dataset_id, config.datasets_dir)
        dataset = load_dataset(config.dataset_id, datasets_dir=config.datasets_dir,
                               input_path=config.input_npz, use_bias=config.use_bias,
                               project_unitball=config.project_unitball)
        if config.horizon is not None and config.horizon > len(dataset.labels):
            raise ValueError(f"horizon exceeds OpenML {config.dataset_id} row count")
        config.output_dir.mkdir(parents=True, exist_ok=True)
        with (config.output_dir / ".run.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError(f"another process is writing {config.output_dir}") from error
            _check_execution(config)
            if workers > 1:
                with ProcessPoolExecutor(max_workers=workers,
                        mp_context=multiprocessing.get_context("spawn"),
                        initializer=_initialize_worker, initargs=(config,)) as pool:
                    for output in pool.map(_run_worker, planned):
                        print(output, flush=True)
            else:
                for task in planned:
                    print(run_task(config, task, dataset), flush=True)
            outputs.append(select(config, dataset))
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--workers", type=int, default=1, help="parallel worker processes")
    args = parser.parse_args()
    try:
        for output in run_config(args.config, args.workers):
            print(output, flush=True)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
