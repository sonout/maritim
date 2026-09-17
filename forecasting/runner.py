"""One training/evaluation lifecycle shared by every model adapter."""

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from evaluation.metrics import evaluate_weighted_modes

from .adapters import build_adapter, load_vessel_history_normalization
from .config import Config, load_config, seed_everything
from .data import load_raw_split


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n")


def evaluate_output(output, targets, data_config):
    output.validate(len(targets), targets.targets.shape[1])
    if not np.array_equal(output.trajectory_ids, targets.trajectory_ids):
        raise ValueError("Prediction and target trajectory IDs are not aligned.")
    roi_min = np.asarray([data_config.lat_min, data_config.lon_min])
    roi_range = np.asarray(
        [
            data_config.lat_max - data_config.lat_min,
            data_config.lon_max - data_config.lon_min,
        ]
    )
    predictions_deg = output.trajectories * roi_range + roi_min
    targets_deg = targets.targets * roi_range + roi_min
    return evaluate_weighted_modes(
        predictions_deg.transpose(1, 0, 2, 3),
        output.probabilities,
        targets_deg,
        targets.masks,
        cadence_seconds=data_config.cadence_seconds,
        clip_masks=output.clip_mask.transpose(1, 0, 2),
    )


def save_prediction_artifact(path, output, targets, model_family):
    if not np.array_equal(output.trajectory_ids, targets.trajectory_ids):
        raise ValueError("Prediction artifact IDs do not align with targets.")
    np.savez_compressed(
        path,
        artifact_layout=np.asarray("examples,modes,steps,coordinates"),
        model_family=np.asarray(model_family),
        predictions=output.trajectories,
        probabilities=output.probabilities,
        clip_mask=output.clip_mask,
        targets=targets.targets,
        target_mask=targets.masks,
        trajectory_ids=targets.trajectory_ids,
        vessel_ids=targets.vessel_ids,
    )


def _run_epoch(
    adapter,
    model,
    loader,
    config,
    data_config,
    device,
    *,
    optimization=None,
):
    training = optimization is not None
    model.train(training)
    totals, examples = {}, 0
    for batch in loader:
        if training:
            optimization.optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            losses = adapter.loss(model, batch, config, data_config, device)
            if training:
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                optimization.optimizer.step()
                if optimization.scheduler_interval == "step":
                    optimization.scheduler.step()
        count = int(batch[0].shape[0])
        examples += count
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * count
    return {name: value / examples for name, value in totals.items()}


def train_experiment(config_name, run_dir, device, *, seed=None, smoke=False):
    run_dir = Path(run_dir)
    if run_dir.exists():
        raise FileExistsError(f"Refusing to reuse run directory {run_dir}.")
    config = load_config(config_name, "model")
    if seed is not None:
        config.seed = int(seed)
    if smoke:
        config.epochs = 1
        config.num_workers = 0
        # A mechanics check needs one forecast, not a full stochastic ensemble.
        config.num_modes = 1
    config.seed = seed_everything(config.seed)
    data_config = load_config(config.dataset, "dataset")
    adapter = build_adapter(config)
    adapter.validate(config, data_config)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable.")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    train_raw = load_raw_split(data_config, "train")
    valid_raw = load_raw_split(data_config, "valid")
    prepared = adapter.prepare_training(
        config, data_config, train_raw, valid_raw, smoke=smoke
    )
    run_dir.mkdir(parents=True)
    write_json(
        run_dir / "resolved_design.json",
        {
            **config.to_dict(),
            "adapter": adapter.name,
            "prediction_layout": "examples,modes,steps,coordinates",
            "protocol": "shared train-to-validation runner; test not loaded",
            "adapter_metadata": prepared.metadata,
        },
    )
    if prepared.artifacts:
        np.savez_compressed(run_dir / "vessel_history_context.npz", **prepared.artifacts)

    model = adapter.build_model(config, data_config).to(device)
    train_loader = adapter.training_loader(prepared.train_dataset, config, shuffle=True)
    valid_loader = adapter.training_loader(
        prepared.validation_loss_dataset, config, shuffle=False
    )
    optimization = adapter.optimization(model, config, train_loader)
    history, best_state = [], None
    best_loss, best_epoch, stale = float("inf"), -1, 0
    started = time.perf_counter()
    for epoch in range(int(config.epochs)):
        train_losses = _run_epoch(
            adapter,
            model,
            train_loader,
            config,
            data_config,
            device,
            optimization=optimization,
        )
        valid_losses = _run_epoch(
            adapter, model, valid_loader, config, data_config, device
        )
        record = {
            "epoch": epoch,
            "learning_rate": optimization.optimizer.param_groups[0]["lr"],
            "train": train_losses,
            "valid": valid_losses,
        }
        history.append(record)
        print(record)
        if np.isfinite(valid_losses["loss"]) and valid_losses["loss"] < best_loss:
            best_loss = valid_losses["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(
                {name: value.detach().cpu() for name, value in model.state_dict().items()}
            )
            stale = 0
        else:
            stale += 1
        if optimization.scheduler_interval == "epoch":
            optimization.scheduler.step()
        if bool(getattr(config, "early_stopping", True)) and stale >= int(config.patience):
            break
    if best_state is None:
        raise RuntimeError("No finite validation checkpoint was selected.")
    torch.save(best_state, run_dir / "best_validation_state.pt")
    model.load_state_dict(best_state, strict=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_started = time.perf_counter()
    output = adapter.predict(
        model, prepared.validation_model_dataset, config, device
    )
    metrics = evaluate_output(output, prepared.validation_targets, data_config)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_started
    save_prediction_artifact(
        run_dir / "validation_predictions.npz",
        output,
        prepared.validation_targets,
        config.model_family,
    )
    write_json(
        run_dir / "validation_report.json",
        {
            "status": "shared validation-only artifact; test was not loaded",
            "config_name": config_name,
            "seed": config.seed,
            "device": str(device),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "training_seconds": training_seconds,
            "validation_inference_seconds": inference_seconds,
            "best_validation_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "history": history,
            "validation_metrics": metrics,
        },
    )
    return run_dir


def evaluate_experiment(
    config_name,
    output_dir,
    split,
    device,
    *,
    run_dir=None,
    checkpoint=None,
    confirm_frozen_design=False,
):
    if split == "test" and not confirm_frozen_design:
        raise RuntimeError("Test evaluation requires --confirm-frozen-design.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if run_dir is not None:
        run_dir = Path(run_dir)
        design = Path(run_dir) / "resolved_design.json"
        config = Config(json.loads(design.read_text()))
        if config.name != config_name:
            raise ValueError("Run config does not match the requested model.")
        checkpoint = checkpoint or Path(run_dir) / "best_validation_state.pt"
    else:
        config = load_config(config_name, "model")
    if checkpoint is None:
        raise ValueError("Evaluation requires a run directory or explicit checkpoint.")
    config.seed = seed_everything(config.seed)
    data_config = load_config(config.dataset, "dataset")
    adapter = build_adapter(config)
    adapter.validate(config, data_config)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable.")
    train_raw = load_raw_split(data_config, "train")
    query_raw = load_raw_split(data_config, split)
    normalization = None
    if int(getattr(config, "vessel_history_dim", 0)):
        if run_dir is None:
            raise ValueError(
                "B3A evaluation requires its run directory for training-only "
                "vessel-history normalization."
            )
        context_path = Path(run_dir) / "vessel_history_context.npz"
        normalization = load_vessel_history_normalization(context_path)
    prepared = adapter.prepare_evaluation(
        config,
        data_config,
        train_raw,
        query_raw,
        normalization=normalization,
    )
    model = adapter.build_model(config, data_config)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True), strict=True
    )
    model.to(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = adapter.predict(model, prepared.model_dataset, config, device)
    metrics = evaluate_output(output, prepared.targets, data_config)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - started
    metrics.update(
        {
            "inference_seconds": inference_seconds,
            "trajectories_per_second": len(prepared.targets) / inference_seconds,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "peak_gpu_memory_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            ),
            "adapter_metadata": prepared.metadata,
        }
    )
    save_prediction_artifact(
        output_dir / f"{split}_predictions.npz",
        output,
        prepared.targets,
        config.model_family,
    )
    if prepared.artifacts:
        np.savez_compressed(
            output_dir / f"{split}_vessel_history_context.npz", **prepared.artifacts
        )
    write_json(output_dir / f"{split}_metrics.json", metrics)
    return output, metrics
