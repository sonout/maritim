"""Small model-specific adapters used by the shared experiment runner."""

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.baselines import TrAISformer
from models.direct.coordinates import max_step_distance_nmi, physical_coordinate_scale
from models.direct.losses import masked_trajectory_loss
from models.direct.model import DirectTrajectoryPredictor
from models.direct.vessel_history import (
    HISTORY_FEATURE_NAMES,
    VesselHistoryNormalization,
    chronological_support_quintiles,
    prepare_vessel_history_context,
)

from .config import seed_worker
from .data import (
    AutoregressiveTrajectoryDataset,
    DirectForecastDataset,
    autoregressive_collate,
    prepare_forecast_frame,
)
from .interface import ForecastOutput


@dataclass
class PreparedData:
    train_dataset: object
    validation_loss_dataset: object
    validation_model_dataset: object
    validation_targets: DirectForecastDataset
    metadata: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)


@dataclass
class EvaluationData:
    model_dataset: object
    targets: DirectForecastDataset
    metadata: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)


@dataclass
class OptimizationPlan:
    optimizer: torch.optim.Optimizer
    scheduler: object
    scheduler_interval: str


def _loader(dataset, config, *, shuffle, batch_size=None, collate_fn=None):
    return DataLoader(
        dataset,
        batch_size=int(batch_size or config.batch_size),
        shuffle=shuffle,
        num_workers=int(config.num_workers),
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(int(config.seed)),
    )


def _raw_by_id(frame):
    return frame.set_index("TRAJECTORY_ID", verify_integrity=True)


def _history_summary(prepared):
    support = prepared.diagnostics["support"]
    pool = prepared.diagnostics["pool_sizes"][support]
    distance = prepared.raw_features[support, 4]
    return {
        "count": int(support.sum()),
        "fraction": float(support.mean()),
        "pool_size_median": float(np.median(pool)) if len(pool) else None,
        "pool_size_iqr": (
            [float(np.quantile(pool, 0.25)), float(np.quantile(pool, 0.75))]
            if len(pool)
            else None
        ),
        "match_distance_median_nmi": float(np.median(distance)) if len(distance) else None,
        "match_distance_p90_nmi": (
            float(np.quantile(distance, 0.9)) if len(distance) else None
        ),
    }


class DirectAdapter:
    name = "direct"

    def validate(self, config, data_config):
        if config.dataset != getattr(data_config, "name", config.dataset):
            raise ValueError("Model and dataset profiles must match.")
        if int(config.num_modes) != 1 or config.encoder != "transformer":
            raise ValueError("B0/B3A require a deterministic Transformer.")
        for name in ("input_steps", "output_steps", "hidden_dim", "encoder_layers"):
            if int(getattr(config, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        if int(config.attention_heads) <= 0:
            raise ValueError("attention_heads must be positive.")
        if int(config.hidden_dim) % int(config.attention_heads):
            raise ValueError("hidden_dim must be divisible by attention_heads.")
        if float(config.max_speed_knots) <= 0:
            raise ValueError("max_speed_knots must be positive.")
        if float(data_config.cadence_seconds) <= 0:
            raise ValueError("Dataset cadence_seconds must be positive.")
        history_dim = int(getattr(config, "vessel_history_dim", 0))
        if history_dim not in (0, 7):
            raise ValueError("vessel_history_dim must be 0 for B0 or 7 for B3A.")
        if history_dim and int(getattr(config, "vessel_history_hidden_dim", 0)) <= 0:
            raise ValueError("vessel_history_hidden_dim must be positive for B3A.")
        if not data_config.targets_are_normalized:
            raise ValueError("Direct models currently require normalized dataset artifacts.")

    def build_model(self, config, data_config):
        return DirectTrajectoryPredictor(
            input_steps=config.input_steps,
            output_steps=config.output_steps,
            hidden_dim=config.hidden_dim,
            encoder_layers=config.encoder_layers,
            attention_heads=config.attention_heads,
            dropout=config.dropout,
            include_global=config.include_global,
            delta_scale=config.delta_scale,
            vessel_history_dim=int(getattr(config, "vessel_history_dim", 0)),
            vessel_history_hidden_dim=int(getattr(config, "vessel_history_hidden_dim", 32)),
            max_step_nmi=max_step_distance_nmi(
                config.max_speed_knots, data_config.cadence_seconds
            ),
            physical_scale=physical_coordinate_scale(data_config),
        )

    def prepare_training(self, config, data_config, train_raw, valid_raw, *, smoke):
        train_frame = prepare_forecast_frame(
            train_raw, data_config, config.input_steps, config.output_steps
        )
        valid_frame = prepare_forecast_frame(
            valid_raw, data_config, config.input_steps, config.output_steps
        )
        base_train = DirectForecastDataset(
            train_frame, config.input_steps, config.output_steps
        )
        base_valid = DirectForecastDataset(
            valid_frame, config.input_steps, config.output_steps
        )
        train_auxiliary = valid_auxiliary = None
        metadata, artifacts = {}, {}
        if int(getattr(config, "vessel_history_dim", 0)):
            train_context = prepare_vessel_history_context(
                source_dataset=base_train,
                query_dataset=base_train,
                source_raw_by_id=_raw_by_id(train_raw),
                query_raw_by_id=_raw_by_id(train_raw),
                input_steps=config.input_steps,
                physical_scale=physical_coordinate_scale(data_config),
            )
            valid_context = prepare_vessel_history_context(
                source_dataset=base_train,
                query_dataset=base_valid,
                source_raw_by_id=_raw_by_id(train_raw),
                query_raw_by_id=_raw_by_id(valid_raw),
                input_steps=config.input_steps,
                physical_scale=physical_coordinate_scale(data_config),
                normalization=train_context.normalization,
            )
            train_auxiliary = train_context.features
            valid_auxiliary = valid_context.features
            metadata = {
                "feature_names": HISTORY_FEATURE_NAMES,
                "descriptor_steps": [35, 71],
                "heading_sectors": 8,
                "source_boundary": "training split only",
                "strict_chronology": "source 6 h endpoint precedes query origin",
                "train": _history_summary(train_context),
                "validation": _history_summary(valid_context),
                "train_support_by_chronological_quintile": chronological_support_quintiles(
                    train_context.query_origin_timestamps,
                    train_context.diagnostics["support"],
                ),
            }
            artifacts = {
                "train_trajectory_ids": base_train.trajectory_ids,
                "validation_trajectory_ids": base_valid.trajectory_ids,
                "train_raw_features": train_context.raw_features,
                "validation_raw_features": valid_context.raw_features,
                "train_features": train_context.features,
                "validation_features": valid_context.features,
                "train_match_trajectory_ids": train_context.diagnostics[
                    "match_trajectory_ids"
                ],
                "validation_match_trajectory_ids": valid_context.diagnostics[
                    "match_trajectory_ids"
                ],
                "train_pool_sizes": train_context.diagnostics["pool_sizes"],
                "validation_pool_sizes": valid_context.diagnostics["pool_sizes"],
                "normalization_mean": train_context.normalization.mean,
                "normalization_scale": train_context.normalization.scale,
            }
        if smoke:
            train_frame = train_frame.iloc[:64].copy()
            valid_frame = valid_frame.iloc[:32].copy()
            if train_auxiliary is not None:
                train_auxiliary = train_auxiliary[:64]
                valid_auxiliary = valid_auxiliary[:32]
        train = DirectForecastDataset(
            train_frame,
            config.input_steps,
            config.output_steps,
            auxiliary_context=train_auxiliary,
        )
        valid = DirectForecastDataset(
            valid_frame,
            config.input_steps,
            config.output_steps,
            auxiliary_context=valid_auxiliary,
        )
        return PreparedData(train, valid, valid, valid, metadata, artifacts)

    def prepare_evaluation(
        self, config, data_config, train_raw, query_raw, *, normalization=None
    ):
        query_frame = prepare_forecast_frame(
            query_raw, data_config, config.input_steps, config.output_steps
        )
        target = DirectForecastDataset(
            query_frame, config.input_steps, config.output_steps
        )
        auxiliary, metadata, artifacts = None, {}, {}
        if int(getattr(config, "vessel_history_dim", 0)):
            if normalization is None:
                raise ValueError("B3A evaluation requires training normalization.")
            train_frame = prepare_forecast_frame(
                train_raw, data_config, config.input_steps, config.output_steps
            )
            source = DirectForecastDataset(
                train_frame, config.input_steps, config.output_steps
            )
            prepared = prepare_vessel_history_context(
                source_dataset=source,
                query_dataset=target,
                source_raw_by_id=_raw_by_id(train_raw),
                query_raw_by_id=_raw_by_id(query_raw),
                input_steps=config.input_steps,
                physical_scale=physical_coordinate_scale(data_config),
                normalization=normalization,
            )
            auxiliary = prepared.features
            metadata = _history_summary(prepared)
            metadata["source_boundary"] = "training split only"
            artifacts = {
                "trajectory_ids": target.trajectory_ids,
                "raw_features": prepared.raw_features,
                "features": prepared.features,
                "match_trajectory_ids": prepared.diagnostics["match_trajectory_ids"],
                "pool_sizes": prepared.diagnostics["pool_sizes"],
                "normalization_mean": normalization.mean,
                "normalization_scale": normalization.scale,
            }
        model_dataset = DirectForecastDataset(
            query_frame,
            config.input_steps,
            config.output_steps,
            auxiliary_context=auxiliary,
        )
        return EvaluationData(model_dataset, model_dataset, metadata, artifacts)

    def training_loader(self, dataset, config, *, shuffle):
        return _loader(dataset, config, shuffle=shuffle)

    def prediction_loader(self, dataset, config):
        return _loader(dataset, config, shuffle=False)

    def loss(self, model, batch, config, data_config, device):
        expected = 6 if int(getattr(config, "vessel_history_dim", 0)) else 5
        if len(batch) != expected:
            raise ValueError("Direct-model batch does not match configured context.")
        context, targets, masks = (item.to(device) for item in batch[:3])
        auxiliary = batch[5].to(device) if expected == 6 else None
        prediction = model(context, auxiliary)
        scale = physical_coordinate_scale(data_config)
        return masked_trajectory_loss(prediction, targets, masks, scale)

    def optimization(self, model, config, train_loader):
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=config.learning_rate * 0.1
        )
        return OptimizationPlan(optimizer, scheduler, "epoch")

    def predict(self, model, dataset, config, device):
        model.eval()
        trajectories, probabilities, clipping, ids = [], [], [], []
        with torch.no_grad():
            for batch in self.prediction_loader(dataset, config):
                context = batch[0].to(device)
                auxiliary = batch[5].to(device) if len(batch) == 6 else None
                prediction = model(context, auxiliary)
                trajectories.append(prediction.trajectories.cpu())
                probabilities.append(prediction.probabilities.cpu())
                clipping.append(prediction.clip_mask.cpu())
                ids.append(batch[4])
        output = ForecastOutput(
            trajectories=torch.cat(trajectories).numpy(),
            probabilities=torch.cat(probabilities).numpy(),
            clip_mask=torch.cat(clipping).numpy(),
            trajectory_ids=torch.cat(ids).numpy(),
        )
        output.validate(len(dataset), config.output_steps)
        return output


class TrAISformerAdapter:
    name = "traisformer"

    def validate(self, config, data_config):
        if config.dataset != data_config.name:
            raise ValueError("Model and dataset profiles must match.")
        if int(config.training_sequence_len) != int(config.max_seqlen) + 1:
            raise ValueError("Teacher forcing requires max_seqlen + 1 training tokens.")
        if config.crop_policy != "prefix":
            raise ValueError("TrAISformer requires deterministic prefix cropping.")
        if int(config.num_modes) < 1 or float(config.temperature) <= 0:
            raise ValueError("TrAISformer sampling controls are invalid.")
        for name in (
            "input_steps",
            "output_steps",
            "max_seqlen",
            "n_head",
            "n_layer",
            "lat_size",
            "lon_size",
            "sog_size",
            "cog_size",
        ):
            if int(getattr(config, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        embedding_dim = sum(
            int(getattr(config, name))
            for name in ("n_lat_embd", "n_lon_embd", "n_sog_embd", "n_cog_embd")
        )
        if embedding_dim % int(config.n_head):
            raise ValueError("TrAISformer embedding dimension must divide its heads.")
        if not bool(config.sample_predictions) and int(config.num_modes) > 1:
            raise ValueError("Repeated deterministic predictions are not an ensemble.")
        if not data_config.targets_are_normalized:
            raise ValueError("The accepted TrAISformer config requires normalized artifacts.")

    def build_model(self, config, data_config):
        return TrAISformer(config)

    def prepare_training(self, config, data_config, train_raw, valid_raw, *, smoke):
        if smoke:
            train_raw = train_raw.iloc[:64].copy()
            valid_raw = valid_raw.iloc[:32].copy()
            train_raw.attrs["split"] = "train"
            valid_raw.attrs["split"] = "valid"
        train = AutoregressiveTrajectoryDataset(
            train_raw, config, data_config, training=True
        )
        valid_loss = AutoregressiveTrajectoryDataset(
            valid_raw, config, data_config, training=True
        )
        prepared_valid = prepare_forecast_frame(
            valid_raw, data_config, config.input_steps, config.output_steps
        )
        validation_targets = DirectForecastDataset(
            prepared_valid, config.input_steps, config.output_steps
        )
        validation_model = AutoregressiveTrajectoryDataset(
            prepared_valid, config, data_config, training=False
        )
        return PreparedData(train, valid_loss, validation_model, validation_targets)

    def prepare_evaluation(
        self, config, data_config, train_raw, query_raw, *, normalization=None
    ):
        prepared = prepare_forecast_frame(
            query_raw, data_config, config.input_steps, config.output_steps
        )
        targets = DirectForecastDataset(prepared, config.input_steps, config.output_steps)
        model_dataset = AutoregressiveTrajectoryDataset(
            prepared, config, data_config, training=False
        )
        return EvaluationData(model_dataset, targets)

    def training_loader(self, dataset, config, *, shuffle):
        return _loader(
            dataset,
            config,
            shuffle=shuffle,
            collate_fn=autoregressive_collate,
        )

    def prediction_loader(self, dataset, config):
        return _loader(
            dataset,
            config,
            shuffle=False,
            batch_size=config.evaluation_batch_size,
            collate_fn=autoregressive_collate,
        )

    def loss(self, model, batch, config, data_config, device):
        moved = tuple(item.to(device) for item in batch)
        return {"loss": model.training_loss(moved)}

    def optimization(self, model, config, train_loader):
        optimizer = model.traisformer.configure_optimizer(config)
        total_steps = max(1, int(config.epochs) * len(train_loader))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: model.lr_multiplier(step, config.warmup_steps, total_steps),
        )
        return OptimizationPlan(optimizer, scheduler, "step")

    def predict(self, model, dataset, config, device):
        rounds, reference_ids = [], None
        for _ in range(int(config.num_modes)):
            predictions, ids = [], []
            for batch in self.prediction_loader(dataset, config):
                sequence = batch[0].to(device)
                predictions.append(model.rollout(sequence, config.output_steps).cpu())
                ids.append(batch[4])
            round_ids = torch.cat(ids).numpy()
            if reference_ids is None:
                reference_ids = round_ids
            elif not np.array_equal(reference_ids, round_ids):
                raise ValueError("Trajectory ordering changed between sampling rounds.")
            rounds.append(torch.cat(predictions))
        trajectories = torch.stack(rounds, dim=1).numpy()
        examples, modes = trajectories.shape[:2]
        output = ForecastOutput(
            trajectories=trajectories,
            probabilities=np.full((examples, modes), 1.0 / modes, dtype=np.float32),
            clip_mask=np.zeros(trajectories.shape[:3], dtype=bool),
            trajectory_ids=reference_ids,
        )
        output.validate(len(dataset), config.output_steps)
        return output


def build_adapter(config):
    if config.model_family == "direct":
        return DirectAdapter()
    if config.model_family == "traisformer":
        return TrAISformerAdapter()
    raise ValueError(f"Unsupported model_family={config.model_family!r}.")


def load_vessel_history_normalization(path):
    with np.load(path) as saved:
        return VesselHistoryNormalization(
            mean=saved["normalization_mean"], scale=saved["normalization_scale"]
        )
