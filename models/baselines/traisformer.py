# Copyright 2021, Duong Nguyen
#
# Licensed under the CECILL-C License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.cecill.info
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from CIA-Oceanix/TrAISformer (models.py and trainers.py).
# Transformer components build on minGPT, Copyright (c) 2020 Andrej Karpathy.
# See the combined LICENSE file for CeCILL-C terms and minGPT MIT notices.
# Local modifications: plain-PyTorch wrapper, corrected blur loss, shared
# adapter integration, configuration validation, and checkpoint compatibility.
"""Corrected TrAISformer reference implemented without Lightning."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _top_k_logits(logits, k):
    values, _ = torch.topk(logits, k)
    output = logits.clone()
    output[output < values[:, [-1]]] = -float("inf")
    return output


def _nearest_logits(logits, current_indices, vicinity):
    indices = torch.arange(logits.shape[-1], device=logits.device).repeat(
        logits.shape[0], 1
    )
    output = logits.clone()
    output[torch.abs(indices - current_indices) >= vicinity / 2] = -float("inf")
    return output


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.n_embd % config.n_head:
            raise ValueError("TrAISformer embedding dimension must divide its heads.")
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.max_seqlen, config.max_seqlen)).view(
                1, 1, config.max_seqlen, config.max_seqlen
            ),
        )
        self.n_head = config.n_head

    def forward(self, values):
        batch, steps, channels = values.shape
        keys = self.key(values).view(
            batch, steps, self.n_head, channels // self.n_head
        ).transpose(1, 2)
        queries = self.query(values).view(
            batch, steps, self.n_head, channels // self.n_head
        ).transpose(1, 2)
        projected_values = self.value(values).view(
            batch, steps, self.n_head, channels // self.n_head
        ).transpose(1, 2)
        attention = (queries @ keys.transpose(-2, -1)) * (
            1.0 / math.sqrt(keys.shape[-1])
        )
        attention = attention.masked_fill(
            self.mask[:, :, :steps, :steps] == 0, float("-inf")
        )
        attention = self.attn_drop(F.softmax(attention, dim=-1))
        output = attention @ projected_values
        output = output.transpose(1, 2).contiguous().view(batch, steps, channels)
        return self.resid_drop(self.proj(output))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, values):
        values = values + self.attn(self.ln1(values))
        return values + self.mlp(self.ln2(values))


class TrAISformerModel(nn.Module):
    """Original four-hot causal Transformer with corrected blur loss."""

    def __init__(self, config):
        super().__init__()
        self.lat_size = config.lat_size
        self.lon_size = config.lon_size
        self.sog_size = config.sog_size
        self.cog_size = config.cog_size
        self.full_size = sum(
            (config.lat_size, config.lon_size, config.sog_size, config.cog_size)
        )
        self.n_lat_embd = config.n_lat_embd
        self.n_lon_embd = config.n_lon_embd
        self.n_sog_embd = config.n_sog_embd
        self.n_cog_embd = config.n_cog_embd
        self.register_buffer(
            "att_sizes",
            torch.tensor(
                [config.lat_size, config.lon_size, config.sog_size, config.cog_size]
            ),
        )
        self.register_buffer(
            "emb_sizes",
            torch.tensor(
                [config.n_lat_embd, config.n_lon_embd, config.n_sog_embd, config.n_cog_embd]
            ),
        )
        self.partition_mode = getattr(config, "partition_mode", "uniform")
        self.blur = bool(config.blur)
        self.blur_loss_w = float(config.blur_loss_w)
        self.blur_n = int(config.blur_n)
        if self.blur:
            self.blur_module = nn.Conv1d(1, 1, 3, padding=0, groups=1, bias=False)
            if not config.blur_learnable:
                self.blur_module.weight.requires_grad = False
                self.blur_module.weight.data.fill_(1 / 3)
        else:
            self.blur_module = None
        self.mode = getattr(config, "mode", "pos")
        self.lat_emb = nn.Embedding(self.lat_size, config.n_lat_embd)
        self.lon_emb = nn.Embedding(self.lon_size, config.n_lon_embd)
        self.sog_emb = nn.Embedding(self.sog_size, config.n_sog_embd)
        self.cog_emb = nn.Embedding(self.cog_size, config.n_cog_embd)
        config.n_embd = sum(
            (config.n_lat_embd, config.n_lon_embd, config.n_sog_embd, config.n_cog_embd)
        )
        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_seqlen, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        output_dim = config.n_embd if self.mode in ("mlp_pos", "mlp") else self.full_size
        self.head = nn.Linear(config.n_embd, output_dim, bias=False)
        self.max_seqlen = config.max_seqlen
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _blur_probs(self, probabilities):
        if self.blur_module is None:
            return probabilities
        padded = torch.cat(
            (probabilities[..., :1], probabilities, probabilities[..., -1:]), dim=-1
        )
        return self.blur_module(padded)

    def to_indexes(self, values):
        indices = (values * self.att_sizes).long()
        return indices, indices

    def forward(self, values, masks=None, *, with_targets=False):
        indices, uniform_indices = self.to_indexes(values)
        if with_targets:
            inputs = indices[:, :-1].contiguous()
            targets = indices[:, 1:].contiguous()
        else:
            inputs = indices
            targets = None
        batch, steps, _ = inputs.shape
        if steps > self.max_seqlen:
            raise ValueError("TrAISformer context exceeds max_seqlen.")
        embeddings = torch.cat(
            (
                self.lat_emb(inputs[:, :, 0]),
                self.lon_emb(inputs[:, :, 1]),
                self.sog_emb(inputs[:, :, 2]),
                self.cog_emb(inputs[:, :, 3]),
            ),
            dim=-1,
        )
        features = self.drop(embeddings + self.pos_emb[:, :steps])
        logits = self.head(self.ln_f(self.blocks(features)))
        if targets is None:
            return logits, None
        sizes = (self.lat_size, self.lon_size, self.sog_size, self.cog_size)
        component_logits = torch.split(logits, sizes, dim=-1)
        losses = [
            F.cross_entropy(
                component.reshape(-1, size),
                targets[:, :, index].reshape(-1),
                reduction="none",
            ).view(batch, steps)
            for index, (component, size) in enumerate(zip(component_logits, sizes))
        ]
        if self.blur:
            probabilities = [F.softmax(component, dim=-1) for component in component_logits]
            for _ in range(self.blur_n):
                blurred = [
                    self._blur_probs(probability.reshape(-1, 1, size)).reshape(
                        probability.shape
                    )
                    for probability, size in zip(probabilities, sizes)
                ]
                for index, (probability, size) in enumerate(zip(blurred, sizes)):
                    losses[index] = losses[index] + self.blur_loss_w * F.nll_loss(
                        torch.log(probability.reshape(-1, size) + 1e-9),
                        targets[:, :, index].reshape(-1),
                        reduction="none",
                    ).view(batch, steps)
                probabilities = blurred
        loss = sum(losses)
        if masks is not None:
            counts = masks.sum(dim=1)
            if torch.any(counts == 0):
                raise ValueError("Every sequence needs a valid next-token target.")
            loss = (loss * masks).sum(dim=1) / counts
        return logits, loss.mean()

    def configure_optimizer(self, config):
        decay, no_decay = set(), set()
        decay_modules = (nn.Linear, nn.Conv1d)
        no_decay_modules = (nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for parameter_name, _ in module.named_parameters():
                full_name = (
                    f"{module_name}.{parameter_name}" if module_name else parameter_name
                )
                if parameter_name.endswith("bias"):
                    no_decay.add(full_name)
                elif parameter_name.endswith("weight") and isinstance(module, decay_modules):
                    decay.add(full_name)
                elif parameter_name.endswith("weight") and isinstance(
                    module, no_decay_modules
                ):
                    no_decay.add(full_name)
        no_decay.add("pos_emb")
        parameters = {
            name: parameter
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        decay &= parameters.keys()
        no_decay &= parameters.keys()
        if decay & no_decay or parameters.keys() - (decay | no_decay):
            raise RuntimeError("TrAISformer optimizer parameter grouping is incomplete.")
        groups = [
            {
                "params": [parameters[name] for name in sorted(decay)],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [parameters[name] for name in sorted(no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return torch.optim.AdamW(
            groups, lr=config.learning_rate, betas=tuple(config.betas)
        )


class TrAISformer(nn.Module):
    """Plain-PyTorch wrapper preserving accepted checkpoint parameter names."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.traisformer = TrAISformerModel(config)

    def training_loss(self, batch):
        sequence, mask = batch[:2]
        _, loss = self.traisformer(
            sequence, mask[:, 1:], with_targets=True
        )
        return loss

    @staticmethod
    def lr_multiplier(step, warmup_steps, total_steps):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(
            1.0,
            (step - warmup_steps) / max(1, total_steps - warmup_steps),
        )
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    @torch.no_grad()
    def rollout(self, sequences, output_steps):
        self.eval()
        for _ in range(output_steps):
            conditioned = (
                sequences
                if sequences.shape[1] <= self.config.max_seqlen
                else sequences[:, -self.config.max_seqlen :]
            )
            logits, _ = self.traisformer(conditioned, with_targets=False)
            next_logits = logits[:, -1] / self.config.temperature
            lat, lon, sog, cog = torch.split(
                next_logits,
                (
                    self.config.lat_size,
                    self.config.lon_size,
                    self.config.sog_size,
                    self.config.cog_size,
                ),
                dim=-1,
            )
            if self.config.sample_mode == "pos_vicinity":
                _, current = self.traisformer.to_indexes(conditioned[:, -1:])
                lat = _nearest_logits(lat, current[:, 0, 0:1], self.config.r_vicinity)
                lon = _nearest_logits(lon, current[:, 0, 1:2], self.config.r_vicinity)
            if self.config.top_k is not None:
                lat = _top_k_logits(lat, self.config.top_k)
                lon = _top_k_logits(lon, self.config.top_k)
                sog = _top_k_logits(sog, self.config.top_k)
                cog = _top_k_logits(cog, self.config.top_k)
            components = [lat, lon, sog, cog]
            if self.config.sample_predictions:
                sampled = [
                    torch.multinomial(F.softmax(component, dim=-1), 1)
                    for component in components
                ]
            else:
                sampled = [component.argmax(dim=-1, keepdim=True) for component in components]
            indices = torch.cat(sampled, dim=-1)
            next_value = (indices.float() + 0.5) / self.traisformer.att_sizes
            sequences = torch.cat((sequences, next_value.unsqueeze(1)), dim=1)
        return sequences[:, -output_steps:, :2]
