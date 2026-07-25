"""Transformer action autoencoder used as the PAIR action-side teacher."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ActionAEConfig:
    """Architecture settings for the action autoencoder."""

    horizon: int = 8
    action_dim: int = 7
    hidden_dim: int = 64
    latent_dim: int = 16
    encoder_layers: int = 4
    decoder_layers: int = 2
    num_heads: int = 4
    ffn_dim: int = 256
    dropout: float = 0.0
    activation: str = "gelu"
    norm_first: bool = True

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ActionAEConfig":
        allowed = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in allowed})


@dataclass(frozen=True)
class ActionPerceptionAEConfig:
    """Architecture settings for the perception-conditioned action autoencoder."""

    horizon: int = 8
    action_dim: int = 7
    hidden_dim: int = 64
    latent_dim: int = 16
    perception_dim: int = 896
    encoder_layers: int = 1
    decoder_layers: int = 2
    num_heads: int = 4
    ffn_dim: int = 256
    perception_heads: int = 4
    perception_layers: int = 1
    dropout: float = 0.0
    activation: str = "gelu"
    norm_first: bool = True
    mask_mode: str = "random"
    mask_prob: float = 0.3
    mask_count: int = 4
    noise_std: float = 0.05

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ActionPerceptionAEConfig":
        allowed = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in allowed})


@dataclass(frozen=True)
class ActionPerceptionAEOutput:
    recon_actions: Tensor
    latents: Tensor
    corrupted_actions: Tensor
    action_mask: Tensor


def _make_transformer_stack(
    *,
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    ffn_dim: int,
    dropout: float,
    activation: str,
    norm_first: bool,
) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=hidden_dim,
        nhead=num_heads,
        dim_feedforward=ffn_dim,
        dropout=dropout,
        activation=activation,
        batch_first=True,
        norm_first=norm_first,
    )
    return nn.TransformerEncoder(layer, num_layers=num_layers)


class ActionEncoder(nn.Module):
    """Maps normalized action chunks `[B, H, action_dim]` to `[B, H, latent_dim]`."""

    def __init__(self, config: ActionAEConfig) -> None:
        super().__init__()
        self.config = config
        self.requires_perception = False
        self.latent_dim = config.latent_dim
        self.input_proj = nn.Linear(config.action_dim, config.hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.horizon, config.hidden_dim))
        self.blocks = _make_transformer_stack(
            hidden_dim=config.hidden_dim,
            num_layers=config.encoder_layers,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
            activation=config.activation,
            norm_first=config.norm_first,
        )
        self.latent_proj = nn.Linear(config.hidden_dim, config.latent_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, actions: Tensor) -> Tensor:
        if actions.ndim != 3:
            raise ValueError(f"Expected actions with shape [B, H, A], got {tuple(actions.shape)}")
        if actions.shape[1:] != (self.config.horizon, self.config.action_dim):
            raise ValueError(
                "Expected actions with trailing shape "
                f"[{self.config.horizon}, {self.config.action_dim}], got {tuple(actions.shape[1:])}"
            )

        x = self.input_proj(actions)
        x = x + self.pos_embed.to(dtype=x.dtype)
        x = self.blocks(x)
        return self.latent_proj(x)


class ActionDecoder(nn.Module):
    """Reconstructs normalized action chunks from action latents."""

    def __init__(self, config: ActionAEConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.latent_dim, config.hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.horizon, config.hidden_dim))
        self.blocks = _make_transformer_stack(
            hidden_dim=config.hidden_dim,
            num_layers=config.decoder_layers,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
            activation=config.activation,
            norm_first=config.norm_first,
        )
        self.output_proj = nn.Linear(config.hidden_dim, config.action_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, latents: Tensor) -> Tensor:
        if latents.ndim != 3:
            raise ValueError(f"Expected latents with shape [B, H, Z], got {tuple(latents.shape)}")
        if latents.shape[1:] != (self.config.horizon, self.config.latent_dim):
            raise ValueError(
                "Expected latents with trailing shape "
                f"[{self.config.horizon}, {self.config.latent_dim}], got {tuple(latents.shape[1:])}"
            )

        x = self.input_proj(latents)
        x = x + self.pos_embed.to(dtype=x.dtype)
        x = self.blocks(x)
        return self.output_proj(x)


class ActionTransformerAE(nn.Module):
    """Small action autoencoder for producing frozen action-side latent teachers."""

    def __init__(self, config: ActionAEConfig | None = None) -> None:
        super().__init__()
        self.config = config or ActionAEConfig()
        self.encoder = ActionEncoder(self.config)
        self.decoder = ActionDecoder(self.config)

    def encode(self, actions: Tensor) -> Tensor:
        return self.encoder(actions)

    def decode(self, latents: Tensor) -> Tensor:
        return self.decoder(latents)

    def forward(self, actions: Tensor) -> Tuple[Tensor, Tensor]:
        latents = self.encode(actions)
        recon_actions = self.decode(latents)
        return recon_actions, latents


def corrupt_actions(
    actions: Tensor,
    *,
    mask_prob: float,
    noise_std: float,
    mask_mode: str = "random",
    mask_count: int = 4,
    training: bool = True,
) -> Tuple[Tensor, Tensor]:
    if actions.ndim != 3:
        raise ValueError(f"Expected actions with shape [B, H, A], got {tuple(actions.shape)}")
    if mask_mode not in {"random", "chunk"}:
        raise ValueError(f"Unsupported mask_mode={mask_mode!r}; expected 'random' or 'chunk'")
    if mask_mode == "chunk":
        if not isinstance(mask_count, int):
            raise ValueError(f"mask_count must be an integer, got {type(mask_count).__name__}")
        horizon = actions.shape[1]
        if not 0 <= mask_count <= horizon:
            raise ValueError(f"mask_count must be between 0 and horizon ({horizon}), got {mask_count}")

    mask = torch.zeros(actions.shape[:2], dtype=torch.bool, device=actions.device)
    corrupted = actions
    if training and noise_std > 0:
        corrupted = corrupted + torch.randn_like(corrupted) * noise_std
    if training and mask_mode == "random" and mask_prob > 0:
        mask = torch.rand(actions.shape[:2], device=actions.device) < mask_prob
        corrupted = corrupted.masked_fill(mask.unsqueeze(-1), 0.0)
    elif training and mask_mode == "chunk" and mask_count > 0:
        batch_size, horizon = actions.shape[:2]
        starts = torch.randint(
            low=0,
            high=horizon - mask_count + 1,
            size=(batch_size,),
            device=actions.device,
        )
        positions = torch.arange(horizon, device=actions.device).unsqueeze(0)
        mask = (positions >= starts.unsqueeze(1)) & (positions < (starts + mask_count).unsqueeze(1))
        corrupted = corrupted.masked_fill(mask.unsqueeze(-1), 0.0)
    return corrupted, mask


class PerceptionCrossAttentionBlock(nn.Module):
    """Cross-attends action hidden states to perception memory, then applies an MLP."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )

    def forward(self, x: Tensor, memory: Tensor, key_padding_mask: Optional[Tensor]) -> Tensor:
        memory = self.memory_norm(memory)
        attended, _ = self.cross_attn(
            query=self.query_norm(x),
            key=memory,
            value=memory,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attended
        return x + self.mlp(self.mlp_norm(x))


class ActionPerceptionEncoder(nn.Module):
    """Maps actions plus `[V0;T0]` perception tokens to `[B,H,latent_dim]`."""

    def __init__(self, config: ActionPerceptionAEConfig) -> None:
        super().__init__()
        self.config = config
        self.requires_perception = True
        self.latent_dim = config.latent_dim
        if config.perception_layers < 1:
            raise ValueError("ActionPerceptionAEConfig.perception_layers must be >= 1")
        self.input_proj = nn.Linear(config.action_dim, config.hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.horizon, config.hidden_dim))
        self.blocks = _make_transformer_stack(
            hidden_dim=config.hidden_dim,
            num_layers=config.encoder_layers,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
            activation=config.activation,
            norm_first=config.norm_first,
        )
        self.perception_proj = nn.Linear(config.perception_dim, config.hidden_dim)
        self.cross_blocks = nn.ModuleList(
            [
                PerceptionCrossAttentionBlock(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.perception_heads,
                    ffn_dim=config.ffn_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.perception_layers)
            ]
        )
        self.latent_proj = nn.Linear(config.hidden_dim, config.latent_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(
        self,
        actions: Tensor,
        perception_tokens: Tensor,
        perception_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if actions.ndim != 3:
            raise ValueError(f"Expected actions with shape [B, H, A], got {tuple(actions.shape)}")
        if actions.shape[1:] != (self.config.horizon, self.config.action_dim):
            raise ValueError(
                "Expected actions with trailing shape "
                f"[{self.config.horizon}, {self.config.action_dim}], got {tuple(actions.shape[1:])}"
            )
        if perception_tokens.ndim != 3:
            raise ValueError(f"Expected perception tokens [B,N,D], got {tuple(perception_tokens.shape)}")
        if perception_tokens.shape[0] != actions.shape[0] or perception_tokens.shape[-1] != self.config.perception_dim:
            raise ValueError(
                "Expected perception tokens with shape "
                f"[{actions.shape[0]},N,{self.config.perception_dim}], got {tuple(perception_tokens.shape)}"
            )
        if perception_mask is not None and perception_mask.shape != perception_tokens.shape[:2]:
            raise ValueError(
                f"Expected perception mask shape {tuple(perception_tokens.shape[:2])}, got {tuple(perception_mask.shape)}"
            )

        x = self.input_proj(actions.float())
        x = x + self.pos_embed.to(dtype=x.dtype)
        x = self.blocks(x)

        memory = self.perception_proj(perception_tokens.float())
        key_padding_mask = None if perception_mask is None else ~perception_mask.bool()
        for block in self.cross_blocks:
            x = block(x, memory, key_padding_mask)
        return self.latent_proj(x)


class ActionPerceptionTransformerAE(nn.Module):
    """Perception-conditioned denoising action autoencoder."""

    def __init__(self, config: ActionPerceptionAEConfig | None = None) -> None:
        super().__init__()
        self.config = config or ActionPerceptionAEConfig()
        self.encoder = ActionPerceptionEncoder(self.config)
        decoder_config = ActionAEConfig(
            horizon=self.config.horizon,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
            latent_dim=self.config.latent_dim,
            encoder_layers=self.config.encoder_layers,
            decoder_layers=self.config.decoder_layers,
            num_heads=self.config.num_heads,
            ffn_dim=self.config.ffn_dim,
            dropout=self.config.dropout,
            activation=self.config.activation,
            norm_first=self.config.norm_first,
        )
        self.decoder = ActionDecoder(decoder_config)

    def encode(
        self,
        actions: Tensor,
        perception_tokens: Tensor,
        perception_mask: Optional[Tensor] = None,
    ) -> Tensor:
        return self.encoder(actions, perception_tokens, perception_mask)

    def decode(self, latents: Tensor) -> Tensor:
        return self.decoder(latents)

    def forward(
        self,
        actions: Tensor,
        perception_tokens: Tensor,
        perception_mask: Optional[Tensor] = None,
        *,
        corrupt: bool = True,
    ) -> ActionPerceptionAEOutput:
        corrupted_actions, action_mask = corrupt_actions(
            actions.float(),
            mask_prob=self.config.mask_prob,
            noise_std=self.config.noise_std,
            mask_mode=self.config.mask_mode,
            mask_count=self.config.mask_count,
            training=corrupt and self.training,
        )
        latents = self.encode(corrupted_actions, perception_tokens, perception_mask)
        recon_actions = self.decode(latents)
        return ActionPerceptionAEOutput(
            recon_actions=recon_actions,
            latents=latents,
            corrupted_actions=corrupted_actions,
            action_mask=action_mask,
        )
