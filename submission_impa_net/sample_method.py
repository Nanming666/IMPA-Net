"""
Compact sample code for the paper:

IMPA-Net: Meteorology-Aware Multi-Scale Attention and Dynamic Loss
for Extreme Convective Radar Nowcasting

This file is intentionally self-contained and simplified for sharing a
representative implementation with example NetCDF data. It preserves the
main ideas used in the full project:

1. Meteorology-aware multi-channel input fusion
2. Multi-scale spatial feature extraction
3. Temporal mixing over the input sequence
4. Dynamic loss weighting for extreme convective echoes

It is not the full training or reproduction pipeline.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import xarray as xr
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError("xarray is required to read NetCDF sample files.") from exc


@dataclass
class SampleConfig:
    input_steps: int = 20
    forecast_steps: int = 20
    input_channels: int = 4
    output_channels: int = 1
    hidden_channels: int = 64
    extreme_threshold: float = 35.0 / 85.0


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class MultiScaleSpatialGate(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                ConvNormAct(channels, channels, kernel_size=3),
                ConvNormAct(channels, channels, kernel_size=5),
                ConvNormAct(channels, channels, kernel_size=7),
            ]
        )
        self.merge = ConvNormAct(channels * 3, channels, kernel_size=1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multi_scale = torch.cat([branch(x) for branch in self.branches], dim=1)
        fused = self.merge(multi_scale)
        return fused * self.gate(fused)


class MeteorologyAwareEncoder(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.radar_stem = nn.Sequential(
            ConvNormAct(1, hidden_channels // 2, kernel_size=5),
            ConvNormAct(hidden_channels // 2, hidden_channels, kernel_size=3),
        )

        aux_channels = max(input_channels - 1, 0)
        self.aux_stem = None
        if aux_channels > 0:
            self.aux_stem = nn.Sequential(
                ConvNormAct(aux_channels, hidden_channels // 2, kernel_size=1),
                ConvNormAct(hidden_channels // 2, hidden_channels, kernel_size=3),
            )

        self.fuse = ConvNormAct(hidden_channels * 2, hidden_channels, kernel_size=1)
        self.pre_down_gate = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.downsample = ConvNormAct(hidden_channels, hidden_channels, kernel_size=3, stride=2)
        self.spatial_gate = MultiScaleSpatialGate(hidden_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        radar_feat = self.radar_stem(x[:, :1])

        if self.aux_stem is not None and x.shape[1] > 1:
            aux_feat = self.aux_stem(x[:, 1:])
            fused = self.fuse(torch.cat([radar_feat, aux_feat], dim=1))
        else:
            fused = radar_feat

        skip = fused * self.pre_down_gate(fused)
        latent = self.downsample(skip)
        latent = self.spatial_gate(latent)
        return latent, skip


class TemporalMixer(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.GELU(),
        )
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(channels, channels, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
                    nn.BatchNorm3d(channels),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv3d(channels, channels, kernel_size=(5, 3, 3), padding=(2, 1, 1), bias=False),
                    nn.BatchNorm3d(channels),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv3d(channels, channels, kernel_size=(7, 3, 3), padding=(3, 1, 1), bias=False),
                    nn.BatchNorm3d(channels),
                    nn.GELU(),
                ),
            ]
        )
        self.merge = nn.Sequential(
            nn.Conv3d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.GELU(),
        )
        squeeze_channels = max(channels // 4, 8)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, squeeze_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv3d(squeeze_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.stem(x)
        multi_scale = torch.cat([branch(x) for branch in self.branches], dim=1)
        x = self.merge(multi_scale)
        x = x * self.attn(x)
        x = self.refine(x)
        x = x + residual.permute(0, 2, 1, 3, 4).contiguous()
        return x.permute(0, 2, 1, 3, 4).contiguous()


class ForecastDecoder(nn.Module):
    def __init__(self, hidden_channels: int, output_channels: int) -> None:
        super().__init__()
        self.skip_proj = ConvNormAct(hidden_channels, hidden_channels, kernel_size=3)
        self.fuse = ConvNormAct(hidden_channels * 2, hidden_channels, kernel_size=3)
        self.refine = nn.Sequential(
            ConvNormAct(hidden_channels, hidden_channels, kernel_size=3),
            ConvNormAct(hidden_channels, hidden_channels, kernel_size=3),
        )
        self.readout = nn.Conv2d(hidden_channels, output_channels, kernel_size=1)

    def forward(self, latent: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        latent = F.interpolate(latent, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.skip_proj(skip)
        fused = self.fuse(torch.cat([latent, skip], dim=1))
        fused = self.refine(fused)
        return self.readout(fused)


class DynamicExtremeLoss(nn.Module):
    def __init__(
        self,
        extreme_threshold: float = 35.0 / 85.0,
        extreme_boost: float = 3.0,
        temporal_weight: float = 0.2,
        gradient_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.extreme_threshold = extreme_threshold
        self.extreme_boost = extreme_boost
        self.temporal_weight = temporal_weight
        self.gradient_weight = gradient_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        if pred.shape != target.shape:
            raise ValueError(f"Prediction shape {pred.shape} does not match target shape {target.shape}.")

        _, steps, _, _, _ = pred.shape
        frame_weights = torch.linspace(1.0, 1.6, steps=steps, device=pred.device, dtype=pred.dtype)
        frame_weights = frame_weights.view(1, steps, 1, 1, 1)

        extreme_mask = (target >= self.extreme_threshold).float()
        pixel_weights = 1.0 + self.extreme_boost * extreme_mask
        weighted_mse = ((pred - target) ** 2 * frame_weights * pixel_weights).mean()

        temporal_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        if steps > 1:
            temporal_loss = F.l1_loss(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])

        grad_x_pred = pred[..., :, 1:] - pred[..., :, :-1]
        grad_x_tgt = target[..., :, 1:] - target[..., :, :-1]
        grad_y_pred = pred[..., 1:, :] - pred[..., :-1, :]
        grad_y_tgt = target[..., 1:, :] - target[..., :-1, :]
        gradient_loss = 0.5 * (
            F.l1_loss(grad_x_pred, grad_x_tgt) + F.l1_loss(grad_y_pred, grad_y_tgt)
        )

        total = weighted_mse + self.temporal_weight * temporal_loss + self.gradient_weight * gradient_loss
        logs = {
            "total_loss": float(total.detach().cpu()),
            "weighted_mse": float(weighted_mse.detach().cpu()),
            "temporal_loss": float(temporal_loss.detach().cpu()),
            "gradient_loss": float(gradient_loss.detach().cpu()),
        }
        return total, logs


class IMPANetSample(nn.Module):
    def __init__(self, config: SampleConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = MeteorologyAwareEncoder(config.input_channels, config.hidden_channels)
        self.temporal_mixer = TemporalMixer(config.hidden_channels)
        self.temporal_projector = nn.Linear(config.input_steps, config.forecast_steps)
        self.decoder = ForecastDecoder(config.hidden_channels, config.output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected input of shape [B, T, C, H, W], got {tuple(x.shape)}.")

        batch_size, input_steps, channels, height, width = x.shape
        if channels != self.config.input_channels:
            raise ValueError(
                f"Expected {self.config.input_channels} channels, got {channels}. "
                "Check the NetCDF variable mapping."
            )
        if input_steps != self.config.input_steps:
            raise ValueError(
                f"Expected {self.config.input_steps} input steps, got {input_steps}. "
                "Update SampleConfig or provide matching sample data."
            )

        frames_2d = x.view(batch_size * input_steps, channels, height, width)
        latent, skip = self.encoder(frames_2d)
        latent_h, latent_w = latent.shape[-2:]

        latent = latent.view(batch_size, input_steps, self.config.hidden_channels, latent_h, latent_w)
        latent = self.temporal_mixer(latent)

        latent = latent.permute(0, 2, 3, 4, 1).contiguous()
        latent = self.temporal_projector(latent)
        latent = latent.permute(0, 4, 1, 2, 3).contiguous()

        skip = skip.view(batch_size, input_steps, self.config.hidden_channels, height, width)
        skip = skip[:, -1:].expand(-1, self.config.forecast_steps, -1, -1, -1).contiguous()

        decoded = self.decoder(
            latent.view(batch_size * self.config.forecast_steps, self.config.hidden_channels, latent_h, latent_w),
            skip.view(batch_size * self.config.forecast_steps, self.config.hidden_channels, height, width),
        )
        return decoded.view(
            batch_size,
            self.config.forecast_steps,
            self.config.output_channels,
            height,
            width,
        )


def _read_case_variable(ds: xr.Dataset, variable_name: str, case_index: int) -> Optional[np.ndarray]:
    if variable_name not in ds:
        return None
    data_array = ds[variable_name]
    if "case" in data_array.dims:
        data_array = data_array.isel(case=case_index)
    return data_array.load().values.astype(np.float32)


def load_case_from_nc(
    nc_path: str | Path,
    case_index: int = 0,
    radar_var: str = "radar_in",
    precip_var: str = "precip_in",
    topo_var: str = "topography",
    wind_var: str = "along_slope_wind",
    target_var: str = "radar_out",
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, np.ndarray]]:
    """
    Load one large-area sample case from a NetCDF file.

    Expected variable layout:
    - radar_in(case, t_in, y, x)
    - precip_in(case, t_in, y, x)                 optional
    - radar_out(case, t_out, y, x)                optional
    - topography(y, x)                            optional
    - along_slope_wind(y, x)                      optional
    - lat(y, x) / lon(y, x) / time_*              optional metadata
    """

    nc_path = Path(nc_path)
    if not nc_path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {nc_path}")

    with xr.open_dataset(nc_path) as ds:
        radar_in = _read_case_variable(ds, radar_var, case_index)
        if radar_in is None:
            raise KeyError(f"Required variable '{radar_var}' is missing from {nc_path}.")

        if radar_in.ndim != 3:
            raise ValueError(f"Expected '{radar_var}' to have shape [T, Y, X], got {radar_in.shape}.")

        input_steps, height, width = radar_in.shape
        input_channels = [radar_in[:, None, :, :]]

        precip_in = _read_case_variable(ds, precip_var, case_index)
        if precip_in is not None:
            if precip_in.shape != radar_in.shape:
                raise ValueError(f"'{precip_var}' shape {precip_in.shape} does not match '{radar_var}' shape {radar_in.shape}.")
            input_channels.append(precip_in[:, None, :, :])

        topography = _read_case_variable(ds, topo_var, case_index)
        if topography is not None:
            if topography.ndim != 2 or topography.shape != (height, width):
                raise ValueError(f"'{topo_var}' must have shape [Y, X], got {topography.shape}.")
            input_channels.append(np.broadcast_to(topography[None, None, :, :], (input_steps, 1, height, width)))

        along_slope_wind = _read_case_variable(ds, wind_var, case_index)
        if along_slope_wind is not None:
            if along_slope_wind.ndim != 2 or along_slope_wind.shape != (height, width):
                raise ValueError(f"'{wind_var}' must have shape [Y, X], got {along_slope_wind.shape}.")
            input_channels.append(
                np.broadcast_to(along_slope_wind[None, None, :, :], (input_steps, 1, height, width))
            )

        target = _read_case_variable(ds, target_var, case_index)
        metadata = {}
        for key in ("lat", "lon", "time_in", "time_out"):
            value = _read_case_variable(ds, key, case_index)
            if value is not None:
                metadata[key] = value

    stacked = np.concatenate(input_channels, axis=1)
    input_tensor = torch.from_numpy(stacked).float()
    target_tensor = None if target is None else torch.from_numpy(target[:, None, :, :]).float()
    return input_tensor, target_tensor, metadata


def build_model_from_sample(nc_path: str | Path, case_index: int = 0) -> Tuple[IMPANetSample, torch.Tensor, Optional[torch.Tensor]]:
    input_tensor, target_tensor, _ = load_case_from_nc(nc_path, case_index=case_index)
    forecast_steps = target_tensor.shape[0] if target_tensor is not None else input_tensor.shape[0]
    config = SampleConfig(
        input_steps=input_tensor.shape[0],
        forecast_steps=forecast_steps,
        input_channels=input_tensor.shape[1],
    )
    model = IMPANetSample(config)
    return model, input_tensor.unsqueeze(0), None if target_tensor is None else target_tensor.unsqueeze(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the compact IMPA-Net sample on one NetCDF case.")
    parser.add_argument("--nc_path", type=str, required=True, help="Path to the sample NetCDF file.")
    parser.add_argument("--case_index", type=int, default=0, help="Case index in the NetCDF file.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device, e.g. cpu or cuda:0.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model, input_batch, target_batch = build_model_from_sample(args.nc_path, case_index=args.case_index)
    model = model.to(device)
    input_batch = input_batch.to(device)

    with torch.no_grad():
        pred = model(input_batch)

    print(f"Input batch shape:   {tuple(input_batch.shape)}")
    print(f"Prediction shape:    {tuple(pred.shape)}")

    if target_batch is not None:
        target_batch = target_batch.to(device)
        loss_fn = DynamicExtremeLoss(extreme_threshold=model.config.extreme_threshold)
        loss, logs = loss_fn(pred, target_batch)
        print(f"Target batch shape:  {tuple(target_batch.shape)}")
        print(f"Sample loss:         {float(loss.detach().cpu()):.6f}")
        print(f"Loss breakdown:      {logs}")
    else:
        print("No 'radar_out' variable was found. Only the forward pass was demonstrated.")


if __name__ == "__main__":
    main()
