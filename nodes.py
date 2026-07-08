from __future__ import annotations

import copy
import importlib
import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_DCT_CACHE: Dict[Tuple[int, str, str], torch.Tensor] = {}


def _dtype_key(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _device_key(device: torch.device) -> str:
    if device.type == "cuda" and device.index is not None:
        return f"cuda:{device.index}"
    return device.type


def _get_dct_matrix(size: int, reference: torch.Tensor) -> torch.Tensor:
    key = (size, _device_key(reference.device), _dtype_key(reference.dtype))
    cached = _DCT_CACHE.get(key)
    if cached is not None:
        return cached

    n = torch.arange(size, device=reference.device, dtype=reference.dtype).unsqueeze(0)
    k = torch.arange(size, device=reference.device, dtype=reference.dtype).unsqueeze(1)
    mat = torch.cos(math.pi / size * (n + 0.5) * k)
    mat[0] *= math.sqrt(1.0 / size)
    if size > 1:
        mat[1:] *= math.sqrt(2.0 / size)
    _DCT_CACHE[key] = mat
    return mat


def _dct2(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("_dct2 expects a 2D tensor")
    h, w = x.shape
    ch = _get_dct_matrix(h, x)
    cw = _get_dct_matrix(w, x)
    return ch @ x @ cw.t()


def _idct2(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("_idct2 expects a 2D tensor")
    h, w = x.shape
    ch = _get_dct_matrix(h, x)
    cw = _get_dct_matrix(w, x)
    return ch.t() @ x @ cw


def _beta_weights(num_bands: int, alpha: float, beta: float, device: torch.device) -> torch.Tensor:
    if num_bands <= 0:
        return torch.empty(0, device=device, dtype=torch.float64)
    x = torch.linspace(0.0, 1.0, steps=num_bands + 2, device=device, dtype=torch.float64)[1:-1]
    if alpha <= 0.0 or beta <= 0.0:
        return torch.ones_like(x)
    weights = x.pow(alpha - 1.0) * (1.0 - x).pow(beta - 1.0)
    if not torch.isfinite(weights).all() or weights.sum() <= 0:
        return torch.ones_like(x)
    return weights


def _allocate_band_buckets(
    band_sizes: List[int],
    target_total: int,
    alpha: float,
    beta: float,
    device: torch.device,
) -> List[int]:
    active = [i for i, size in enumerate(band_sizes) if size > 0]
    if not active:
        return [0 for _ in band_sizes]

    max_total = sum(band_sizes)
    target_total = max(len(active), min(int(target_total), max_total))

    scores = _beta_weights(len(band_sizes), alpha, beta, device=device)
    scores = scores.clone()
    for idx, size in enumerate(band_sizes):
        if size == 0:
            scores[idx] = 0.0
    if scores.sum() <= 0:
        scores = torch.tensor(band_sizes, device=device, dtype=torch.float64)
    if scores.sum() <= 0:
        scores = torch.ones(len(band_sizes), device=device, dtype=torch.float64)

    raw = target_total * scores / scores.sum()
    buckets = torch.floor(raw).to(torch.int64)
    for idx, size in enumerate(band_sizes):
        if size > 0 and buckets[idx] < 1:
            buckets[idx] = 1
    buckets = torch.minimum(buckets, torch.tensor(band_sizes, device=device, dtype=torch.int64))

    fractional = raw - torch.floor(raw)
    current = int(buckets.sum().item())

    if current < target_total:
        order = torch.argsort(fractional, descending=True).tolist()
        changed = True
        while current < target_total and changed:
            changed = False
            for idx in order:
                if buckets[idx] < band_sizes[idx]:
                    buckets[idx] += 1
                    current += 1
                    changed = True
                    if current >= target_total:
                        break
    elif current > target_total:
        order = torch.argsort(fractional, descending=False).tolist()
        changed = True
        while current > target_total and changed:
            changed = False
            for idx in order:
                if band_sizes[idx] > 0 and buckets[idx] > 1:
                    buckets[idx] -= 1
                    current -= 1
                    changed = True
                    if current <= target_total:
                        break

    return [max(0, int(v)) for v in buckets.tolist()]


def _float_hash(values: torch.Tensor, seed: int) -> torch.Tensor:
    x = values.to(torch.float64) + float(seed) * 0.123456789
    noise = torch.sin(x * 12.9898 + 78.233) * 43758.5453
    return noise - torch.floor(noise)


def _parse_json_or_default(value: str, default: Any) -> Any:
    text = (value or "").strip()
    if not text:
        return default
    return json.loads(text)


def _resolve_callable(path: str):
    spec = (path or "").strip()
    if not spec:
        raise ValueError("model_class_path is required")

    if ":" in spec:
        module_ref, attr_name = spec.rsplit(":", 1)
    else:
        module_ref, attr_name = spec.rsplit(".", 1)

    module_ref = module_ref.strip()
    attr_name = attr_name.strip()

    if module_ref.endswith(".py"):
        source_path = Path(module_ref).expanduser()
        if not source_path.exists():
            raise FileNotFoundError(f"Model definition file not found: {source_path}")
        module_name = f"_dct_model_{source_path.stem}_{abs(hash(str(source_path)))}"
        module_spec = importlib.util.spec_from_file_location(module_name, str(source_path))
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"Could not import model definition file: {source_path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_ref)

    target = getattr(module, attr_name, None)
    if target is None:
        raise AttributeError(f"Could not resolve '{attr_name}' from '{module_ref}'")
    return target


def _load_state_dict_payload(path: str, map_location: str) -> Any:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resolved}")

    load_kwargs = {"map_location": map_location}
    try:
        load_kwargs["weights_only"] = False
        return torch.load(str(resolved), **load_kwargs)
    except TypeError:
        load_kwargs.pop("weights_only", None)
        return torch.load(str(resolved), **load_kwargs)


def _extract_state_dict(payload: Any, state_dict_key: str = "") -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        if state_dict_key:
            nested = payload.get(state_dict_key)
            if isinstance(nested, dict):
                payload = nested
        elif "state_dict" in payload and isinstance(payload["state_dict"], dict):
            payload = payload["state_dict"]
        elif "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
            payload = payload["model_state_dict"]

    if not isinstance(payload, dict):
        raise TypeError("Expected a state dict or checkpoint dict containing a state dict")

    return payload


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if not all(isinstance(k, str) for k in keys):
        return state_dict
    if not any(k.startswith("module.") for k in keys):
        return state_dict
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def _instantiate_model_from_class_path(
    model_class_path: str,
    model_args_json: str = "",
    model_kwargs_json: str = "",
) -> nn.Module:
    target = _resolve_callable(model_class_path)
    if not callable(target):
        raise TypeError(f"Resolved object is not callable: {model_class_path}")

    args = _parse_json_or_default(model_args_json, [])
    kwargs = _parse_json_or_default(model_kwargs_json, {})

    if not isinstance(args, list):
        raise TypeError("model_args_json must decode to a JSON list")
    if not isinstance(kwargs, dict):
        raise TypeError("model_kwargs_json must decode to a JSON object")

    model = target(*args, **kwargs)
    if not isinstance(model, nn.Module):
        raise TypeError("Resolved callable did not return an nn.Module instance")
    return model


def _load_model_from_state_dict(
    checkpoint_path: str,
    model_class_path: str,
    model_args_json: str,
    model_kwargs_json: str,
    state_dict_key: str,
    map_location: str,
    strict: bool,
) -> nn.Module:
    checkpoint = _load_state_dict_payload(checkpoint_path, map_location)
    state_dict = _extract_state_dict(checkpoint, state_dict_key=state_dict_key)
    state_dict = _strip_module_prefix(state_dict)
    model = _instantiate_model_from_class_path(model_class_path, model_args_json, model_kwargs_json)

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError(f"Strict load failed. missing={missing}, unexpected={unexpected}")
    model.eval()
    return model


def _get_module_device(module: nn.Module) -> torch.device:
    for param in module.parameters():
        return param.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


def _move_tensor_like(x: torch.Tensor, device: torch.device, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if dtype is None:
        return x.to(device=device)
    return x.to(device=device, dtype=dtype)


def _save_model(module: nn.Module, output_path: str, file_name: str, save_to_cpu: bool) -> str:
    base = Path(output_path).expanduser() if output_path else Path.cwd() / "compressed_models"

    if base.suffix in {".pt", ".pth"}:
        target = base
    else:
        base.mkdir(parents=True, exist_ok=True)
        name = (file_name or "compressed_model.pth").strip() or "compressed_model.pth"
        target = base / name

    if target.suffix == "":
        target = target.with_suffix(".pth")

    target.parent.mkdir(parents=True, exist_ok=True)
    candidate = target
    stem = target.stem
    suffix = target.suffix or ".pth"
    counter = 1
    while candidate.exists():
        candidate = target.with_name(f"{stem}_{counter}{suffix}")
        counter += 1

    payload = copy.deepcopy(module).cpu() if save_to_cpu else copy.deepcopy(module)
    payload.eval()
    torch.save(payload, str(candidate))
    return str(candidate)


def _load_torchvision_resnet18(pretrained: bool) -> nn.Module:
    try:
        from torchvision import models
    except Exception as exc:
        raise ImportError(
            "torchvision is required to load ResNet18. Install torchvision in the ComfyUI environment."
        ) from exc

    weights = None
    if pretrained:
        if not hasattr(models, "ResNet18_Weights"):
            raise RuntimeError("This torchvision version does not expose ResNet18_Weights.")
        weights = models.ResNet18_Weights.IMAGENET1K_V1

    model = models.resnet18(weights=weights)
    model.eval()
    return model


@dataclass
class CompressionSummary:
    original_params: int
    compressed_params: int
    compressed_layers: int

    @property
    def ratio(self) -> float:
        if self.original_params == 0:
            return 1.0
        return self.compressed_params / self.original_params

    def as_text(self) -> str:
        return (
            f"compressed_layers={self.compressed_layers}, "
            f"original_params={self.original_params}, "
            f"compressed_params={self.compressed_params}, "
            f"effective_ratio={self.ratio:.6f}"
        )


class DCTCompressedLinear(nn.Module):
    def __init__(
        self,
        layer: nn.Linear,
        target_ratio: float,
        alpha: float,
        beta: float,
        seed: int,
        freeze: bool,
    ) -> None:
        super().__init__()
        self.in_features = layer.in_features
        self.out_features = layer.out_features
        self.bias = None
        if layer.bias is not None:
            self.bias = nn.Parameter(layer.bias.detach().clone(), requires_grad=not freeze)

        weight = layer.weight.detach()
        coeffs = _dct2(weight)
        self._coeff_shape = tuple(coeffs.shape)

        band_sizes = [0 for _ in range(self._coeff_shape[0] + self._coeff_shape[1] - 1)]
        for i in range(self._coeff_shape[0]):
            for j in range(self._coeff_shape[1]):
                band_sizes[i + j] += 1

        target_total = max(1, int(round(weight.numel() * target_ratio)))
        per_band = _allocate_band_buckets(band_sizes, target_total, alpha, beta, coeffs.device)

        band_offsets: List[int] = []
        running = 0
        for size in per_band:
            band_offsets.append(running)
            running += size
        total_buckets = running

        self.register_buffer("_band_offsets", torch.tensor(band_offsets, dtype=torch.int64, device=coeffs.device))
        self.register_buffer("_spatial_index", self._make_spatial_index(weight.shape, coeffs.device))
        self.register_buffer("_band_index", self._make_band_index(weight.shape, coeffs.device))

        bucket_count = torch.tensor(per_band, dtype=torch.int64, device=coeffs.device)
        band_bucket_size = bucket_count[self._band_index]
        hash_source = self._spatial_index + self._band_index * 7919 + int(seed) * 104729
        bucket_fraction = _float_hash(hash_source, seed)
        local_bucket = torch.floor(bucket_fraction * band_bucket_size.to(torch.float64)).to(torch.int64)
        local_bucket = torch.minimum(local_bucket, band_bucket_size - 1)
        global_bucket = self._band_offsets[self._band_index] + local_bucket

        sign_fraction = _float_hash(hash_source + 17, seed + 1)
        sign = torch.where(sign_fraction < 0.5, -torch.ones_like(sign_fraction), torch.ones_like(sign_fraction))

        self.register_buffer("_global_bucket", global_bucket.reshape(-1))
        self.register_buffer("_sign", sign.reshape(-1).to(coeffs.dtype))

        shared = torch.zeros(total_buckets, device=coeffs.device, dtype=coeffs.dtype)
        counts = torch.zeros(total_buckets, device=coeffs.device, dtype=coeffs.dtype)
        coeff_flat = coeffs.reshape(-1)
        signed_coeff = self._sign * coeff_flat
        shared.index_add_(0, self._global_bucket, signed_coeff)
        counts.index_add_(0, self._global_bucket, torch.ones_like(coeff_flat))
        shared = shared / torch.clamp(counts, min=1.0)

        self.shared_weights = nn.Parameter(shared, requires_grad=not freeze)
        self.register_buffer("_counts", counts)

    @staticmethod
    def _make_spatial_index(shape: torch.Size, device: torch.device) -> torch.Tensor:
        out_features, in_features = shape
        return torch.arange(out_features * in_features, device=device, dtype=torch.int64).reshape(out_features, in_features)

    @staticmethod
    def _make_band_index(shape: torch.Size, device: torch.device) -> torch.Tensor:
        out_features, in_features = shape
        rows = torch.arange(out_features, device=device, dtype=torch.int64).unsqueeze(1)
        cols = torch.arange(in_features, device=device, dtype=torch.int64).unsqueeze(0)
        return rows + cols

    def reconstruct_weight(self) -> torch.Tensor:
        coeff = self.shared_weights[self._global_bucket] * self._sign
        coeff = coeff.reshape(self._coeff_shape)
        return _idct2(coeff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.reconstruct_weight()
        return F.linear(x, weight, self.bias)

    @property
    def weight(self) -> torch.Tensor:
        return self.reconstruct_weight()


class DCTCompressedConv2d(nn.Module):
    def __init__(
        self,
        layer: nn.Conv2d,
        target_ratio: float,
        alpha: float,
        beta: float,
        seed: int,
        freeze: bool,
    ) -> None:
        super().__init__()
        self.stride = layer.stride
        self.padding = layer.padding
        self.dilation = layer.dilation
        self.groups = layer.groups
        self.padding_mode = layer.padding_mode
        self._reversed_padding_repeated_twice = layer._reversed_padding_repeated_twice

        self.bias = None
        if layer.bias is not None:
            self.bias = nn.Parameter(layer.bias.detach().clone(), requires_grad=not freeze)

        weight = layer.weight.detach()
        out_channels, in_channels, k_h, k_w = weight.shape
        coeffs = self._dct_weight(weight)
        self._coeff_shape = tuple(coeffs.shape)

        band_sizes = [0 for _ in range(k_h + k_w - 1)]
        for u in range(k_h):
            for v in range(k_w):
                band_sizes[u + v] += out_channels * in_channels

        target_total = max(1, int(round(weight.numel() * target_ratio)))
        per_band = _allocate_band_buckets(band_sizes, target_total, alpha, beta, coeffs.device)

        band_offsets: List[int] = []
        running = 0
        for size in per_band:
            band_offsets.append(running)
            running += size
        total_buckets = running

        self.register_buffer("_band_offsets", torch.tensor(band_offsets, dtype=torch.int64, device=coeffs.device))
        self.register_buffer("_spatial_index", self._make_spatial_index(weight.shape, coeffs.device))
        self.register_buffer("_band_index", self._make_band_index(weight.shape, coeffs.device))

        bucket_count = torch.tensor(per_band, dtype=torch.int64, device=coeffs.device)
        band_bucket_size = bucket_count[self._band_index]
        hash_source = self._spatial_index + self._band_index * 7919 + int(seed) * 104729
        bucket_fraction = _float_hash(hash_source, seed)
        local_bucket = torch.floor(bucket_fraction * band_bucket_size.to(torch.float64)).to(torch.int64)
        local_bucket = torch.minimum(local_bucket, band_bucket_size - 1)
        global_bucket = self._band_offsets[self._band_index] + local_bucket

        sign_fraction = _float_hash(hash_source + 17, seed + 1)
        sign = torch.where(sign_fraction < 0.5, -torch.ones_like(sign_fraction), torch.ones_like(sign_fraction))

        self.register_buffer("_global_bucket", global_bucket.reshape(-1))
        self.register_buffer("_sign", sign.reshape(-1).to(coeffs.dtype))

        shared = torch.zeros(total_buckets, device=coeffs.device, dtype=coeffs.dtype)
        counts = torch.zeros(total_buckets, device=coeffs.device, dtype=coeffs.dtype)
        coeff_flat = coeffs.reshape(-1)
        signed_coeff = self._sign * coeff_flat
        shared.index_add_(0, self._global_bucket, signed_coeff)
        counts.index_add_(0, self._global_bucket, torch.ones_like(coeff_flat))
        shared = shared / torch.clamp(counts, min=1.0)

        self.shared_weights = nn.Parameter(shared, requires_grad=not freeze)
        self.register_buffer("_counts", counts)

    @staticmethod
    def _make_spatial_index(shape: torch.Size, device: torch.device) -> torch.Tensor:
        out_channels, in_channels, k_h, k_w = shape
        per_filter = k_h * k_w
        prefix = out_channels * in_channels
        prefix_index = torch.arange(prefix, device=device, dtype=torch.int64).repeat_interleave(per_filter)
        spatial_index = torch.arange(per_filter, device=device, dtype=torch.int64).repeat(prefix)
        return prefix_index * per_filter + spatial_index

    @staticmethod
    def _make_band_index(shape: torch.Size, device: torch.device) -> torch.Tensor:
        _, _, k_h, k_w = shape
        rows = torch.arange(k_h, device=device, dtype=torch.int64).unsqueeze(1)
        cols = torch.arange(k_w, device=device, dtype=torch.int64).unsqueeze(0)
        spatial_band = (rows + cols).reshape(-1)
        prefix = shape[0] * shape[1]
        return spatial_band.repeat(prefix)

    @staticmethod
    def _dct_weight(weight: torch.Tensor) -> torch.Tensor:
        out_channels, in_channels, k_h, k_w = weight.shape
        flat = weight.reshape(-1, k_h, k_w)
        ch = _get_dct_matrix(k_h, weight)
        cw = _get_dct_matrix(k_w, weight)
        coeff = torch.matmul(ch, flat)
        coeff = torch.matmul(coeff, cw.t())
        return coeff.reshape(out_channels, in_channels, k_h, k_w)

    @staticmethod
    def _idct_weight(coeff: torch.Tensor) -> torch.Tensor:
        out_channels, in_channels, k_h, k_w = coeff.shape
        flat = coeff.reshape(-1, k_h, k_w)
        ch = _get_dct_matrix(k_h, coeff)
        cw = _get_dct_matrix(k_w, coeff)
        weight = torch.matmul(ch.t(), flat)
        weight = torch.matmul(weight, cw)
        return weight.reshape(out_channels, in_channels, k_h, k_w)

    def reconstruct_weight(self) -> torch.Tensor:
        coeff = self.shared_weights[self._global_bucket] * self._sign
        coeff = coeff.reshape(self._coeff_shape)
        return self._idct_weight(coeff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.reconstruct_weight()
        if self.padding_mode != "zeros":
            x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = 0
        else:
            padding = self.padding
        return F.conv2d(x, weight, self.bias, self.stride, padding, self.dilation, self.groups)

    @property
    def weight(self) -> torch.Tensor:
        return self.reconstruct_weight()


def _replace_modules(
    module: nn.Module,
    target_ratio: float,
    alpha: float,
    beta: float,
    seed: int,
    freeze: bool,
    include_conv: bool,
    include_linear: bool,
    summary: CompressionSummary,
) -> None:
    for child_name, child in module.named_children():
        if include_conv and isinstance(child, nn.Conv2d):
            setattr(
                module,
                child_name,
                DCTCompressedConv2d(child, target_ratio, alpha, beta, seed, freeze),
            )
            summary.compressed_layers += 1
            summary.original_params += child.weight.numel() + (child.bias.numel() if child.bias is not None else 0)
            compressed_module = getattr(module, child_name)
            summary.compressed_params += compressed_module.shared_weights.numel() + (
                compressed_module.bias.numel() if compressed_module.bias is not None else 0
            )
        elif include_linear and isinstance(child, nn.Linear):
            setattr(
                module,
                child_name,
                DCTCompressedLinear(child, target_ratio, alpha, beta, seed, freeze),
            )
            summary.compressed_layers += 1
            summary.original_params += child.weight.numel() + (child.bias.numel() if child.bias is not None else 0)
            compressed_module = getattr(module, child_name)
            summary.compressed_params += compressed_module.shared_weights.numel() + (
                compressed_module.bias.numel() if compressed_module.bias is not None else 0
            )
        else:
            _replace_modules(
                child,
                target_ratio,
                alpha,
                beta,
                seed,
                freeze,
                include_conv,
                include_linear,
                summary,
            )


def _load_torch_model(path: str, map_location: str) -> nn.Module:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Model file not found: {resolved}")

    load_kwargs = {"map_location": map_location}
    try:
        load_kwargs["weights_only"] = False
        obj = torch.load(str(resolved), **load_kwargs)
    except TypeError:
        load_kwargs.pop("weights_only", None)
        obj = torch.load(str(resolved), **load_kwargs)

    if isinstance(obj, nn.Module):
        return obj
    raise TypeError(
        "torch.load returned a state dict or another object, not an nn.Module. "
        "This node expects a pickled PyTorch model object."
    )


class LoadTorchModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": ""}),
                "map_location": ("STRING", {"default": "cpu"}),
            }
        }

    RETURN_TYPES = ("PYTORCH_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "DCT Compression"

    def load(self, model_path: str, map_location: str):
        model = _load_torch_model(model_path, map_location)
        model.eval()
        return (model,)


class LoadTorchModelFromStateDict:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint_path": ("STRING", {"default": ""}),
                "model_class_path": ("STRING", {"default": ""}),
                "model_args_json": ("STRING", {"default": "[]"}),
                "model_kwargs_json": ("STRING", {"default": "{}"}),
                "state_dict_key": ("STRING", {"default": ""}),
                "map_location": ("STRING", {"default": "cpu"}),
                "strict": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("PYTORCH_MODEL", "STRING")
    RETURN_NAMES = ("model", "summary")
    FUNCTION = "load"
    CATEGORY = "DCT Compression"

    def load(
        self,
        checkpoint_path: str,
        model_class_path: str,
        model_args_json: str,
        model_kwargs_json: str,
        state_dict_key: str,
        map_location: str,
        strict: bool,
    ):
        model = _load_model_from_state_dict(
            checkpoint_path=checkpoint_path,
            model_class_path=model_class_path,
            model_args_json=model_args_json,
            model_kwargs_json=model_kwargs_json,
            state_dict_key=state_dict_key,
            map_location=map_location,
            strict=strict,
        )
        return (model, f"loaded from state_dict: {checkpoint_path}")


class LoadTorchVisionResNet18:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pretrained": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("PYTORCH_MODEL", "STRING")
    RETURN_NAMES = ("model", "summary")
    FUNCTION = "load"
    CATEGORY = "DCT Compression"

    def load(self, pretrained: bool):
        model = _load_torchvision_resnet18(pretrained)
        mode = "pretrained" if pretrained else "random-init"
        return (model, f"loaded torchvision resnet18 ({mode})")


class DCTCompressCNN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PYTORCH_MODEL",),
                "compression_ratio": ("FLOAT", {"default": 0.125, "min": 0.001, "max": 1.0, "step": 0.001}),
                "alpha": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 10.0, "step": 0.01}),
                "beta": ("FLOAT", {"default": 2.5, "min": 0.01, "max": 10.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1, "step": 1}),
                "include_conv": ("BOOLEAN", {"default": True}),
                "include_linear": ("BOOLEAN", {"default": True}),
                "freeze_parameters": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("PYTORCH_MODEL", "STRING")
    RETURN_NAMES = ("model", "summary")
    FUNCTION = "compress"
    CATEGORY = "DCT Compression"

    def compress(
        self,
        model: nn.Module,
        compression_ratio: float,
        alpha: float,
        beta: float,
        seed: int,
        include_conv: bool,
        include_linear: bool,
        freeze_parameters: bool,
    ):
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")

        if compression_ratio <= 0:
            raise ValueError("compression_ratio must be positive")

        compressed = copy.deepcopy(model)
        compressed.eval()

        if compression_ratio >= 1.0:
            return (compressed, "compression_ratio>=1.0, original model returned without layer replacement")

        summary = CompressionSummary(original_params=0, compressed_params=0, compressed_layers=0)
        _replace_modules(
            compressed,
            compression_ratio,
            alpha,
            beta,
            seed,
            freeze_parameters,
            include_conv,
            include_linear,
            summary,
        )

        return (compressed, summary.as_text())


class SaveTorchModel:
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PYTORCH_MODEL",),
                "output_path": ("STRING", {"default": ""}),
                "file_name": ("STRING", {"default": "compressed_model.pth"}),
                "save_to_cpu": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save"
    CATEGORY = "DCT Compression"

    def save(self, model: nn.Module, output_path: str, file_name: str, save_to_cpu: bool):
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        saved_path = _save_model(model, output_path, file_name, save_to_cpu)
        return (saved_path,)


class RunTorchModelInference:
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PYTORCH_MODEL",),
                "input_tensor": ("TENSOR",),
                "return_to_cpu": ("BOOLEAN", {"default": True}),
                "use_no_grad": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("TENSOR", "STRING")
    RETURN_NAMES = ("output_tensor", "summary")
    FUNCTION = "run"
    CATEGORY = "DCT Compression"

    def run(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        return_to_cpu: bool,
        use_no_grad: bool,
    ):
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if not isinstance(input_tensor, torch.Tensor):
            raise TypeError("input_tensor must be a torch.Tensor")

        model.eval()
        model_device = _get_module_device(model)
        reference_dtype = next(iter(model.parameters()), None)
        model_dtype = reference_dtype.dtype if reference_dtype is not None else None
        tensor = _move_tensor_like(input_tensor, model_device, model_dtype)

        if use_no_grad:
            with torch.no_grad():
                output = model(tensor)
        else:
            output = model(tensor)

        if not isinstance(output, torch.Tensor):
            raise TypeError(
                "RunTorchModelInference expects the model to return a torch.Tensor. "
                f"Got {type(output).__name__}"
            )

        output = output.detach()
        if return_to_cpu:
            output = output.cpu()
        summary = (
            f"input_shape={tuple(input_tensor.shape)}, "
            f"output_shape={tuple(output.shape)}, "
            f"device={str(model_device)}"
        )
        return (output, summary)


NODE_CLASS_MAPPINGS = {
    "LoadTorchModel": LoadTorchModel,
    "LoadTorchModelFromStateDict": LoadTorchModelFromStateDict,
    "LoadTorchVisionResNet18": LoadTorchVisionResNet18,
    "DCTCompressCNN": DCTCompressCNN,
    "SaveTorchModel": SaveTorchModel,
    "RunTorchModelInference": RunTorchModelInference,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadTorchModel": "Load Torch CNN Model",
    "LoadTorchModelFromStateDict": "Load Torch Model From State Dict",
    "LoadTorchVisionResNet18": "Load TorchVision ResNet18",
    "DCTCompressCNN": "DCT Compress CNN",
    "SaveTorchModel": "Save Torch Model",
    "RunTorchModelInference": "Run Torch Model Inference",
}
