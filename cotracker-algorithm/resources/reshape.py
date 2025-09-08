import torch
import torch.nn.functional as F


def reshape_video(
    tensor: torch.Tensor,
    target_shape: tuple[int, int],  # (H, W)
) -> torch.Tensor:
    """Resizes 5-, 4-, or 3D tensors of spatial dimensions (H, W) to the specified target shape.
    Assumes first 1 to 3 dimensions are the non-spatial dimensions (batch, time, channels).

    Args:
        tensor: Input tensor with last two dims as (H, W).
        target_shape: Target spatial size as (H, W).

    Returns:
        Resized tensor with spatial dimensions equal to `target_shape`.
    """
    if not (isinstance(target_shape, (tuple, list)) and len(target_shape) == 2):
        raise ValueError(f"target_shape must be (H, W), got {target_shape}")
    if tensor.ndim < 3:
        raise ValueError(
            f"Expected >=3D tensor with spatial dims at the end, got shape {tuple(tensor.shape)}"
        )

    H_out, W_out = int(target_shape[0]), int(target_shape[1])
    if tensor.size(-2) == H_out and tensor.size(-1) == W_out:
        return tensor

    orig_dtype = tensor.dtype

    spatial_in = tensor.shape[-2:]
    non_spatial = tensor.shape[:-2]
    N_flat = int(torch.tensor(non_spatial).prod().item()) if len(non_spatial) > 0 else 1

    x = tensor.reshape(N_flat, 1, spatial_in[0], spatial_in[1])

    # interpolate in float, then cast back
    x = x.to(torch.float32)
    x_resized = F.interpolate(
        x,
        size=(H_out, W_out),
        mode="bilinear",
        align_corners=False,
    )

    out = x_resized.reshape(*non_spatial, H_out, W_out)

    # for boolean/integer masks, keep nearest semantics and cast back
    if orig_dtype == torch.bool:
        # threshold to preserve boolean nature (nearest already preserves exact values)
        return (out > 0.5).to(dtype=torch.bool)
    elif not torch.is_floating_point(torch.empty((), dtype=orig_dtype)):
        # round before casting back to integer types
        return out.round().to(dtype=orig_dtype)
    else:
        return out.to(dtype=orig_dtype)
