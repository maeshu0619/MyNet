def rearrange(tensor, pattern, **sizes):
    """Small fallback for the einops patterns used in this project."""
    key = " ".join(pattern.replace('"', "").replace("'", "").split())
    if key in {"b n c -> b c n", "b c n -> b n c", "b c m -> b m c"}:
        return tensor.permute(0, 2, 1)
    if key in {"c n -> n c", "c m -> m c"}:
        return tensor.permute(1, 0)
    if key == "m k c -> m c k":
        return tensor.permute(0, 2, 1)
    if key == "c p k -> p c k":
        return tensor.permute(1, 0, 2)
    if key == "b c (s k) -> b c s k":
        s = int(sizes["s"])
        b, c, sk = tensor.shape
        if sk % s != 0:
            raise ValueError(f"Cannot split dimension {sk} by s={s}")
        return tensor.reshape(b, c, s, sk // s)
    if key == "b c n -> 1 c (b n)":
        b, c, n = tensor.shape
        return tensor.permute(1, 0, 2).reshape(1, c, b * n)
    raise NotImplementedError(f"einops fallback does not support pattern: {pattern}")


def repeat(tensor, pattern, **sizes):
    raise NotImplementedError(f"einops repeat fallback does not support pattern: {pattern}")
