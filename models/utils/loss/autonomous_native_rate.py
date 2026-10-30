"""Dense teacher rate accounting using physical streams exactly once.

The native dense coder reports main-stream bits plus its native count/AE
side information. Its normalizer may already preserve this logical value.
"""
import math


def exact_dense_logical_rate(result, normalized, scale_sr, scale_ae):
    main = normalized.get('bitstream_file_bits')
    reported = result.get('file_size')
    if (not isinstance(main, (int, float)) or not math.isfinite(main) or main <= 0
            or not isinstance(reported, (int, float)) or not math.isfinite(reported) or reported <= 0):
        raise ValueError('Native dense rate requires measured physical and reported logical bits')
    side = 4. * (max(int(scale_sr), 0) + max(int(scale_ae), 0))
    logical = max(float(reported), float(main) + side)
    return dict(main_bin_bits=float(main), decoder_side_stage_bits=side,
                logical_file_size=logical, side_information_bits=logical-float(main),
                file_size=logical)


def autonomous_rate_schema(args):
    from ..training.sparsepcgc_autonomous_policy import autonomous_search_enabled
    return ('autonomous_native_dense_logical_v3'
            if autonomous_search_enabled(args) and str(getattr(args, 'sparsepcgc_mode', '')).lower() == 'dense_lossy'
            else None)
