"""BD-rate in log10(rate), exact integration of SciPy's piecewise Akima cubic.

Definition follows FAU-LMS/bjontegaard. No rounding, extrapolation, PSNR matching
tolerance, envelope pruning, or silent removal of failed/nonmonotonic points.
"""
import math
import numpy as np
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator


def prepare_curve(rates, qualities, minimum_points=4):
    rates, qualities = np.asarray(rates, dtype=float), np.asarray(qualities, dtype=float)
    warnings = []
    if rates.ndim != 1 or rates.shape != qualities.shape:
        raise ValueError("Rate/PSNR vectors must have equal length")
    # Lossless infinite PSNR cannot be an interpolation knot. Explicitly report it.
    if np.any(~np.isfinite(rates)) or np.any(rates <= 0) or np.any(np.isnan(qualities)):
        raise ValueError("Rates must be finite/positive and PSNR must not be NaN")
    finite = np.isfinite(qualities)
    if not finite.all():
        warnings.append("Excluded {} infinite-PSNR (lossless) point(s)".format(int((~finite).sum())))
    rates, qualities = rates[finite], qualities[finite]
    order = np.argsort(qualities, kind="stable")
    rates, qualities = rates[order], qualities[order]
    # Exactly identical knots are redundant, not near-matched PSNR pairs.
    unique = np.ones(len(rates), dtype=bool)
    for i in range(1, len(rates)):
        if qualities[i] == qualities[i - 1]:
            if rates[i] != rates[i - 1]:
                raise ValueError("Equal PSNR with different rates: ambiguous curve")
            unique[i] = False
    if not unique.all():
        warnings.append("Collapsed {} exactly duplicate RD knot(s)".format(int((~unique).sum())))
    rates, qualities = rates[unique], qualities[unique]
    if len(rates) < minimum_points:
        raise ValueError("Need >= {} distinct finite RD points, got {}".format(minimum_points, len(rates)))
    if np.any(np.diff(rates) <= 0):
        raise ValueError("Nonmonotonic rate versus PSNR; not silently corrected")
    return qualities, np.log10(rates), warnings


def bd_rate(anchor_rate, anchor_psnr, mine_rate, mine_psnr, method="akima",
            minimum_points=4, minimum_overlap_db=0.0, interval=None):
    result = dict(status="invalid", bd_rate_percent=None, method=method, warnings=[])
    try:
        ax, ay, aw = prepare_curve(anchor_rate, anchor_psnr, minimum_points)
        bx, by, bw = prepare_curve(mine_rate, mine_psnr, minimum_points)
        result["warnings"] = aw + bw
        low, high = max(ax[0], bx[0]), min(ax[-1], bx[-1])
        if interval is not None:
            if interval[0] < low or interval[1] > high:
                raise ValueError("Requested interval requires extrapolation")
            low, high = map(float, interval)
        result.update(psnr_low=float(low), psnr_high=float(high),
                      anchor_points=len(ax), mine_points=len(bx))
        if high <= low or high - low < minimum_overlap_db:
            raise ValueError("Insufficient common PSNR range")
        if high - low < 1.0:
            result["warnings"].append("Common PSNR span is less than 1 dB")
        coverage = float((high - low) / (max(ax[-1], bx[-1]) - min(ax[0], bx[0])))
        result["common_psnr_fraction_of_union"] = coverage
        if coverage < .75:
            result["warnings"].append("Common PSNR overlap is less than 75% of the union")
        if min(len(ax), len(bx)) < 6:
            result["warnings"].append("Fewer than six finite RD knots")
        interpolator = {"akima": Akima1DInterpolator, "pchip": PchipInterpolator}[method]
        a, b = interpolator(ax, ay), interpolator(bx, by)
        # Inspect exact piecewise-polynomial extrema, not merely a coarse plot.
        for label, poly in (("Baseline", a), ("MINE", b)):
            derivative = poly.derivative()
            critical = derivative.derivative().roots(extrapolate=False)
            probes = np.r_[low, high, poly.x[(poly.x > low) & (poly.x < high)],
                           critical[(critical > low) & (critical < high)]]
            if np.any(~np.isfinite(poly(probes))):
                raise ValueError("Nonfinite interpolation")
            if np.min(derivative(probes)) < -1e-12:
                result["warnings"].append(label + " interpolation has a negative slope (Akima overshoot)")
        # Polynomial roots on every union segment detect even very narrow crossings.
        knots = np.unique(np.r_[low, high, ax[(ax > low) & (ax < high)], bx[(bx > low) & (bx < high)]])
        crossing = []
        for left, right in zip(knots[:-1], knots[1:]):
            coeff = np.array([float(b.derivative(k)(left) - a.derivative(k)(left)) / math.factorial(k)
                              for k in (3, 2, 1, 0)])
            for root in np.roots(np.trim_zeros(coeff, "f")) if np.any(coeff) else []:
                if abs(root.imag) < 1e-9 and 0 < root.real < right - left:
                    crossing.append(float(left + root.real))
        diff_knots = b(knots) - a(knots)
        if crossing or (np.any(diff_knots > 1e-12) and np.any(diff_knots < -1e-12)):
            result["warnings"].append("Curves cross; BD-rate averages gains and losses")
        delta = float((b.integrate(low, high) - a.integrate(low, high)) / (high - low))
        value = float(np.expm1(delta * np.log(10.0)) * 100.0)
        if not np.isfinite(value):
            raise ValueError("Nonfinite integrated BD-rate")
        result.update(status="ok", bd_rate_percent=value, mean_log10_rate_delta=delta,
                      crossings_psnr=crossing)
    except (ValueError, KeyError, OverflowError) as exc:
        result["error"] = str(exc)
    return result


def compare_sampling(anchor, mine, indices, **kwargs):
    """Full and coarse results on their SAME intersection, plus ordinary ranges."""
    ar, aq, br, bq = map(np.asarray, (anchor[0], anchor[1], mine[0], mine[1]))
    dense = bd_rate(ar, aq, br, bq, **kwargs)
    coarse = bd_rate(ar[indices], aq[indices], br[indices], bq[indices], **kwargs)
    out = dict(dense=dense, coarse=coarse)
    if dense["status"] == coarse["status"] == "ok":
        overlap = (max(dense["psnr_low"], coarse["psnr_low"]), min(dense["psnr_high"], coarse["psnr_high"]))
        common_dense = bd_rate(ar, aq, br, bq, interval=overlap, **kwargs)
        common_coarse = bd_rate(ar[indices], aq[indices], br[indices], bq[indices], interval=overlap, **kwargs)
        out.update(common_dense=common_dense, common_coarse=common_coarse)
        if common_dense["status"] == common_coarse["status"] == "ok":
            out["coarse_minus_dense_percentage_points"] = common_coarse["bd_rate_percent"] - common_dense["bd_rate_percent"]
    return out
