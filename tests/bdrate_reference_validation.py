"""Compare independent Akima integration against an unmodified external wheel.

Download with pip download --no-deps bjontegaard==1.3.0 to a NEW directory.
This validator does not install/modify packages in any existing environment.
"""
import argparse
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import numpy as np
    from utils.bdrate.io import save_json, sha256
    from utils.bdrate.curves import bd_rate
    p = argparse.ArgumentParser()
    p.add_argument("--reference-wheel", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    sys.path.insert(0, args.reference_wheel)
    import bjontegaard as reference
    rng = np.random.RandomState(75)
    differences = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for _ in range(100):
            qa = np.sort(rng.uniform(30, 45, 6)); qb = np.sort(rng.uniform(31, 44, 10))
            ra = np.exp((qa-30)/8 + np.sort(rng.uniform(0, .1, 6)))
            rb = np.exp((qb-30)/8 + np.sort(rng.uniform(0, .1, 10))) * .8
            own = bd_rate(ra, qa, rb, qb)
            ref = reference.bd_rate(ra, qa, rb, qb, method="akima", require_matching_points=False)
            if own["status"] != "ok":
                raise RuntimeError(own)
            differences.append(abs(own["bd_rate_percent"] - ref))
    maximum = max(differences)
    if maximum > 1e-9:
        raise AssertionError("BD-rate reference disagreement: {} percentage points".format(maximum))
    result = dict(reference="FAU-LMS bjontegaard 1.3.0", reference_wheel_sha256=sha256(args.reference_wheel),
                  cases=100, seed=75, maximum_absolute_difference_percentage_points=maximum,
                  tolerance_percentage_points=1e-9, passed=True)
    save_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
