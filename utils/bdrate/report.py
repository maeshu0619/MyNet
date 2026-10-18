"""JSON/Excel/plots are views of full-precision measurements, not rate sources."""
import json
import math
from pathlib import Path
import numpy as np
from .curves import bd_rate, compare_sampling
from .io import json_safe, save_json


def summarize(rows, clouds, expected_points, coarse_indices, minimum_overlap_db=0):
    summaries, diagnostics = [], []
    for cloud in clouds:
        for metric in ("d1", "d2"):
            selected = {name: sorted([r for r in rows if r.get("cloud") == cloud and r.get("series") == name],
                                     key=lambda r: r["rate_index"]) for name in ("Baseline", "MINE")}
            reason = None
            for name, entries in selected.items():
                if len(entries) != expected_points or any(r.get("status") != "ok" for r in entries):
                    reason = "Incomplete {} curve: {}/{} successful points".format(name, sum(r.get("status") == "ok" for r in entries), expected_points)
                elif any(metric + "_psnr" not in r for r in entries):
                    reason = "{} not measured".format(metric)
            result = dict(status="invalid", bd_rate_percent=None, error=reason) if reason else bd_rate(
                [r["bpp"] for r in selected["Baseline"]], [r[metric + "_psnr"] for r in selected["Baseline"]],
                [r["bpp"] for r in selected["MINE"]], [r[metric + "_psnr"] for r in selected["MINE"]],
                minimum_overlap_db=minimum_overlap_db)
            summaries.append(dict(cloud=cloud, metric=metric, **result))
            if not reason and expected_points > len(coarse_indices):
                compare = compare_sampling(
                    ([r["bpp"] for r in selected["Baseline"]], [r[metric + "_psnr"] for r in selected["Baseline"]]),
                    ([r["bpp"] for r in selected["MINE"]], [r[metric + "_psnr"] for r in selected["MINE"]]),
                    coarse_indices, minimum_overlap_db=minimum_overlap_db)
                diagnostics.append(dict(cloud=cloud, metric=metric, **compare))
    for metric in ("d1", "d2"):
        eligible = [r["bd_rate_percent"] for r in summaries if r.get("metric") == metric and r["status"] == "ok"]
        # Never label a valid-only subset mean as a complete dataset result.
        summaries.append(dict(cloud="DATASET_MEAN", metric=metric,
                              status="ok" if len(eligible) == len(clouds) else "incomplete",
                              bd_rate_percent=float(np.mean(eligible)) if len(eligible) == len(clouds) else None,
                              valid_only_mean_percent=float(np.mean(eligible)) if eligible else None,
                              valid_clouds=len(eligible), expected_clouds=len(clouds),
                              averaging="arithmetic mean of per-cloud BD-rate; no pooled RD curve"))
    return summaries, diagnostics


def _cell(value):
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(json_safe(value), ensure_ascii=False)
        return text if len(text) <= 32000 else "Value exceeds Excel cell limit; see full summary.json / configuration.json"
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
        return "'" + value  # prevent Excel formula injection from filenames/logs
    return value


def write_report(folder, rows, summaries, diagnostics, metadata):
    from openpyxl import Workbook
    folder = Path(folder)
    save_json(folder / "summary.json", dict(metadata=metadata, summary=summaries, sampling=diagnostics))
    workbook = Workbook()
    workbook.remove(workbook.active)
    for name, items in (("Run status", [metadata.get("evaluation_status", {"status": "not recorded"})]),
                        ("BD-rate", summaries), ("RD points", rows), ("Sampling validation", diagnostics),
                        ("Provenance", [dict(key=k, value=v) for k, v in metadata.items()])):
        sheet = workbook.create_sheet(name)
        keys = list(dict.fromkeys(k for r in items for k in r))
        if keys:
            sheet.append(keys)
            for row in items:
                sheet.append([_cell(row.get(k)) for k in keys])
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for column in sheet.columns:
                sheet.column_dimensions[column[0].column_letter].width = min(52, max(14, len(str(column[0].value)) + 2))
    path = folder / "results.xlsx"
    if path.exists():
        raise FileExistsError(path)
    workbook.save(path)


def plot_curves(folder, rows, clouds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.interpolate import Akima1DInterpolator
    from .curves import prepare_curve
    for index, cloud in enumerate(clouds):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, metric in zip(axes, ("d1", "d2")):
            for name in ("Baseline", "MINE"):
                points = [r for r in rows if r.get("cloud") == cloud and r.get("series") == name and r.get("status") == "ok" and metric + "_psnr" in r]
                if not points:
                    continue
                rate = np.array([r["bpp"] for r in points]); quality = np.array([r[metric + "_psnr"] for r in points])
                finite = np.isfinite(rate) & np.isfinite(quality)
                ax.scatter(rate[finite], quality[finite], label=name)
                try:
                    x, y, _ = prepare_curve(rate, quality)
                    samples = np.linspace(x[0], x[-1], 300)
                    ax.plot(10 ** Akima1DInterpolator(x, y)(samples), samples)
                except ValueError:
                    pass  # Show invalid measurements, do not invent a repaired curve.
            ax.set(xlabel="bits / Original GT point", ylabel=metric.upper() + " PSNR [dB]")
            ax.grid(True)
            if ax.collections:
                ax.legend()
        fig.suptitle(Path(cloud).name)
        fig.tight_layout()
        fig.savefig(Path(folder) / ("rd_{:04d}.png".format(index)), dpi=150)
        plt.close(fig)
