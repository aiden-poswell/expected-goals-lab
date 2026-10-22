"""Three static validation figures (PNG and SVG). Aggregates only; no shot-level data.

Colour system: blue is the selected model, grey the benchmark, orange an observed statistic.
Every figure carries its data source, the validation-only status and the CC BY attribution.
Output is deterministic: fixed fonts, no timestamps or version strings in the metadata.
"""

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, GREY, ORANGE, INK, LIGHT = "#1b5e9b", "#8a8f98", "#d1731f", "#222222", "#d9dce1"
MARKERS = {"right_foot": "o", "left_foot": "s", "head_or_body": "^"}
PNG_DPI = 200

ATTRIBUTION = (
    "Data: Pappalardo & Massucco (2019), Wyscout open event data, CC BY 4.0, "
    "doi:10.6084/m9.figshare.c.4415000.v5; processed for this project."
)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#555555",
            "axes.titleweight": "bold",
            "svg.fonttype": "path",
            "svg.hashsalt": "expected-goals-lab",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _caption(fig, text: str, width: int = 175) -> None:
    fig.text(
        0.012,
        0.012,
        textwrap.fill(text, width),
        fontsize=8,
        color="#555555",
        ha="left",
        va="bottom",
    )


def _headline(fig, title: str, subtitle: str) -> None:
    fig.text(0.012, 0.965, title, fontsize=13.5, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.012, 0.915, subtitle, fontsize=9.5, color="#444444", ha="left", va="top")


def _save(fig, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png, svg = stem.with_suffix(".png"), stem.with_suffix(".svg")
    fig.savefig(png, format="png", dpi=PNG_DPI, metadata={"Software": None})
    fig.savefig(svg, format="svg", metadata={"Date": None, "Creator": None, "Format": None})
    plt.close(fig)
    return [png, svg]


def _signed(x: float, digits: int) -> str:
    return f"{x:+.{digits}f}".replace("-", "−")


def plot_model_comparison(stem: Path, d: dict) -> list[Path]:
    """Benchmark versus selected model: log loss, Brier score, AUC on validation."""
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(11, 6.4))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.78, bottom=0.38, wspace=0.42)
    _headline(fig, d["headline"], d["subtitle"])
    panels = [
        ("log_loss", "Log loss", "lower is better", 4, d.get("diff_log_loss")),
        ("brier", "Brier score", "lower is better", 5, d.get("diff_brier")),
        ("auc", "ROC AUC", "higher is better", 3, None),
    ]
    for ax, (key, title, direction, digits, interval) in zip(axes, panels, strict=True):
        ref, sel = d["reference"][key], d["selected"][key]
        ax.plot([0, 1], [ref, sel], color=LIGHT, lw=2, zorder=1)
        ax.scatter([0], [ref], s=110, color=GREY, zorder=3)
        ax.scatter([1], [sel], s=110, color=BLUE, zorder=3)
        pad = max(abs(sel - ref) * 2.2, 1e-6)
        ax.set_ylim(min(ref, sel) - pad, max(ref, sel) + pad)
        ax.set_xlim(-0.6, 1.6)
        ax.set_xticks(
            [0, 1], ["Benchmark\ndistance + angle", "Selected\nspline location\n+ body-part tag"]
        )
        ax.tick_params(axis="x", length=0)
        ax.set_title(f"{title}\n({direction})", loc="left")
        for x, value, colour in ((0, ref, GREY), (1, sel, BLUE)):
            ax.annotate(
                f"{value:.{digits}f}",
                (x, value),
                xytext=(0, 12),
                textcoords="offset points",
                ha="center",
                fontsize=10,
                fontweight="bold",
                color=colour,
            )
        delta = sel - ref
        note = f"Selected − benchmark: {_signed(delta, digits)}"
        if interval is not None:
            low, high = _signed(interval[0], digits), _signed(interval[1], digits)
            note += f"\n95% match-bootstrap interval\n[{low}, {high}]"
        else:
            note += "\n(no interval computed)"
        ax.text(0.0, -0.32, note, transform=ax.transAxes, fontsize=9, color=INK, va="top")
    _caption(
        fig,
        f"{d['caption']} Value axes do not start at zero and each panel has its own scale; read "
        "the "
        "labelled values. Intervals are exploratory and unadjusted, from a bootstrap of "
        "validation matches, and do not cover selection among candidates, dependence between "
        f"matches of one team, or other seasons. {ATTRIBUTION}",
    )
    return _save(fig, stem)


def plot_calibration(stem: Path, d: dict) -> list[Path]:
    """Observed versus predicted rate in ten equal-frequency validation bins, per model."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 6.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.83, bottom=0.215, wspace=0.12)
    _headline(fig, d["headline"], d["subtitle"])
    top = 0.45
    for ax, (key, title, colour) in zip(
        axes,
        (
            ("reference", "Benchmark: distance + angle", GREY),
            ("selected", "Selected: spline location + body-part tag", BLUE),
        ),
        strict=True,
    ):
        rows = d["calibration"][key]
        ax.plot([0, top], [0, top], color=ORANGE, lw=1.2, ls="--", zorder=1)
        ax.text(
            top * 0.62,
            top * 0.62 - 0.035,
            "perfect agreement",
            color=ORANGE,
            fontsize=8.5,
            rotation=38,
        )
        for r in rows:
            ax.errorbar(
                r["mean_prediction"],
                r["observed"],
                yerr=[[max(0.0, r["observed"] - r["low"])], [max(0.0, r["high"] - r["observed"])]],
                fmt="o",
                color=colour,
                ecolor=colour,
                elinewidth=1.2,
                capsize=2.5,
                ms=6,
                zorder=3,
            )
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_aspect("equal")
        ax.set_title(title, loc="left")
        ax.set_xlabel("Mean predicted probability in bin")
        tally = d["tally"][key]
        fewest, most = min(r["shots"] for r in rows), max(r["shots"] for r in rows)
        per_bin = f"{fewest}" if fewest == most else f"{fewest}–{most}"
        ax.text(
            0.03,
            0.96,
            f"Sum of predictions: {tally['expected_goals']:.0f}\nObserved goals: {tally['goals']}\n"
            f"{per_bin} shots per bin",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            color=INK,
        )
    axes[0].set_ylabel("Observed goal rate in bin (95% Wilson interval)")
    _caption(
        fig,
        f"{d['caption']} Ten equal-frequency bins of the validation predictions. No calibration "
        "was "
        "fitted and no parameter was chosen on these data. Intervals treat shots as independent, "
        f"so they understate uncertainty from shots within a match. {ATTRIBUTION}",
    )
    return _save(fig, stem)


def plot_body_part(stem: Path, d: dict) -> list[Path]:
    """Observed rates, where the shots were taken, and the model's predictions at fixed spots."""
    _style()
    fig, axes = plt.subplots(
        1, 3, figsize=(14.5, 6.4), gridspec_kw={"width_ratios": [1.0, 0.9, 2.2]}
    )
    fig.subplots_adjust(left=0.115, right=0.99, top=0.78, bottom=0.30, wspace=0.5)
    _headline(fig, d["headline"], d["subtitle"])
    rows = d["raw"]
    ys = list(range(len(rows)))
    labels = [f"{r['label']}\n{r['shots']:,} shots, {r['goals']} goals" for r in rows]

    ax = axes[0]
    ax.set_title("What was observed:\ngoal rate by body-part tag", loc="left")
    for y, r in zip(ys, rows, strict=True):
        if r["conversion"] is None:
            ax.text(0.01, y, "no shots", va="center", color=GREY)
            continue
        ax.plot([r["low"], r["high"]], [y, y], color=ORANGE, lw=2)
        ax.scatter([r["conversion"]], [y], s=80, color=ORANGE, zorder=3)
        ax.annotate(
            f"{100 * r['conversion']:.1f}%",
            (r["conversion"], y),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=INK,
            fontweight="bold",
        )
    ax.axvline(d["overall_rate"], color=GREY, ls=":", lw=1.2)
    ax.text(
        d["overall_rate"],
        len(rows) - 0.4,
        "dotted line: all\nvalidation shots",
        color="#666666",
        fontsize=8.5,
        ha="center",
        va="center",
    )
    ax.set_yticks(ys, labels)
    ax.set_ylim(len(rows) - 0.1, -0.7)  # room above the first row for its value label
    ax.set_xlim(0, max(0.2, max((r["high"] or 0) for r in rows) * 1.15))
    ax.set_xlabel("Goal rate (95% Wilson interval)")

    ax = axes[1]
    ax.set_title("Where the shots were taken:\ndistance to goal centre", loc="left")
    for y, r in zip(ys, rows, strict=True):
        if r["median_distance"] is None:
            continue
        ax.plot([r["q25_distance"], r["q75_distance"]], [y, y], color=ORANGE, lw=2)
        ax.scatter([r["median_distance"]], [y], s=80, color=ORANGE, zorder=3)
        ax.annotate(
            f"{r['median_distance']:.1f} m",
            (r["median_distance"], y),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=INK,
            fontweight="bold",
        )
    ax.set_yticks(ys, [""] * len(rows))
    ax.set_ylim(len(rows) - 0.1, -0.7)
    ax.set_xlim(0, None)
    ax.set_xlabel("Median distance (m), middle half of shots")

    ax = axes[2]
    ax.set_title("Model prediction at fixed spots\n(model output, not observed)", loc="left")
    locations = d["illustrations"]
    top = max(c["probability"] for loc in locations for c in loc["cells"]) * 1.25
    offsets = {"right_foot": -0.27, "left_foot": 0.0, "head_or_body": 0.27}
    for i, loc in enumerate(locations):
        for cell in loc["cells"]:
            x = i + offsets[cell["class"]]
            ax.scatter(
                [x],
                [cell["probability"]],
                marker=MARKERS[cell["class"]],
                s=85,
                color=BLUE,
                facecolors="none" if cell["sparse"] else BLUE,
                linewidths=1.6,
                zorder=3,
            )
            ax.annotate(
                f"{100 * cell['probability']:.0f}%",
                (x, cell["probability"]),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color=BLUE,
            )
    ax.set_xticks(
        range(len(locations)),
        [f"{textwrap.fill(loc['location'], 14)}\n{loc['distance_m']:.1f} m" for loc in locations],
    )
    ax.set_xlim(-0.6, len(locations) - 0.4)
    ax.set_ylim(0, top)
    ax.set_ylabel("Predicted goal probability")
    ax.tick_params(axis="x", labelsize=8.5, length=0)
    handles = [
        plt.Line2D([], [], marker=MARKERS[c], ls="", color=BLUE, label=lab)
        for c, lab in (
            ("right_foot", "right foot"),
            ("left_foot", "left foot"),
            ("head_or_body", "head/body (tag 403)"),
        )
    ]
    handles.append(
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            color=BLUE,
            mfc="none",
            label="hollow: fewer than 30 training\nshots nearby (extrapolated)",
        )
    )
    ax.legend(handles=handles, loc="upper right", fontsize=8.5, frameon=False)
    _caption(
        fig,
        f"{d['caption']} Left and middle panels are observed validation shots. The right panel is "
        "the "
        "model's predicted probability at four fixed pitch positions, using coefficients fitted on "
        "training matches; it is not an observed rate and not a measured effect of changing how "
        "a shot "
        "is struck. Wyscout tag 403 is “head/body”, not a confirmed header. Shots are not "
        "assigned a body "
        f"part at random. {ATTRIBUTION}",
    )
    return _save(fig, stem)
