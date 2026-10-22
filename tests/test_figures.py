import xml.etree.ElementTree as ET

import matplotlib.image as mpimg
import pytest

from expected_goals_lab import figures

CAPTION = (
    "Wyscout open event data, five domestic leagues, 2017/18. Validation only: 100 shots, "
    "10 goals, "
    "matches played 2018-02-26 to 2018-04-13. The later test window is not used."
)


def comparison():
    return {
        "headline": "Adding the body-part tag improved validation log loss and Brier score",
        "subtitle": "Selected versus benchmark, 100 validation shots",
        "reference": {"log_loss": 0.30, "brier": 0.09, "auc": 0.76},
        "selected": {"log_loss": 0.29, "brier": 0.088, "auc": 0.78},
        "diff_log_loss": (-0.015, -0.005),
        "diff_brier": (-0.004, -0.001),
        "caption": CAPTION,
    }


def calibration(n_bins=10):
    rows = [
        {
            "shots": 10,
            "goals": i,
            "mean_prediction": 0.02 + 0.04 * i,
            "observed": 0.1 * i / 2,
            "low": max(0.0, 0.1 * i / 2 - 0.05),
            "high": 0.1 * i / 2 + 0.05,
        }
        for i in range(n_bins)
    ]
    tally = {"expected_goals": 10.5, "goals": 10}
    return {
        "headline": "Predicted and observed goal rates by probability bin",
        "subtitle": "8 of 10 bins inside the interval",
        "calibration": {"reference": rows, "selected": rows},
        "tally": {"reference": tally, "selected": tally},
        "caption": CAPTION,
    }


def raw_row(cls, label, shots, goals, median):
    rate = goals / shots if shots else None
    return {
        "class": cls,
        "label": label,
        "shots": shots,
        "goals": goals,
        "conversion": rate,
        "low": rate - 0.03 if rate is not None else None,
        "high": rate + 0.03 if rate is not None else None,
        "median_distance": median,
        "q25_distance": median - 4 if median else None,
        "q75_distance": median + 4 if median else None,
    }


def body_part(empty_class=False, sparse=False):
    rows = [
        raw_row("right_foot", "right foot", 50, 5, 19.0),
        raw_row(
            "left_foot",
            "left foot",
            0 if empty_class else 30,
            0 if empty_class else 3,
            None if empty_class else 18.0,
        ),
        raw_row("head_or_body", "head/body (tag 403)", 20, 3, 10.5),
    ]
    cells = lambda p: [  # noqa: E731
        {
            "class": c,
            "label": c,
            "probability": p,
            "training_shots_nearby": 5 if sparse else 200,
            "sparse": sparse,
        }
        for c in ("right_foot", "left_foot", "head_or_body")
    ]
    return {
        "headline": "Body-part tag: observed goal rates and location-adjusted predictions",
        "subtitle": "Wyscout tag classes (tag 403, not a confirmed header)",
        "raw": rows,
        "overall_rate": 0.1,
        "illustrations": [
            {"location": "central, penalty-spot area", "distance_m": 10.5, "cells": cells(0.2)},
            {"location": "central, long range", "distance_m": 26.2, "cells": cells(0.02)},
        ],
        "caption": CAPTION,
    }


def assert_valid_pair(paths):
    png, svg = paths
    assert png.suffix == ".png" and svg.suffix == ".svg"
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    image = mpimg.imread(png)
    assert image.shape[0] > 400 and image.shape[1] > 800
    assert ET.parse(svg).getroot().tag.endswith("svg")


@pytest.mark.parametrize(
    ("function", "data", "stem"),
    [
        (figures.plot_model_comparison, comparison(), "validation_model_comparison"),
        (figures.plot_calibration, calibration(), "validation_calibration"),
        (figures.plot_body_part, body_part(), "body_part_and_location"),
    ],
)
def test_each_figure_writes_a_valid_png_and_svg_at_the_requested_path(
    tmp_path, function, data, stem
):
    paths = function(tmp_path / "figs" / stem, data)
    assert [p.name for p in paths] == [f"{stem}.png", f"{stem}.svg"]
    assert_valid_pair(paths)


@pytest.mark.parametrize(
    ("function", "data"),
    [
        (figures.plot_model_comparison, comparison()),
        (figures.plot_calibration, calibration()),
        (figures.plot_body_part, body_part()),
    ],
)
def test_figures_are_byte_for_byte_deterministic(tmp_path, function, data):
    first = function(tmp_path / "a" / "f", data)
    second = function(tmp_path / "b" / "f", data)
    for a, b in zip(first, second, strict=True):
        assert a.read_bytes() == b.read_bytes()


def test_sparse_and_empty_groups_still_render(tmp_path):
    assert_valid_pair(figures.plot_body_part(tmp_path / "empty", body_part(empty_class=True)))
    assert_valid_pair(figures.plot_body_part(tmp_path / "sparse", body_part(sparse=True)))
    assert_valid_pair(figures.plot_calibration(tmp_path / "few", calibration(n_bins=3)))


def test_captions_state_source_validation_only_and_the_tag_caveat(tmp_path, monkeypatch):
    seen: dict[str, str] = {}
    original = figures._caption

    def spy(fig, text, width=175):
        seen[len(seen).__str__()] = text
        return original(fig, text, width)

    monkeypatch.setattr(figures, "_caption", spy)
    figures.plot_model_comparison(tmp_path / "a", comparison())
    figures.plot_calibration(tmp_path / "b", calibration())
    figures.plot_body_part(tmp_path / "c", body_part())

    captions = list(seen.values())
    assert len(captions) == 3
    for text in captions:
        assert "CC BY 4.0" in text and "doi:10.6084/m9.figshare.c.4415000.v5" in text
        assert "Validation only" in text and "test window is not used" in text
        assert "processed for this project" in text
    assert "No calibration was fitted" in captions[1]
    assert "does not start at zero" in captions[0] or "do not start at zero" in captions[0]
    body = captions[2]
    assert "not a confirmed header" in body and "not an observed rate" in body
    assert "not a measured effect" in body
