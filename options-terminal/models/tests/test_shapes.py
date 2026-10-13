"""C-4 cross-field rules and composite recursion. Structural values only, no market data."""

import pytest

from terminal_models.core.validation import ValidationFailure, validate_payload_full

UNIT = {"quantity": "implied_vol", "scale": "decimal"}
AXIS = {
    "label": "k",
    "unit": {"quantity": "moneyness_log", "scale": "absolute"},
    "values": [-0.1, 0, 0.1],
}


def curve(values):
    return {"x": AXIS, "series": [{"key": "s", "label": "s", "unit": UNIT, "values": values}]}


def table(rows, types=("number", "boolean")):
    cols = [{"key": f"c{i}", "label": "c", "type": t} for i, t in enumerate(types)]
    return {"columns": cols, "rows": rows}


@pytest.mark.parametrize(
    "kind, payload",
    [
        ("curve", curve([None, 0.5, None])),
        (
            "surface",
            {"x": AXIS, "y": AXIS, "z": {"label": "iv", "unit": UNIT, "values": [[None] * 3] * 3}},
        ),
        ("table", table([[1.5, True], [None, None]])),
        (
            "scalar_set",
            {"items": [{"key": "a", "label": "A", "value": True, "unit": UNIT, "flags": ["x"]}]},
        ),
        (
            "timeseries",
            {
                "t": ["2026-10-13T17:00:00Z", "2026-10-13T18:00:00Z"],
                "series": [{"key": "s", "label": "s", "unit": UNIT, "values": [None, 1]}],
            },
        ),
    ],
)
def test_valid_shapes(kind, payload):
    validate_payload_full(kind, "1", payload)


@pytest.mark.parametrize(
    "kind, payload, code",
    [
        ("curve", curve([0.5, 0.5]), "invalid_shape"),
        (
            "surface",
            {"x": AXIS, "y": AXIS, "z": {"label": "iv", "unit": UNIT, "values": [[None] * 3] * 2}},
            "invalid_shape",
        ),
        ("table", table([[1.5]]), "invalid_shape"),
        ("table", table([[True, True]]), "invalid_shape"),  # bool is not a number
        ("table", table([["x", True]]), "invalid_shape"),
        (
            "timeseries",
            {"t": ["2026-10-13T18:00:00Z", "2026-10-13T17:00:00Z"], "series": []},
            "invalid_shape",
        ),
        (
            "curve",
            {
                "x": AXIS,
                "series": [
                    {
                        "key": "s",
                        "label": "s",
                        "unit": {"quantity": "vibes", "scale": "decimal"},
                        "values": [1, 2, 3],
                    }
                ],
            },
            "invalid_payload",
        ),
    ],
)
def test_invalid_shapes(kind, payload, code):
    with pytest.raises(ValidationFailure) as exc:
        validate_payload_full(kind, "1", payload)
    assert exc.value.code == code


def test_composite_recurses_and_names_section():
    payload = {
        "sections": [
            {
                "key": "smile",
                "label": "Smile",
                "payload_kind": "curve",
                "payload_version": "1",
                "payload": curve([0.5, 0.5]),
            },
        ]
    }
    with pytest.raises(ValidationFailure) as exc:
        validate_payload_full("composite", "1", payload)
    assert "section smile" in exc.value.message


def test_composite_cannot_nest():
    inner = {"sections": []}
    payload = {
        "sections": [
            {
                "key": "c",
                "label": "c",
                "payload_kind": "composite",
                "payload_version": "1",
                "payload": inner,
            }
        ]
    }
    with pytest.raises(ValidationFailure):
        validate_payload_full("composite", "1", payload)
