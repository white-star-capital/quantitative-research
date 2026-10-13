import json

import pytest

from terminal_models.core.paths import contracts_dir
from terminal_models.core.scope import scope_key

VECTORS = json.loads((contracts_dir() / "fixtures" / "scope_key_vectors.json").read_text())


@pytest.mark.parametrize("vector", VECTORS, ids=lambda v: v["key"] or "<empty>")
def test_scope_key_matches_shared_vectors(vector):
    assert scope_key(vector["scope"]) == vector["key"]


def test_scope_key_ignores_insertion_order():
    assert scope_key({"b": "2", "a": "1"}) == scope_key({"a": "1", "b": "2"})
