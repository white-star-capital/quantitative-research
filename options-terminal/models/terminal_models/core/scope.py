"""Canonical scope_key (CONTRACTS C-9).

Shared vectors with the API: contracts/fixtures/scope_key_vectors.json.
"""

from urllib.parse import quote


def scope_key(scope: dict[str, str]) -> str:
    """Keys sorted, `k=v` joined by `&`, keys and values percent-encoded (RFC 3986)."""
    return "&".join(f"{quote(k, safe='')}={quote(v, safe='')}" for k, v in sorted(scope.items()))
