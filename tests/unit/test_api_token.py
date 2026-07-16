"""Unit tests for the HTTP API bearer-token provisioning helper.

Guards audit finding A5 — the launchd HTTP service must fail closed with a
per-install token even when nothing is configured. No Neo4j / network needed.
"""

import stat

import pytest

from atlas_core.api.http_server import (
    API_TOKEN_ENV,
    load_or_create_api_token,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv(API_TOKEN_ENV, raising=False)


def test_env_token_takes_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv(API_TOKEN_ENV, "env-token-123")
    assert load_or_create_api_token(tmp_path) == "env-token-123"
    # env wins outright — no file is written.
    assert not (tmp_path / "api_token").exists()


def test_env_token_blank_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv(API_TOKEN_ENV, "   ")
    token = load_or_create_api_token(tmp_path)
    assert token
    assert token.strip() == token


def test_mints_and_persists_token(tmp_path):
    token = load_or_create_api_token(tmp_path)
    assert token
    token_file = tmp_path / "api_token"
    assert token_file.exists()
    assert token_file.read_text(encoding="utf-8").strip() == token


def test_token_file_is_owner_only(tmp_path):
    load_or_create_api_token(tmp_path)
    mode = stat.S_IMODE((tmp_path / "api_token").stat().st_mode)
    assert mode == 0o600


def test_reuses_existing_token(tmp_path):
    first = load_or_create_api_token(tmp_path)
    second = load_or_create_api_token(tmp_path)
    assert first == second


def test_creates_missing_data_dir(tmp_path):
    nested = tmp_path / "does" / "not" / "exist"
    token = load_or_create_api_token(nested)
    assert token
    assert (nested / "api_token").exists()
