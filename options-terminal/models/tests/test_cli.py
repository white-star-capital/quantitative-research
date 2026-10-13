from terminal_models.runner.cli import main


def test_validate_registry_ok(capsys):
    assert main(["validate-registry"]) == 0
    assert "registry ok" in capsys.readouterr().out


def test_validate_registry_reports_problems(tmp_path, capsys):
    bad = tmp_path / "registry.yaml"
    bad.write_text(
        "slots:\n  vol_surface:\n    production: {implementation: vol_surface_ghost, "
        "version: 1.0.0}\n    scopes: [{underlying: BTC}]\n"
    )
    assert main(["validate-registry", "--registry", str(bad)]) == 1
    assert "not found in library" in capsys.readouterr().err


def test_unimplemented_commands_fail_loudly(capsys):
    assert main(["run", "vol_surface", "--scope", "underlying=BTC", "--dry-run"]) == 2
    assert "not implemented" in capsys.readouterr().err
