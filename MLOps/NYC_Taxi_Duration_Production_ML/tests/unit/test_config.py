from pathlib import Path

from taxi_duration.config import canonical_config_json, load_config


def test_inherited_config_overrides_base() -> None:
    config = load_config(Path("configs/development.yaml"))
    assert config.environment == "development"
    assert config.model.max_iter == 80
    assert config.data.valid_zone_max == 263


def test_canonical_config_includes_resolved_base_values(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(Path("configs/base.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    child.write_text("extends: base.yaml\nenvironment: test\n", encoding="utf-8")
    before = canonical_config_json(load_config(child))
    base.write_text(
        base.read_text(encoding="utf-8").replace("random_seed: 42", "random_seed: 43"),
        encoding="utf-8",
    )
    after = canonical_config_json(load_config(child))
    assert before != after
