from pathlib import Path

import pytest

from quant.app.cli import main


def test_analyze_on_empty_data_produces_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = tmp_path / "local.yaml"
    cfg.write_text(f"data_dir: {tmp_path.as_posix()}\nlog_json: false\n", encoding="utf-8")
    publish = tmp_path / "published"
    assert main(["--config", str(cfg), "traders", "analyze", "--publish", str(publish)]) == 0
    report_path = Path(capsys.readouterr().out.strip())
    assert report_path.exists()
    assert "не найдено" in report_path.read_text(encoding="utf-8")
    assert len(list(publish.glob("traders-*.md"))) == 1


def test_bad_config_exits_with_code_2(tmp_path: Path) -> None:
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("mode: live\n", encoding="utf-8")
    assert main(["--config", str(cfg), "traders", "analyze"]) == 2
