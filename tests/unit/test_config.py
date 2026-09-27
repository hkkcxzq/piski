from pathlib import Path

import pytest

from quant.app.config import ConfigError, RiskLimits, Settings, TradingMode, load_settings


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "c.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_defaults_are_safe() -> None:
    s = Settings()
    assert s.mode is TradingMode.BACKTEST
    assert not s.live_enabled
    assert s.risk.max_position_risk == 0.0025
    assert s.risk.max_daily_loss == 0.02
    assert set(s.universe.trade) <= set(s.universe.record)


def test_repo_default_config_loads() -> None:
    s = load_settings([Path("config/default.yaml")], env={})
    assert s.universe.trade == ("BTCUSDT", "ETHUSDT")
    assert "XAUTUSDT" in s.universe.record


def test_live_mode_is_locked_by_default(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="live_enabled"):
        load_settings([_write(tmp_path, "mode: live\n")], env={})


def test_live_mode_via_env_is_also_locked(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_settings([_write(tmp_path, "mode: demo\n")], env={"QUANT_MODE": "live"})


def test_secret_in_yaml_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="environment"):
        load_settings([_write(tmp_path, "exchange:\n  api_key: abc\n")], env={})


def test_secrets_come_from_env(tmp_path: Path) -> None:
    s = load_settings(
        [_write(tmp_path, "mode: demo\n")], env={"QUANT_BYBIT_API_KEY": "k", "QUANT_BYBIT_API_SECRET": "s"}
    )
    assert s.mode is TradingMode.DEMO
    assert s.bybit_api_key is not None and s.bybit_api_key.get_secret_value() == "k"
    assert "k" not in repr(s.bybit_api_key)


def test_later_files_override_earlier(tmp_path: Path) -> None:
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("risk:\n  max_daily_loss: 0.02\n  max_weekly_loss: 0.05\n", encoding="utf-8")
    b.write_text("risk:\n  max_daily_loss: 0.01\n", encoding="utf-8")
    s = load_settings([a, b], env={})
    assert s.risk.max_daily_loss == 0.01
    assert s.risk.max_weekly_loss == 0.05


def test_inconsistent_risk_limits_rejected() -> None:
    with pytest.raises(ValueError, match="max_position_risk"):
        RiskLimits(max_position_risk=0.02, max_total_risk=0.01)


def test_traded_symbols_must_be_recorded(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="recorded"):
        load_settings([_write(tmp_path, "universe:\n  trade: [DOGEUSDT]\n  record: [BTCUSDT]\n")], env={})


def test_unknown_keys_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_settings([_write(tmp_path, "risk:\n  max_dayly_loss: 0.01\n")], env={})
