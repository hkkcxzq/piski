"""Application configuration.

Non-secret settings come from YAML files (``config/default.yaml`` plus an optional
local override). Secrets come **only** from environment variables: a secret-looking
key inside a YAML file is a configuration error, so keys never end up in git.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
LOCAL_CONFIG_PATH = Path("config/local.yaml")

_SECRET_MARKERS = ("secret", "api_key", "apikey", "password", "token", "private_key")


class ConfigError(ValueError):
    """Raised when configuration is invalid or unsafe."""


class TradingMode(StrEnum):
    BACKTEST = "backtest"
    PAPER = "paper"
    DEMO = "demo"
    LIVE = "live"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RiskLimits(_Frozen):
    """Portfolio-level risk limits, expressed as fractions of equity."""

    max_position_risk: float = Field(0.0025, gt=0, le=0.02)
    max_total_risk: float = Field(0.01, gt=0, le=0.05)
    max_daily_loss: float = Field(0.02, gt=0, le=0.1)
    max_weekly_loss: float = Field(0.05, gt=0, le=0.2)
    max_drawdown: float = Field(0.10, gt=0, le=0.5)
    max_leverage: float = Field(5.0, gt=0, le=20)
    max_correlated_exposure: float = Field(0.02, gt=0, le=0.1)

    @model_validator(mode="after")
    def _consistent(self) -> RiskLimits:
        if self.max_position_risk > self.max_total_risk:
            raise ValueError("max_position_risk must not exceed max_total_risk")
        if self.max_daily_loss > self.max_weekly_loss:
            raise ValueError("max_daily_loss must not exceed max_weekly_loss")
        if self.max_weekly_loss > self.max_drawdown:
            raise ValueError("max_weekly_loss must not exceed max_drawdown")
        return self


class UniverseConfig(_Frozen):
    """Instruments: ``trade`` may receive orders, ``record`` is data collection only."""

    trade: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")
    record: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XAUTUSDT", "XAUUSDT")

    @model_validator(mode="after")
    def _trade_subset_of_record(self) -> UniverseConfig:
        missing = set(self.trade) - set(self.record)
        if missing:
            raise ValueError(f"traded symbols must also be recorded: {sorted(missing)}")
        return self


class HyperliquidConfig(_Frozen):
    api_url: str = "https://api.hyperliquid.xyz"
    stats_url: str = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"
    # Hyperliquid allows 1200 request-weight per minute per IP; we use a polite share.
    weight_budget_per_minute: int = Field(800, gt=0, le=1200)
    timeout_seconds: float = Field(30.0, gt=0)
    max_retries: int = Field(5, ge=0, le=10)


class TraderResearchConfig(_Frozen):
    top_by_all_time: int = Field(100, ge=0)
    top_by_month: int = Field(50, ge=0)
    control_sample: int = Field(60, ge=0)
    min_account_value: float = Field(10_000.0, ge=0)
    min_all_time_volume: float = Field(1_000_000.0, ge=0)
    min_trades: int = Field(50, ge=1)
    min_history_days: float = Field(30.0, ge=0)
    refresh_after_hours: float = Field(24.0, ge=0)
    fills_lookback_days: float = Field(180.0, gt=0)
    max_fills_per_trader: int = Field(20_000, ge=2_000)
    entry_cooldown_minutes: float = Field(60.0, ge=0)
    outcome_horizon_minutes: float = Field(60.0, gt=0)
    # per-address funding history is expensive (hourly rows per coin); 0 disables it.
    # Account-level metrics include funding anyway via the venue's PnL history.
    user_funding_days: float = Field(0.0, ge=0)
    context_coins: tuple[str, ...] = ("BTC", "ETH", "SOL")
    fdr_q: float = Field(0.05, gt=0, lt=1)
    min_supporting_traders: int = Field(3, ge=1)
    random_seed: int = 20260927


class Settings(_Frozen):
    mode: TradingMode = TradingMode.BACKTEST
    live_enabled: bool = False
    data_dir: Path = Path("data")
    log_level: str = "INFO"
    log_json: bool = True
    universe: UniverseConfig = UniverseConfig()
    risk: RiskLimits = RiskLimits()
    hyperliquid: HyperliquidConfig = HyperliquidConfig()
    trader_research: TraderResearchConfig = TraderResearchConfig()
    # Secrets: populated from the environment only.
    bybit_api_key: SecretStr | None = None
    bybit_api_secret: SecretStr | None = None
    telegram_bot_token: SecretStr | None = None

    @model_validator(mode="after")
    def _live_guard(self) -> Settings:
        if self.mode is TradingMode.LIVE and not self.live_enabled:
            raise ValueError(
                "LIVE mode requires live_enabled=true; real-money trading is locked until "
                "a strategy has passed demo trading and the owner explicitly enables it"
            )
        return self


def _find_secret_keys(data: Mapping[str, Any], prefix: str = "") -> list[str]:
    found: list[str] = []
    for key, value in data.items():
        path = f"{prefix}{key}"
        if any(marker in str(key).lower() for marker in _SECRET_MARKERS):
            found.append(path)
        if isinstance(value, Mapping):
            found.extend(_find_secret_keys(value, prefix=f"{path}."))
    return found


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    secrets = _find_secret_keys(loaded)
    if secrets:
        raise ConfigError(f"{path}: secrets must come from environment variables, found {secrets}")
    return loaded


_ENV_SECRETS = {
    "bybit_api_key": "QUANT_BYBIT_API_KEY",
    "bybit_api_secret": "QUANT_BYBIT_API_SECRET",
    "telegram_bot_token": "QUANT_TELEGRAM_BOT_TOKEN",
}


def load_settings(
    paths: list[Path] | None = None,
    env: Mapping[str, str] | None = None,
) -> Settings:
    """Load settings from YAML files (later files override earlier ones) and env secrets."""
    env = os.environ if env is None else env
    if paths is None:
        paths = [DEFAULT_CONFIG_PATH, LOCAL_CONFIG_PATH]
        extra = env.get("QUANT_CONFIG")
        if extra:
            paths.append(Path(extra))
    data: dict[str, Any] = {}
    for path in paths:
        if path.exists():
            data = _deep_merge(data, _read_yaml(path))
    if mode := env.get("QUANT_MODE"):
        data["mode"] = mode
    for field_name, env_name in _ENV_SECRETS.items():
        if value := env.get(env_name):
            data[field_name] = value
    try:
        return Settings.model_validate(data)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
