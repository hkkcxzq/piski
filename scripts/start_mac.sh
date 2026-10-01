#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# venv необязателен: если его нет, используется quant из PATH
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi
if ! command -v quant >/dev/null 2>&1; then
  echo "Команда quant не найдена. Выполните в этой папке:"
  echo "  python3.12 -m venv .venv && source .venv/bin/activate && pip install -e '.[dev]'"
  exit 1
fi
set -a; [ -f .env ] && source .env; set +a
export QUANT_MODE="${QUANT_MODE:-demo}"
mkdir -p data/logs
trap 'kill 0' EXIT
caffeinate -ims &  # не даём Mac уснуть, экран при этом может гаснуть
quant record >> data/logs/record.log 2>&1 &
quant news watch --interval 180 >> data/logs/news.log 2>&1 &
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
  echo "Новости: оценивает Claude"
else
  echo "Новости: бесплатные правила по ключевым словам (ANTHROPIC_API_KEY не задан)"
fi
echo "Движок запущен (режим: $QUANT_MODE). Логи: data/logs/. Статус: quant live status. Аварийно: quant live kill"
quant live run --interval 60 2>&1 | tee -a data/logs/live.log
