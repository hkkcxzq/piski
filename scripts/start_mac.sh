#!/usr/bin/env bash
# Запуск всего на Mac: новостной монитор, запись данных Bybit и торговый движок (демо).
# Mac не засыпает, пока работает скрипт (caffeinate). Остановка: Ctrl+C.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
set -a; [ -f .env ] && source .env; set +a
export QUANT_MODE="${QUANT_MODE:-demo}"
mkdir -p data/logs
trap 'kill 0' EXIT
caffeinate -dimsu &
quant record >> data/logs/record.log 2>&1 &
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
  quant news watch --interval 180 >> data/logs/news.log 2>&1 &
else
  echo "ANTHROPIC_API_KEY не задан — новостной монитор не запущен (фильтр ФРС всё равно действует)"
fi
echo "Движок запущен (режим: $QUANT_MODE). Логи: data/logs/. Статус: quant live status. Аварийно: quant live kill"
quant live run --interval 60 2>&1 | tee -a data/logs/live.log
