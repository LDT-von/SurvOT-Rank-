#!/bin/bash
# 每 60 秒刷新一次训练进度
# 用法: bash scripts/watch_v38_20ep.sh
while true; do
    clear
    python3 scripts/monitor_v38_20ep.py
    sleep 60
done
