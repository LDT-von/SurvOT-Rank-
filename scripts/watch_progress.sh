#!/usr/bin/env bash
# BRCA / DCT 训练进度监控 (每 30s 刷新)
# 用法: bash scripts/watch_progress.sh
# Ctrl+C 退出

watch -n 30 -c '
echo "================================="
echo " $(date +%H:%M:%S) — BRCA Recovery"
echo "================================="
echo ""
echo "存活进程: $(ps aux | grep survot_rank.cli | grep -v grep | wc -l)"
echo "GPU 显存: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || echo N/A)"
echo ""

printf "%-15s %5s %8s %8s %5s %5s\n" variant ep best_c latest_c best_ep left
echo "----------------------------------------------------------"
for v in a30 a30_legacy norank norank_legacy reg reg_legacy; do
  log="/home/ubuntu/SurvOT-Rank/logs/brca_recovery_${v}.log"
  [ ! -f "$log" ] && continue
  latest=$(grep "val cindex" "$log" 2>/dev/null | tail -1)
  ep=$(echo "$latest" | grep -oP "Epoch \K[0-9]+" || echo "?")
  lc=$(echo "$latest" | grep -oP "cindex=\K[0-9.]+" || echo "?")
  best_line=$(grep "val cindex" "$log" 2>/dev/null | sort -t= -k2 -nr | head -1)
  bc=$(echo "$best_line" | grep -oP "cindex=\K[0-9.]+" || echo "?")
  be=$(echo "$best_line" | grep -oP "Epoch \K[0-9]+" || echo "?")
  rem=$((50 - ep 2>/dev/null || 0))
  printf "%-15s %5s %8s %8s %5s %5s\n" "$v" "$ep" "$bc" "$lc" "$be" "$rem"
done

echo ""
echo "已完成: $(find results/dct_brca_recovery -name epoch_curve_fold0.csv | wc -l) 变体"
echo ""
echo "Ctrl+C 退出 | 30s 刷新"
'