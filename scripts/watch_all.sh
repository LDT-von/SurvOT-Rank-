#!/usr/bin/env bash
watch -n 30 -c '
echo "============================================"
echo " $(date +%H:%M:%S)  GPU: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
echo "============================================"
echo "=== v3.5R fold2 ==="
printf "  %-6s %5s %8s %5s %8s\n" cancer ep best_c @ep latest_c
for c in luad lusc skcm brca; do
  log="/home/ubuntu/SurvOT-Rank/logs/v35r_${c}_fold2.log"
  [ ! -f "$log" ] && continue
  latest=$(grep "val cindex" "$log" | tail -1)
  ep=$(echo "$latest" | grep -oP "Epoch \K[0-9]+" || echo "?")
  lc=$(echo "$latest" | grep -oP "cindex=\K[0-9.]+" || echo "?")
  best=$(grep "val cindex" "$log" | sort -t= -k2 -nr | head -1)
  bc=$(echo "$best" | grep -oP "cindex=\K[0-9.]+" || echo "?")
  be=$(echo "$best" | grep -oP "Epoch \K[0-9]+" || echo "?")
  printf "  %-6s %5s %8s %5s %8s\n" "$c" "$ep" "$bc" "$be" "$lc"
done
echo "  (blca done: 0.6589@6)"
echo ""
echo "=== BRCA Recovery fold2 ==="
printf "  %-16s %5s %8s %5s %8s\n" variant ep best_c @ep latest_c
for g in g0 g4 g8; do
  log="/home/ubuntu/SurvOT-Rank/logs/brca_recovery_fold2_${g}.log"
  [ ! -f "$log" ] && continue
  cur=$(grep -oP "variant \K\S+" "$log" | tail -1)
  latest=$(grep "val cindex" "$log" | tail -1)
  ep=$(echo "$latest" | grep -oP "Epoch \K[0-9]+" || echo "?")
  lc=$(echo "$latest" | grep -oP "cindex=\K[0-9.]+" || echo "?")
  ln=$(grep -n "variant ${cur}" "$log" | tail -1 | cut -d: -f1)
  best=$(tail -n +${ln} "$log" | grep "val cindex" | sort -t= -k2 -nr | head -1)
  bc=$(echo "$best" | grep -oP "cindex=\K[0-9.]+" || echo "?")
  be=$(echo "$best" | grep -oP "Epoch \K[0-9]+" || echo "?")
  printf "  %-16s %5s %8s %5s %8s\n" "$cur" "$ep" "$bc" "$be" "$lc"
done
echo ""
echo "ETAR: all done | 进程: $(ps aux | grep survot_rank | grep -v grep | wc -l)"
echo "Ctrl+C 退出 | 30s 刷新"
'
