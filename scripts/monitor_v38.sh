#!/usr/bin/env bash
# v3.8 训练监控脚本
# 用法: bash scripts/monitor_v38.sh          # 单次快照
#       watch -n 30 bash scripts/monitor_v38.sh   # 每30秒刷新

LOG_FILE="/data1/SurvOT-Rank/logs/v38_remaining.log"
RESULT_DIR="/data1/SurvOT-Rank/results/dct_v3.8_transport_consistency/highscore/full"

echo "=============================================================="
echo "  DCT v3.8 Transport Consistency — 训练监控"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

# 1. 进程状态
echo ""
echo "--- 进程状态 ---"
PID=$(ps aux | grep 'run_dct_v38_transport_consistency' | grep -v grep | awk '{print $2}' | head -1)
if [ -n "$PID" ]; then
    RUNTIME=$(ps -p "$PID" -o etime --no-headers 2>/dev/null | xargs)
    echo "  PID: $PID  运行时间: $RUNTIME  状态: 🔄 运行中"
else
    echo "  ⚠️ 没有找到运行中的 v3.8 进程"
fi

# 2. GPU 使用
echo ""
echo "--- GPU 使用 ---"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null | while IFS=, read -r idx mem_used mem_total util temp; do
    printf "  GPU %s: %s / %s  利用率: %s  温度: %s\n" "$idx" "$mem_used" "$mem_total" "$util" "$temp"
done

# 3. 当前训练的癌种和折
echo ""
echo "--- 当前进度 ---"
CURRENT_FOLD=$(grep '\[Fold [0-9]\] start' "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP 'Fold \d+')

# 获取最新 epoch 和 val cindex（格式: [Epoch N] val cindex=X.XXXX）
LAST_EPOCH_LINE=$(grep '\[Epoch [0-9]*\] val cindex' "$LOG_FILE" 2>/dev/null | tail -1)
CURRENT_EPOCH=$(echo "$LAST_EPOCH_LINE" | grep -oP 'Epoch \d+' | head -1)
LAST_CINDEX=$(echo "$LAST_EPOCH_LINE" | grep -oP 'cindex=\d+\.\d+' | head -1)

# 最佳 C-Index 及 epoch
BEST_LINE=$(grep '\[Epoch [0-9]*\] val cindex' "$LOG_FILE" 2>/dev/null | awk -F'val cindex=' '{print $2}' | awk '{print $1}' | sort -rn | head -1)
BEST_EPOCH=$(grep "val cindex=${BEST_LINE}" "$LOG_FILE" 2>/dev/null | grep -oP 'Epoch \d+' | head -1)

# 从日志判断当前癌种
LAST_CANCER_LINE=$(grep 'study:' "$LOG_FILE" 2>/dev/null | tail -1)
if echo "$LAST_CANCER_LINE" | grep -q "brca"; then
    CURRENT_CANCER="BRCA"
elif echo "$LAST_CANCER_LINE" | grep -q "luad"; then
    CURRENT_CANCER="LUAD"
elif echo "$LAST_CANCER_LINE" | grep -q "lusc"; then
    CURRENT_CANCER="LUSC"
else
    CURRENT_CANCER="?"
fi

echo "  癌种: $CURRENT_CANCER | $CURRENT_FOLD | $CURRENT_EPOCH"
echo "  最新:  $LAST_EPOCH_LINE"
if [ -n "$BEST_LINE" ] && [ -n "$BEST_EPOCH" ]; then
    echo "  最佳:  val cindex=$BEST_LINE  (Epoch $BEST_EPOCH)"
fi

# 4. 各癌种各折完成情况
echo ""
echo "--- 各折完成情况 ---"
printf "  %-6s" "癌种"
for f in 0 1 2 3 4; do printf " Fold%-1s" "$f"; done
printf "  Mean     状态\n"

for cancer in blca brca luad lusc; do
    CANCER_UPPER=$(echo "$cancer" | tr '[:lower:]' '[:upper:]')
    printf "  %-6s" "$CANCER_UPPER"
    
    bests=()
    all_done=true
    any_done=false
    
    for f in 0 1 2 3 4; do
        # 查找 results_final.pkl
        found=$(find "$RESULT_DIR/$cancer" -name "split_${f}_results_final.pkl" 2>/dev/null | head -1)
        
        if [ -n "$found" ]; then
            # 已完成的折，读取 epoch_curve 获取最佳 C-Index
            epoch_curve_dir=$(dirname "$found")
            epoch_file="$epoch_curve_dir/epoch_curve_fold${f}.csv"
            if [ -f "$epoch_file" ]; then
                best=$(awk -F',' 'NR>1 {if($2>max){max=$2}} END{printf "%.4f", max}' "$epoch_file")
                bests+=("$best")
                printf " %s" "$best"
                any_done=true
            else
                printf "  ?    "
                all_done=false
            fi
        else
            # 未完成的折
            printf "  -    "
            all_done=false
        fi
    done
    
    # 计算均值
    if $all_done && [ ${#bests[@]} -eq 5 ]; then
        sum=0
        for v in "${bests[@]}"; do
            sum=$(echo "$sum + $v" | bc -l)
        done
        mean=$(echo "scale=4; $sum / 5" | bc -l)
        printf " %s  ✅" "$mean"
    elif $any_done; then
        printf "  -      🔄"
    else
        printf "  -      ⏳"
    fi
    echo ""
done

# 5. 最新日志尾部
echo ""
echo "--- 最新日志 (最后10行) ---"
tail -10 "$LOG_FILE" 2>/dev/null | grep -v '^\s*$' | sed 's/^/  /'

echo ""
echo "=============================================================="
echo "  监控命令: watch -n 30 bash scripts/monitor_v38.sh"
echo "  查看C-Index: grep 'val cindex' $LOG_FILE | tail -20"
echo "=============================================================="
