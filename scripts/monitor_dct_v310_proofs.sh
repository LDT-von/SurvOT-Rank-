#!/bin/bash
# DCT v3.10 Proof Experiments Monitor
# 用法: bash scripts/monitor_dct_v310_proofs.sh

REPO_ROOT="/data1/SurvOT-Rank"
RESULTS_DIR="$REPO_ROOT/results"
TRISURV_PYTHON="/home/ubuntu/.conda/envs/trisurv/bin/python"
PYTORCH_CUDA="cuda"

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}==============================================${NC}"
echo -e "${BLUE}  DCT v3.10 Proof Experiments Monitor${NC}"
echo -e "${BLUE}  $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo -e "${BLUE}==============================================${NC}"
echo ""

# 检查实验进程
check_process() {
    local name=$1
    local pid=$2
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $name running (PID: $pid)"
        return 1
    else
        echo -e "${YELLOW}○${NC} $name not running"
        return 0
    fi
}

# 实验 1: Mechanism Verification
echo -e "${BLUE}[Experiment 1] Mechanism Verification${NC}"
MECH_DIR="$RESULTS_DIR/dct_v382_mechanism_verification"
if [ -d "$MECH_DIR" ]; then
    echo "  Results directory exists: $MECH_DIR"
    echo "  Subdirectories:"
    ls -d "$MECH_DIR"/*/ 2>/dev/null | while read d; do
        echo "    - $(basename $d)"
    done
    # 检查是否有结果文件
    SENSITIVITY_DIR="$MECH_DIR/sensitivity"
    if [ -d "$SENSITIVITY_DIR" ]; then
        echo "  Sensitivity results:"
        ls "$SENSITIVITY_DIR" 2>/dev/null | head -10 | while read f; do
            echo "    - $f"
        done
    fi
else
    echo "  ${YELLOW}No results yet${NC}"
fi
echo ""

# 实验 2: Legacy Cross Cancer
echo -e "${BLUE}[Experiment 2] Legacy Cross Cancer (v3.10)${NC}"
LEGACY_DIR="$RESULTS_DIR/dct_v310_legacy/blca"
if [ -d "$LEGACY_DIR" ]; then
    echo "  Results directory exists: $LEGACY_DIR"
    echo "  Folds completed:"
    for fold in 0 1 2 3 4; do
        FOLD_DIR="$LEGACY_DIR/fold${fold}"
        if [ -d "$FOLD_DIR" ]; then
            if [ -f "$FOLD_DIR/model_best.pth" ]; then
                echo -e "    fold${fold}: ${GREEN}✓ completed${NC}"
            else
                echo -e "    fold${fold}: ${YELLOW}○ in progress${NC}"
            fi
        else
            echo -e "    fold${fold}: ${RED}✗ not started${NC}"
        fi
    done
else
    echo "  ${YELLOW}No results yet${NC}"
fi
echo ""

# 最新 JSON 结果文件
echo -e "${BLUE}[Latest Results]${NC}"
echo "  Mechanism verification proofs:"
ls -t "$RESULTS_DIR"/act_surv_v5/proofs/*.json 2>/dev/null | head -3 | while read f; do
    echo "    - $(basename $f) ($(stat -c %y "$f" 2>/dev/null | cut -d' ' -f1))"
done
echo "  DCT v3.10 proofs:"
ls -t "$RESULTS_DIR"/dct_v310/proofs/*.json 2>/dev/null | head -3 | while read f; do
    echo "    - $(basename $f) ($(stat -c %y "$f" 2>/dev/null | cut -d' ' -f1))"
done
echo ""

# 检查 GPU 使用
echo -e "${BLUE}[GPU Status]${NC}"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | while IFS=, read -r idx name util mem_used mem_total; do
    util=$(echo $util | tr -d ' %')
    if [ "$util" -gt 80 ]; then
        color=$RED
    elif [ "$util" -gt 50 ]; then
        color=$YELLOW
    else
        color=$GREEN
    fi
    echo -e "  GPU $idx: ${color}${name}${NC} | Util: ${color}${util}%${NC} | Mem: ${mem_used} / ${mem_total}"
done
echo ""

# 最近的日志文件
echo -e "${BLUE}[Recent Logs]${NC}"
find "$RESULTS_DIR" -name "*.log" -mmin -60 2>/dev/null | head -5 | while read f; do
    size=$(stat -c%s "$f" 2>/dev/null)
    echo "  - $(basename $f) ($(numfmt --to=iec $size 2>/dev/null || echo ${size}B))"
    tail -3 "$f" 2>/dev/null | sed 's/^/    /'
done
echo ""

# 建议的下一步
echo -e "${BLUE}[Suggested Next Steps]${NC}"
echo "  1. Run mechanism verification:"
echo "     $TRISURV_PYTHON scripts/run_dct_mechanism_verification.py run --gpu 0 --study blca --fold 0"
echo ""
echo "  2. Run v3.10 legacy (after mechanism verification):"
echo "     $TRISURV_PYTHON scripts/run_dct_v310_legacy_cross_cancer.py run --variant v310 --cancer blca --folds 0 1 2 3 4"
echo ""
