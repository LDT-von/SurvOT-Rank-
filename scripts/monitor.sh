#!/bin/bash
# ============================================================
# SurvOT-Rank 三路训练监控 (v3.3 / v3.6 / v3.7)
# 用法: watch -n 30 bash scripts/monitor.sh
# ============================================================

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LOGDIR="$ROOT/logs"; RESDIR="$ROOT/results"

BOLD='\033[1m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'
CYAN='\033[0;36m'; DIM='\033[2m'; NC='\033[0m'
hr() { printf "%${1:-80}s\n" | tr ' ' '═'; }

# ── parse log → epoch|best|latest ──
parse_log() {
    local lf="$1" ep="-" best="-" latest="-"
    [ ! -f "$lf" ] || [ ! -s "$lf" ] && { echo "$ep|$best|$latest"; return; }
    ep=$(grep -oP '\[Fold \d+\] Epoch \d+/\d+' "$lf" 2>/dev/null | tail -1)
    if [ -n "$ep" ]; then
        local ec=$(echo "$ep" | grep -oP '\d+(?=/\d+)' | head -1)
        local et=$(echo "$ep" | grep -oP '(?<=/)\d+' | head -1)
        ep="${ec}/${et}"
    fi
    best=$(grep -oP 'best cindex:?\s*\K[0-9.]+' "$lf" 2>/dev/null | tail -1 || echo "-")
    latest=$(grep "val cindex" "$lf" 2>/dev/null | tail -1 | grep -oP '[\d.]+$' || echo "-")
    echo "$ep|$best|$latest"
}

# ── v3.3 state ──
v33_state() {
    local lf="$1"
    [ ! -f "$lf" ] || [ ! -s "$lf" ] && { echo "WAIT"; return; }
    head -1 "$lf" 2>/dev/null | grep -q "^usage:" && { echo "FAIL"; return; }
    local mt=$(stat -c %Y "$lf" 2>/dev/null || echo 0)
    local age=$(( $(date +%s) - mt ))
    local ep=$(grep -oP '\[Fold \d+\] Epoch \d+/\d+' "$lf" 2>/dev/null | tail -1)
    [ "$age" -lt 120 ] && [ -n "$ep" ] && { echo "RUN"; return; }
    grep -q "train_cindex\|best cindex" "$lf" 2>/dev/null && { echo "DONE"; return; }
    echo "ERR"
}

# ═══════ R1: v3.3 ═══════
route_v33() {
    echo ""; hr 60
    printf "  ${BOLD}%-48s${NC}  ${DIM}6 cancers × 5 folds = 30 jobs${NC}\n" "R1: v3.3 Score-First (UNI v1)"
    hr 60
    local cancers=(coadread hnsc kirc skcm stad ucec)
    local done=0 fail=0 run=0 wait=0
    for cancer in "${cancers[@]}"; do
        printf "  ${CYAN}%-10s${NC}" "$cancer"
        local _f
        for _f in 0 1 2 3 4; do
            local lf="$LOGDIR/v33_${cancer}_fold${_f}.log"
            local st=$(v33_state "$lf")
            local info=$(parse_log "$lf")
            local ep=$(echo "$info" | cut -d'|' -f1)
            local best=$(echo "$info" | cut -d'|' -f2)
            case "$st" in
                RUN)  printf " ${GREEN}F${_f}:RUN${NC}"; ((run++));;
                DONE) printf " ${GREEN}F${_f}:DONE${NC} ${DIM}E%-5s${NC}" "$ep"
                      [ "$best" != "-" ] && printf " ${YELLOW}b=%-6s${NC}" "$best"; ((done++));;
                FAIL) printf " ${RED}F${_f}:FAIL${NC}"; ((fail++));;
                ERR)  printf " ${RED}F${_f}:ERR${NC}"; ((fail++));;
                *)    printf " ${DIM}F${_f}:·${NC}"; ((wait++));;
            esac
        done
        echo ""
    done
    printf "  ${DIM}Done=%d ${RED}Fail=%d${NC} ${GREEN}Run=%d${NC} ${DIM}Wait=%d${NC}\n" "$done" "$fail" "$run" "$wait"
}

# ═══════ R2: v3.6 ═══════
route_v36() {
    echo ""; hr 60
    printf "  ${BOLD}%-48s${NC}  ${DIM}4 cancers × 6 variants × 2 folds = 48 jobs${NC}\n" "R2: v3.6 Listwise Transport"
    hr 60
    local cancers=(blca brca luad lusc)
    local variants=(nll ipcw etar ipcw_etar gpl tcl)
    local folds=(0 2) c_total=0

    # Check running v3.6 jobs via ps keyword matching
    local v36_proc=$(ps -eo args --noheaders | grep "survot_rank" | grep -v grep \
        | grep -oP 'dct36_\w+_\w+' | sort -u)
    if [ -n "$v36_proc" ]; then
        while IFS= read -r line; do
            printf "  ${GREEN}▶ RUN${NC} %s\n" "$line"
        done <<< "$v36_proc"
    fi

    echo ""
    for cancer in "${cancers[@]}"; do
        printf "  ${CYAN}%-8s${NC}" "$cancer"
        for variant in "${variants[@]}"; do
            local v_done=0
            for fold in "${folds[@]}"; do
                [ -f "$RESDIR/dct_v3.6_listwise/${variant}/${cancer}/split_${fold}_results_final.pkl" ] && ((v_done++))
            done
            case "$v_done" in
                2) printf " ${GREEN}✓${NC}"; ((c_total++));;
                1) printf " ${YELLOW}½${NC}";;
                *) printf " ${DIM}·${NC}";;
            esac
        done
        echo ""
    done
    printf "  ${DIM}Completed:${NC} ${GREEN}%d${NC}/48\n" "$c_total"
    [ -f "$LOGDIR/v36_listwise.log" ] && {
        local errs=$(grep -c "FAILED\|FATAL\|ModuleNotFoundError" "$LOGDIR/v36_listwise.log" 2>/dev/null || true)
        [ -n "$errs" ] && [ "$errs" -gt 0 ] 2>/dev/null && printf "  ${RED}Errors: %d${NC}\n" "$errs"
    }
}

# ═══════ R3: v3.7 ═══════
route_v37() {
    echo ""; hr 60
    printf "  ${BOLD}%-48s${NC}  ${DIM}10 cancers × 5 folds = 50 jobs${NC}\n" "R3: v3.7 UNI2-h Highscore"
    hr 60
    local cancers=(blca brca coadread hnsc kirc luad lusc skcm stad ucec) c_total=0

    # Check running v3.7 jobs
    local v37_proc=$(ps -eo args --noheaders | grep "survot_rank" | grep -v grep \
        | grep "uni2-h" | grep -v "dct36_\|dct_v3.6" \
        | grep -oP 'highscore_\w+' | sort -u)
    if [ -n "$v37_proc" ]; then
        while IFS= read -r line; do
            local c=$(echo "$line" | sed 's/highscore_//')
            printf "  ${GREEN}▶ RUN${NC} %s\n" "$c"
        done <<< "$v37_proc"
    fi

    echo ""
    for cancer in "${cancers[@]}"; do
        printf "  ${CYAN}%-10s${NC}" "$cancer"
        local c_done=0
        local _f; for _f in 0 1 2 3 4; do
            if [ -f "$RESDIR/dct_v3.7_uni2h/highscore/${cancer}/split_${_f}_results_final.pkl" ]; then
                printf " ${GREEN}F${_f}${NC}"; ((c_done++)); ((c_total++))
            else
                printf " ${DIM}F${_f}${NC}"
            fi
        done
        printf " ${DIM}[%d/5]${NC}\n" "$c_done"
    done
    printf "  ${DIM}Completed:${NC} ${GREEN}%d${NC}/50\n" "$c_total"
    [ -f "$LOGDIR/v37_uni2h.log" ] && {
        local errs=$(grep -c "FAILED\|FATAL\|ModuleNotFoundError" "$LOGDIR/v37_uni2h.log" 2>/dev/null || true)
        [ -n "$errs" ] && [ "$errs" -gt 0 ] 2>/dev/null && printf "  ${RED}Errors: %d${NC}\n" "$errs"
    }
}

# ═══════ Live Epoch ═══════
live_epochs() {
    echo ""; hr 60
    printf "  ${BOLD}Live Epoch / Score${NC}\n"; hr 60
    local tnow=$(date +%s) found=0
    for lf in "$LOGDIR"/v33_*.log; do
        [ ! -f "$lf" ] && continue
        local age=$(( tnow - $(stat -c %Y "$lf" 2>/dev/null || echo 0) ))
        [ "$age" -gt 120 ] && continue
        local nm=$(basename "$lf" .log | sed 's/v33_//')
        local ep=$(grep -oP '\[Fold \d+\] Epoch \d+/\d+' "$lf" 2>/dev/null | tail -1)
        local val=$(grep "val cindex" "$lf" 2>/dev/null | tail -1 | grep -oP 'val cindex=[0-9.]+|ipcw=[0-9.]+|IBS=[0-9.]+|iauc=[0-9.]+' | tr '\n' ' ')
        printf "  ${GREEN}v3.3 %-24s${NC} %s\n" "$nm" "$ep"
        [ -n "$val" ] && printf "    val:   %s\n" "$val"
        found=1
    done
    [ -f "$LOGDIR/v36_listwise.log" ] && {
        local ep=$(grep -oP '\[Fold \d+\] Epoch \d+/\d+' "$LOGDIR/v36_listwise.log" 2>/dev/null | tail -1)
        if [ -n "$ep" ]; then
            local val=$(grep "val cindex" "$LOGDIR/v36_listwise.log" 2>/dev/null | tail -1 | grep -oP 'val cindex=[0-9.]+|ipcw=[0-9.]+|IBS=[0-9.]+|iauc=[0-9.]+' | tr '\n' ' ')
            printf "  ${YELLOW}v3.6 listwise${NC}     %s\n" "$ep"
            [ -n "$val" ] && printf "    val:   %s\n" "$val"
            found=1
        fi
    }
    [ -f "$LOGDIR/v37_uni2h.log" ] && {
        local ep=$(grep -oP '\[Fold \d+\] Epoch \d+/\d+' "$LOGDIR/v37_uni2h.log" 2>/dev/null | tail -1)
        if [ -n "$ep" ]; then
            local val=$(grep "val cindex" "$LOGDIR/v37_uni2h.log" 2>/dev/null | tail -1 | grep -oP 'val cindex=[0-9.]+|ipcw=[0-9.]+|IBS=[0-9.]+|iauc=[0-9.]+' | tr '\n' ' ')
            printf "  ${CYAN}v3.7 UNI2-h${NC}       %s\n" "$ep"
            [ -n "$val" ] && printf "    val:   %s\n" "$val"
            found=1
        fi
    }
    [ "$found" = "0" ] && printf "  ${DIM}(no active training detected)${NC}\n"
}

# ═══════ MAIN ═══════
main() {
    echo ""; hr 80
    printf "  ${BOLD}SurvOT-Rank Training Monitor${NC}  ${DIM}$(date '+%Y-%m-%d %H:%M:%S')${NC}\n"
    hr 80
    # GPU
    local gpu=$(nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>/dev/null)
    if [ -n "$gpu" ]; then
        IFS=',' read -r idx name mu mt util temp <<< "$gpu"
        local mu_s=$(echo "$mu" | xargs) mt_s=$(echo "$mt" | xargs)
        local util_s=$(echo "$util" | xargs) temp_s=$(echo "$temp" | xargs)
        printf "  GPU: ${BOLD}%s${NC}  ${CYAN}%s/%s MiB (%d%%)${NC}  util=${GREEN}%s%%${NC}  temp=%s°C\n" \
            "$(echo "$name" | xargs)" "$mu_s" "$mt_s" "$(( mu_s * 100 / mt_s ))" "$util_s" "$temp_s"
    fi
    # Proc counts (using wc -l for reliable counting)
    local v33_n=$(ps -eo args --noheaders | grep "survot_rank" | grep -v grep | grep -c "coadread" 2>/dev/null | tr -d "\n")
    local v36_n=$(ps -eo args --noheaders | grep "survot_rank" | grep -v grep | grep -c "dct36_" 2>/dev/null | tr -d "\n")
    local v37_n=$(ps -eo args --noheaders | grep "survot_rank" | grep -v grep | grep -c "uni2-h" 2>/dev/null | tr -d "\n")
    printf "  ${BOLD}Procs:${NC} v3.3=${GREEN}%s${NC} v3.6=${YELLOW}%s${NC} v3.7=${CYAN}%s${NC}\n" "${v33_n:-0}" "${v36_n:-0}" "${v37_n:-0}"

    live_epochs
    route_v33
    route_v36
    route_v37
    echo ""; hr 80
    printf "  ${DIM}刷新: watch -n 30 bash scripts/monitor.sh${NC}\n"
    hr 80; echo ""
}

main
