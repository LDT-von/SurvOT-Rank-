# P0 实验一键运行脚本 (Windows PowerShell 版本)
# 只跑 fold 0 和 fold 2，各 30 epoch
# 用法: .\scripts\run_p0_experiments.ps1

param(
    [string]$ProjectRoot = "E:\SurvOT-Rank",
    [string]$CondaEnv = "trisurv",
    [int]$GPU = 0
)

$ErrorActionPreference = "Stop"

function Invoke-Experiment {
    param(
        [string]$ConfigPath
    )
    
    $expName = [System.IO.Path]::GetFileNameWithoutExtension($ConfigPath)
    Write-Host "=============================================="
    Write-Host "[P0] Running: $expName"
    Write-Host "=============================================="
    
    Push-Location $ProjectRoot
    
    # 激活 conda 环境
    conda activate $CondaEnv
    
    # 只跑 fold 0
    python -m survot_rank.cli train `
        --config $ConfigPath `
        --set "gpu=$GPU" `
        --set "num_workers=0" `
        -- --k_start 0 --k_end 1
    
    # 只跑 fold 2
    python -m survot_rank.cli train `
        --config $ConfigPath `
        --set "gpu=$GPU" `
        --set "num_workers=0" `
        -- --k_start 2 --k_end 3
    
    Pop-Location
    Write-Host "[P0] Completed: $expName"
}

# P0-1: v45 全 8 损失 + 分箱 B 对照
Invoke-Experiment -ConfigPath "configs\p0_experiments\v45_baseline_globalbin_blca.yaml"

# P0-2: v50_norank 固定 seed 复核
Invoke-Experiment -ConfigPath "configs\p0_experiments\v50_norank_seed3_blca.yaml"
Invoke-Experiment -ConfigPath "configs\p0_experiments\v50_norank_seed5_blca.yaml"

# P0-3: v50 损失消融
Invoke-Experiment -ConfigPath "configs\p0_experiments\v50_ablation_only_ot_eventsurv_blca.yaml"
Invoke-Experiment -ConfigPath "configs\p0_experiments\v50_ablation_spec_cover_blca.yaml"

Write-Host "=============================================="
Write-Host "[P0] All experiments completed!"
Write-Host "=============================================="
