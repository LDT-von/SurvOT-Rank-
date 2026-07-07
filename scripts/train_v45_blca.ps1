param(
    [string]$Config = "configs/v45_blca.yaml",
    [string]$Gpu = "0",
    [string]$Seed = "3"
)

$ErrorActionPreference = "Stop"
python -m survot_rank.cli train --config $Config --set gpu=$Gpu --set seed=$Seed

