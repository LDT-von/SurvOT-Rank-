param(
    [ValidateSet("doctor", "smoke", "verify", "full", "fix")]
    [string]$Mode = $(if ($env:MODE) { $env:MODE } else { "verify" }),
    [string]$Gpu = $(if ($env:GPU) { $env:GPU } else { "0" }),
    [string]$NumWorkers = $(if ($env:NUM_WORKERS) { $env:NUM_WORKERS } else { "4" }),
    [string]$Python = $(if ($env:PYTHON) { $env:PYTHON } else { "python" }),
    [string]$Config = $(if ($env:CONFIG) { $env:CONFIG } else { "configs/distributional_counterfactual_transport_blca.yaml" }),
    [string]$FixConfig = $(if ($env:FIX_CONFIG) { $env:FIX_CONFIG } else { "configs/fix/distributional_counterfactual_transport_fix_blca.yaml" })
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

function Invoke-DctTrain {
    param(
        [string]$RunConfig,
        [int]$KStart,
        [int]$KEnd,
        [string[]]$ExtraSet = @()
    )

    Write-Host "[DCT-v3] config=$RunConfig folds=$KStart..$($KEnd - 1) gpu=$Gpu"

    $args = @(
        "-m", "survot_rank.cli", "train",
        "--config", $RunConfig,
        "--set", "gpu=$Gpu",
        "--set", "num_workers=$NumWorkers"
    )

    foreach ($item in $ExtraSet) {
        $args += @("--set", $item)
    }

    $args += @("--", "--k_start", "$KStart", "--k_end", "$KEnd")
    & $Python @args
}

switch ($Mode) {
    "doctor" {
        & $Python -m survot_rank.cli doctor
    }
    "smoke" {
        Invoke-DctTrain -RunConfig $Config -KStart 0 -KEnd 1 -ExtraSet @("max_epochs=1")
    }
    "verify" {
        Invoke-DctTrain -RunConfig $Config -KStart 0 -KEnd 1
        Invoke-DctTrain -RunConfig $Config -KStart 2 -KEnd 3
    }
    "full" {
        Invoke-DctTrain -RunConfig $Config -KStart 0 -KEnd 5
    }
    "fix" {
        Invoke-DctTrain -RunConfig $FixConfig -KStart 2 -KEnd 3
    }
}

Write-Host "[DCT-v3] Done. Check results/distributional_counterfactual_transport_blca/ or the fix results directory."
