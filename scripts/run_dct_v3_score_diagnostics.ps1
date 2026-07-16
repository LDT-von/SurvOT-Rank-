param(
    [ValidateSet("doctor", "smoke", "run", "summarize")]
    [string]$Mode = "run",
    [ValidateSet("full", "nll_only", "unweighted_rank", "legacy_six_loss", "all")]
    [string]$Variant = "all",
    [string]$Folds = "0,2,3",
    [string]$Gpu = "0",
    [string]$NumWorkers = "4",
    [string]$Python = "python",
    [string]$Config = "configs/diagnostics/dct_v3_score_blca.yaml"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
$ResultsRoot = "results/dct_v3_score_first_diagnostics"

function Get-Variants([string]$Selection) {
    if ($Selection -eq "all") { return @("full", "nll_only", "unweighted_rank", "legacy_six_loss") }
    return @($Selection)
}

function Get-VariantOverrides([string]$Name) {
    switch ($Name) {
        "full" { return @() }
        "nll_only" { return @("dct_lambda_ipcw_rank=0.0") }
        "unweighted_rank" { return @("dct_lambda_ipcw_rank=0.0", "dct_lambda_rank=0.05") }
        "legacy_six_loss" { return @(
            "dct_lambda_ipcw_rank=0.0",
            "dct_lambda_ot=0.06",
            "dct_lambda_rank=0.05",
            "dct_lambda_anchor=0.03",
            "dct_lambda_stage_risk=0.05",
            "dct_lambda_coordinate=0.01"
        ) }
        default { throw "Unknown variant: $Name" }
    }
}

function Invoke-Variant([string]$Name, [string]$FoldList, [switch]$Smoke) {
    foreach ($FoldText in $FoldList.Split(",")) {
        $Fold = [int]$FoldText.Trim()
        $EndFold = $Fold + 1
        $Args = @("-m", "survot_rank.cli", "train", "--config", $Config,
            "--set", "gpu=$Gpu", "--set", "num_workers=$NumWorkers",
            "--set", "results_dir=$ResultsRoot/$Name", "--set", "specific_simple=dct_v3_score_first_$Name")
        foreach ($Override in (Get-VariantOverrides $Name)) { $Args += @("--set", $Override) }
        if ($Smoke) { $Args += @("--set", "max_epochs=1") }
        $Args += @("--", "--k_start", "$Fold", "--k_end", "$EndFold")
        Write-Host "Running $Name, fold $Fold"
        & $Python @Args
        if ($LASTEXITCODE -ne 0) { throw "DCT diagnostic failed: $Name fold $Fold (exit $LASTEXITCODE)" }
    }
}

switch ($Mode) {
    "doctor" { & $Python -m survot_rank.cli doctor; exit $LASTEXITCODE }
    "smoke" { Invoke-Variant -Name "full" -FoldList (($Folds.Split(",")[0]).Trim()) -Smoke }
    "run" { foreach ($Name in (Get-Variants $Variant)) { Invoke-Variant -Name $Name -FoldList $Folds } }
    "summarize" { }
}

& $Python scripts/summarize_dct_v3_score_diagnostics.py --root $ResultsRoot --expected-folds $Folds
exit $LASTEXITCODE
