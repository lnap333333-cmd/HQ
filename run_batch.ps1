# 运行批量对比测试（改进后算法）
# 用法: .\run_batch.ps1  或  powershell -ExecutionPolicy Bypass -File run_batch.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
Write-Host "运行批量对比 (改进后 NSGA-II / MOEA/D / HHO)..." -ForegroundColor Cyan
python scripts/run_compare_batch.py 2>&1
if ($LASTEXITCODE -eq 0) {
    $latest = Get-ChildItem outputs -Filter "compare_batch_*" -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latest) {
        Write-Host "`n结果目录: $($latest.FullName)" -ForegroundColor Green
        Get-Content "$($latest.FullName)\batch_metrics.txt"
    }
} else {
    Write-Host "运行失败，请检查 Python 环境" -ForegroundColor Red
    exit 1
}
