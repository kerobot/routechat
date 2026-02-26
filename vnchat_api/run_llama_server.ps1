param(
    [Parameter(Mandatory = $true)]
    [string]$ServerBinary,

    [Parameter(Mandatory = $true)]
    [string]$ModelPath,

    [int]$Port = 8080,
    [int]$ContextSize = 8192,
    [int]$GpuLayers = -1,
    [int]$Threads = 0,
    [string]$BindHost = "127.0.0.1"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ServerBinary)) {
    throw "llama-server バイナリが見つからない: $ServerBinary"
}

if (-not (Test-Path $ModelPath)) {
    throw "モデルファイルが見つからない: $ModelPath"
}

$arguments = @(
    "-m", $ModelPath,
    "--host", $BindHost,
    "--port", $Port,
    "-c", $ContextSize,
    "--n-gpu-layers", $GpuLayers
)

if ($Threads -gt 0) {
    $arguments += @("-t", $Threads)
}

Write-Host "[INFO] llama-server を起動します" -ForegroundColor Cyan
Write-Host "  binary : $ServerBinary"
Write-Host "  model  : $ModelPath"
Write-Host "  host   : $BindHost"
Write-Host "  port   : $Port"
Write-Host "  ctx    : $ContextSize"
Write-Host "  gpu    : $GpuLayers"

& $ServerBinary @arguments
