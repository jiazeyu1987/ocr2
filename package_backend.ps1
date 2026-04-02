param(
    [string]$CondaHookPath = "D:\miniconda3\shell\condabin\conda-hook.ps1",
    [string]$BaseEnvName = "base",
    [string]$BuildEnvName = "houyang",
    [string]$SpecFileName = "ocrapp_pureray.spec",
    [string]$ZipOutputPath = "dist\ocrapp_pureray.zip",
    [switch]$SkipBuildExt,
    [switch]$SkipPyInstaller,
    [switch]$SkipZip
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Assert-PathExists {
    param(
        [string]$Path,
        [string]$Label
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Label not found: $Path"
    }
}

function Invoke-Checked {
    param(
        [string]$Label,
        [scriptblock]$Action
    )

    Write-Step $Label
    $global:LASTEXITCODE = 0
    & $Action
    $commandSucceeded = $?
    $exitCode = $global:LASTEXITCODE
    if (-not $commandSucceeded -or $exitCode -ne 0) {
        throw "$Label failed with exit code $exitCode"
    }
}

function Assert-NoRunningPackageProcess {
    $runningProcesses = Get-Process -Name "ocrapp_pureray" -ErrorAction SilentlyContinue
    if ($runningProcesses) {
        $processSummary = ($runningProcesses | ForEach-Object { "$($_.Id):$($_.ProcessName)" }) -join ", "
        throw "Packaging cannot continue while ocrapp_pureray is running. Stop the process first. Running: $processSummary"
    }
}

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$specPath = Join-Path $repoRoot $SpecFileName
$distDir = Join-Path $repoRoot "dist\ocrapp_pureray"
$internalDir = Join-Path $distDir "_internal"
$zipPath = if ([System.IO.Path]::IsPathRooted($ZipOutputPath)) {
    $ZipOutputPath
} else {
    Join-Path $repoRoot $ZipOutputPath
}

Write-Host "Backend package root: $repoRoot" -ForegroundColor Green
Set-Location $repoRoot

Assert-PathExists -Path $CondaHookPath -Label "Conda hook"
Assert-PathExists -Path $specPath -Label "PyInstaller spec"
Assert-PathExists -Path (Join-Path $repoRoot "setup.py") -Label "setup.py"
Assert-NoRunningPackageProcess

Write-Step "Load conda hook"
. $CondaHookPath

Invoke-Checked -Label "Activate conda env '$BaseEnvName'" -Action { conda activate $BaseEnvName }
Invoke-Checked -Label "Activate conda env '$BuildEnvName'" -Action { conda activate $BuildEnvName }

if (-not $SkipBuildExt) {
    Invoke-Checked -Label "Build Cython extensions" -Action { python setup.py build_ext --inplace }
} else {
    Write-Step "Skip Cython extension build"
}

if (-not $SkipPyInstaller) {
    Invoke-Checked -Label "Build PyInstaller package" -Action { pyinstaller --noconfirm --clean $specPath }
} else {
    Write-Step "Skip PyInstaller package build"
}

Assert-PathExists -Path $distDir -Label "Dist directory"
Assert-PathExists -Path $internalDir -Label "Internal dist directory"

$excludedTopLevelDirs = @("dist", "build", ".git", "__pycache__")
$sourceFiles = Get-ChildItem -Path $repoRoot -Recurse -File | Where-Object {
    if ($_.Extension -notin @(".py", ".pyd")) {
        return $false
    }
    $relativePath = $_.FullName.Substring($repoRoot.Length).TrimStart("\", "/")
    if ([string]::IsNullOrWhiteSpace($relativePath)) {
        return $false
    }
    $topLevelSegment = ([regex]::Split($relativePath, "[\\/]", 2))[0]
    return $topLevelSegment -notin $excludedTopLevelDirs
}

if (-not $sourceFiles) {
    throw "No .py or .pyd source files found under $repoRoot"
}

Write-Step "Copy .py and .pyd files to dist internal"
foreach ($sourceFile in $sourceFiles) {
    $relativePath = $sourceFile.FullName.Substring($repoRoot.Length).TrimStart("\", "/")
    $destination = Join-Path $internalDir $relativePath
    $destinationParent = Split-Path -Parent $destination
    if (-not (Test-Path -LiteralPath $destinationParent)) {
        New-Item -ItemType Directory -Path $destinationParent -Force | Out-Null
    }
    Copy-Item -LiteralPath $sourceFile.FullName -Destination $destination -Force
    Write-Host "Copied $relativePath" -ForegroundColor DarkGray
}

if (-not $SkipZip) {
    Write-Step "Create zip package"
    $zipParent = Split-Path -Parent $zipPath
    if (-not (Test-Path -LiteralPath $zipParent)) {
        New-Item -ItemType Directory -Path $zipParent | Out-Null
    }
    if (Test-Path -LiteralPath $zipPath) {
        Remove-Item -LiteralPath $zipPath -Force
    }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::CreateFromDirectory(
        $distDir,
        $zipPath,
        [System.IO.Compression.CompressionLevel]::Optimal,
        $true
    )
    Write-Host "Zip created: $zipPath" -ForegroundColor Green
} else {
    Write-Step "Skip zip package creation"
}

Write-Host ""
Write-Host "Backend package completed successfully." -ForegroundColor Green
Write-Host "Dist directory: $distDir" -ForegroundColor Green
if (-not $SkipZip) {
    Write-Host "Zip file: $zipPath" -ForegroundColor Green
}
