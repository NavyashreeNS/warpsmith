<#
.SYNOPSIS
  Configure, build, test and benchmark warpsmith on Windows.

.DESCRIPTION
  nvcc needs the MSVC host compiler on its PATH, and CMake needs to be able to
  find it. This script locates the newest Visual Studio installation with
  vswhere, imports its environment, then drives CMake. It exists because
  "just run cmake" fails on a stock Windows box for reasons that have nothing to
  do with this project.

.EXAMPLE
  .\scripts\build.ps1
  .\scripts\build.ps1 -Bench -Suite sgemm
#>
[CmdletBinding()]
param(
  [string]$BuildDir = "build",
  [ValidateSet("Release", "RelWithDebInfo", "Debug")]
  [string]$Config = "Release",
  [string]$Architectures = "",   # e.g. "86" or "80;86"; empty means auto-detect.
  [switch]$Test,
  [switch]$Bench,
  [string]$Suite = "all",
  [switch]$Clean
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

function Import-VisualStudioEnvironment {
  $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
  if (-not (Test-Path $vswhere)) {
    throw "vswhere.exe not found. Install Visual Studio with the 'Desktop development with C++' workload."
  }
  $install = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
  if (-not $install) { throw "No Visual Studio installation with the C++ toolset was found." }

  $vcvars = Join-Path $install "VC\Auxiliary\Build\vcvars64.bat"
  if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found under $install" }

  Write-Host "Importing MSVC environment from $install" -ForegroundColor DarkGray
  # Run vcvars in cmd, dump the resulting environment, and copy it into this session.
  & cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
      Set-Item -Path "Env:$($matches[1])" -Value $matches[2] -ErrorAction SilentlyContinue
    }
  }
}

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
  Import-VisualStudioEnvironment
}

if (-not (Get-Command nvcc.exe -ErrorAction SilentlyContinue)) {
  throw "nvcc not found on PATH. Install the CUDA Toolkit (12.0 or newer)."
}

if ($Clean -and (Test-Path $BuildDir)) {
  Write-Host "Removing $BuildDir" -ForegroundColor DarkGray
  Remove-Item -Recurse -Force $BuildDir
}

$cmakeArgs = @("-S", ".", "-B", $BuildDir, "-DCMAKE_BUILD_TYPE=$Config")
if ($Architectures) { $cmakeArgs += "-DCMAKE_CUDA_ARCHITECTURES=$Architectures" }

Write-Host "==> cmake configure" -ForegroundColor Cyan
& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

Write-Host "==> cmake build" -ForegroundColor Cyan
& cmake --build $BuildDir --config $Config --parallel
if ($LASTEXITCODE -ne 0) { throw "build failed" }

# Multi-config generators nest binaries one level deeper.
$exeDir = if (Test-Path (Join-Path $BuildDir $Config)) { Join-Path $BuildDir $Config } else { $BuildDir }
Write-Host "Binaries in $exeDir" -ForegroundColor Green

if ($Test) {
  Write-Host "==> ctest" -ForegroundColor Cyan
  & ctest --test-dir $BuildDir -C $Config --output-on-failure
  if ($LASTEXITCODE -ne 0) { throw "tests failed" }
}

if ($Bench) {
  New-Item -ItemType Directory -Force -Path "results" | Out-Null
  Write-Host "==> benchmark ($Suite)" -ForegroundColor Cyan
  & (Join-Path $exeDir "warpsmith_bench.exe") --suite $Suite --json "results/results.json"
  if ($LASTEXITCODE -ne 0) { throw "benchmark reported correctness failures" }
  Write-Host "==> regenerating report and charts" -ForegroundColor Cyan
  & python tools/report.py results/results.json --out results/REPORT.md
  & python tools/plot.py results/results.json --outdir docs/charts
}
