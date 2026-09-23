<#
.SYNOPSIS
Builds the training runtime bundle: a private Python with the trainer installed,
zipped, hashed, and described by a manifest the plug-in reads.

.DESCRIPTION
The Train component needs a Python with scikit-learn, skl2onnx and onnxruntime in
it, and a user of the plug-in installs none of that by hand. This script makes the
one file they get instead:

  1. downloads a python-build-standalone CPython 3.12 for Windows x64 (the
     install_only variant: a plain folder with python.exe at its root, no
     installer, nothing registered);
  2. pip-installs the trainer package from this checkout, with its dependencies,
     into that Python;
  3. writes bundle.json at the bundle root, which is how
     OtterLogic.MachineLearning.Training.TrainerRuntime knows a zip is a bundle
     and where its interpreter is;
  4. zips it to dist\otterlogic-trainer-<version>-win-x64.zip;
  5. writes dist\trainer-manifest.json - version, download URL, SHA-256, size -
     in the shape of TrainerRuntime.TrainerBundle.

The version is read from pyproject.toml, so the bundle and the package it holds
can never disagree. The manifest's URL assumes the zip will be attached to a
GitHub release of this repo tagged trainer-v<version>; the release is cut by
hand, and the manifest goes on it beside the zip, where
TrainerRuntime.ManifestUrl (releases/latest/download/trainer-manifest.json)
finds it.

Windows PowerShell 5.1 is enough: no &&, no ternary, tar.exe from Windows 10.

.PARAMETER PythonUrl
The python-build-standalone archive to build on. Must be a Windows x64
install_only .tar.gz; the default is a CPython 3.12 release, matching the version
the fixtures were made with. A local path to an already-downloaded archive is
accepted too, for a machine that cannot reach GitHub.

.PARAMETER PythonSha256
The archive's SHA-256, as published beside it (the .sha256 file on the release).
Checked when given; a bundle built on an archive nobody checked is a bundle
nobody should ship, so give it for a release build.

.PARAMETER OutputFolder
Where the zip and the manifest go. dist\ beside this script by default.

.PARAMETER ReleaseUrlBase
The releases/download root of the repo the zip will be attached to. The manifest
URL is <ReleaseUrlBase>/trainer-v<version>/<zip name>.

.PARAMETER KeepWork
Leave the unpacked bundle under the output folder after zipping, for a look.

.EXAMPLE
.\build-bundle.ps1 -PythonSha256 <hash from the release>
gh release create trainer-v0.2.0 dist\otterlogic-trainer-0.2.0-win-x64.zip dist\trainer-manifest.json
#>
[CmdletBinding()]
param(
    [string] $PythonUrl = "https://github.com/astral-sh/python-build-standalone/releases/download/20250818/cpython-3.12.11+20250818-x86_64-pc-windows-msvc-install_only.tar.gz",
    [string] $PythonSha256 = "",
    [string] $OutputFolder = "",
    [string] $ReleaseUrlBase = "https://github.com/Otter-Logic/MachineLearning/releases/download",
    [switch] $KeepWork
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

# Invoke-WebRequest's progress bar makes a 30 MB download take minutes on 5.1.
$ProgressPreference = "SilentlyContinue"

function Fail([string] $message) {
    throw "build-bundle: $message"
}

function Invoke-Native([string] $description, [string] $exe, [string[]] $arguments) {
    # Native exit codes do not throw on their own; every step that matters is
    # checked here so a failed pip install cannot become a zip that looks fine.
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) {
        Fail "$description failed with exit code $LASTEXITCODE."
    }
}

function Invoke-Captured([string] $exe, [string[]] $arguments) {
    # Both streams as one string, with the exit code left in $LASTEXITCODE for
    # the caller. Under ErrorActionPreference Stop, 5.1 turns a redirected
    # stderr line into a terminating error - so a harmless warning from Python
    # would abort the build - which is why the preference is relaxed here only.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        return (& $exe @arguments 2>&1 | Out-String)
    }
    finally {
        $ErrorActionPreference = $previous
    }
}

# -- where things are ---------------------------------------------------------

$trainerFolder = $PSScriptRoot
$pyproject = Join-Path $trainerFolder "pyproject.toml"
if (-not (Test-Path $pyproject)) {
    Fail "There is no pyproject.toml beside this script; run it from python\trainer."
}

if ([string]::IsNullOrWhiteSpace($OutputFolder)) {
    $OutputFolder = Join-Path $trainerFolder "dist"
}

# The version comes from the one place it is declared. The trainer prints the
# same string from __init__.py, and the build checks the two agree below.
$versionPattern = '^\s*version\s*=\s*"([^"]+)"'
$versionLine = Get-Content $pyproject | Where-Object { $_ -match $versionPattern } | Select-Object -First 1
if ($null -eq $versionLine -or -not ($versionLine -match $versionPattern)) {
    Fail "pyproject.toml has no version line."
}
$version = $Matches[1]

$zipName = "otterlogic-trainer-$version-win-x64.zip"
$zipPath = Join-Path $OutputFolder $zipName
$manifestPath = Join-Path $OutputFolder "trainer-manifest.json"
$workFolder = Join-Path $OutputFolder "work"
$bundleFolder = Join-Path $workFolder "bundle"

Write-Host "Building otterlogic-trainer $version into $OutputFolder"

New-Item -ItemType Directory -Force -Path $OutputFolder | Out-Null
if (Test-Path $workFolder) {
    Remove-Item -Recurse -Force $workFolder
}
New-Item -ItemType Directory -Force -Path $bundleFolder | Out-Null

# -- 1. the interpreter --------------------------------------------------------

$archiveName = [System.IO.Path]::GetFileName($PythonUrl)
if ($archiveName -notmatch 'x86_64-pc-windows-msvc-install_only') {
    Fail "'$archiveName' is not a Windows x64 install_only archive. The bundle needs one: a plain folder with python.exe at its root."
}
$archivePath = Join-Path $workFolder $archiveName

if (Test-Path $PythonUrl) {
    Write-Host "Using the archive at $PythonUrl"
    Copy-Item $PythonUrl $archivePath
}
else {
    Write-Host "Downloading $PythonUrl"
    # 5.1 defaults to TLS 1.0, which GitHub refuses.
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $PythonUrl -OutFile $archivePath -UseBasicParsing
}

if (-not [string]::IsNullOrWhiteSpace($PythonSha256)) {
    $actual = (Get-FileHash -Algorithm SHA256 $archivePath).Hash
    if ($actual -ne $PythonSha256.Trim()) {
        Fail "The Python archive's SHA-256 is $actual, not $PythonSha256. Nothing was built."
    }
    Write-Host "Archive hash checked."
}
else {
    Write-Warning "No -PythonSha256 given, so the archive was not checked. Give it for a release build."
}

# tar.exe (bsdtar) ships with Windows 10 1803 and later; Expand-Archive only
# reads zips, and install_only is a .tar.gz.
$tar = Get-Command tar.exe -ErrorAction SilentlyContinue
if ($null -eq $tar) {
    Fail "tar.exe was not found. It ships with Windows 10 1803 and later; on an older machine unpack the archive by hand and pass the folder's parent as -PythonUrl."
}

Write-Host "Unpacking the interpreter."
Invoke-Native "Unpacking $archiveName" $tar.Source @("-xzf", $archivePath, "-C", $bundleFolder)

# install_only unpacks to a folder called python with python.exe at its root,
# so the bundle keeps that folder and bundle.json points into it.
$pythonRelative = "python/python.exe"
$python = Join-Path $bundleFolder "python\python.exe"
if (-not (Test-Path $python)) {
    Fail "The archive unpacked, but there is no python\python.exe in it. Is it really an install_only build?"
}

# -- 2. the trainer and its wheels ---------------------------------------------

Write-Host "Installing the trainer and its dependencies."
$null = Invoke-Captured $python @("-m", "pip", "--version")
if ($LASTEXITCODE -ne 0) {
    Invoke-Native "ensurepip" $python @("-m", "ensurepip", "--upgrade")
}

# Not editable, and not from a wheel already built somewhere: pip builds the
# package from this checkout, so what ships is what is here.
Invoke-Native "pip install" $python @(
    "-m", "pip", "install",
    "--no-cache-dir",
    "--no-warn-script-location",
    "--disable-pip-version-check",
    $trainerFolder
)

# The interpreter starts and the package imports, or there is no bundle. The
# same command the plug-in could run after an install to prove one works.
$reported = (Invoke-Captured $python @("-m", "otterlogic_trainer", "--version")).Trim()
if ($LASTEXITCODE -ne 0 -or $reported -ne "otterlogic-trainer $version") {
    Fail "The installed trainer answers '$reported' to --version; expected 'otterlogic-trainer $version'. Does __init__.py agree with pyproject.toml?"
}
Write-Host "Installed: $reported"

# -- 3. trim ---------------------------------------------------------------------

# What the trainer will never use and a user will never see. Roughly a third of
# the archive: the interpreter's own test suite, IDLE, turtle, and Tk, none of
# which scikit-learn or onnxruntime touch. pip's caches and every __pycache__ go
# too; Python rebuilds them in the install folder on first run, which costs a
# few seconds once and halves the zip.
$trim = @(
    "python\Lib\test",
    "python\Lib\idlelib",
    "python\Lib\turtledemo",
    "python\Lib\tkinter",
    "python\tcl",
    "python\Lib\site-packages\pip\_vendor\certifi\__pycache__"
)
foreach ($relative in $trim) {
    $path = Join-Path $bundleFolder $relative
    if (Test-Path $path) {
        Remove-Item -Recurse -Force $path
    }
}
Get-ChildItem -Path $bundleFolder -Recurse -Directory -Filter "__pycache__" -Force |
    ForEach-Object { Remove-Item -Recurse -Force $_.FullName }

# -- 4. bundle.json ----------------------------------------------------------------

# Exactly what TrainerRuntime reads: the version names the install folder, and
# python says where the interpreter is relative to the bundle root. UTF-8
# without a BOM, because System.Text.Json does not skip one.
$descriptor = "{`"version`": `"$version`", `"python`": `"$pythonRelative`"}"
$utf8 = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText((Join-Path $bundleFolder "bundle.json"), $descriptor, $utf8)

# -- 5. zip ------------------------------------------------------------------------

# The bundle's own Python does the zipping. Compress-Archive on 5.1 and
# ZipFile.CreateFromDirectory on .NET Framework both write entry names with
# backslashes, which is not a conformant zip and which TrainerRuntime's
# path check would then have to special-case. zipfile writes forward slashes,
# sorted, and skips nothing by accident.
if (Test-Path $zipPath) {
    Remove-Item -Force $zipPath
}

$zipScript = Join-Path $workFolder "zip-bundle.py"
@'
import os
import sys
import zipfile

root, out = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for name in sorted(files):
            full = os.path.join(folder, name)
            archive.write(full, os.path.relpath(full, root).replace(os.sep, "/"))
'@ | Set-Content -Path $zipScript -Encoding Ascii

Write-Host "Zipping to $zipPath"
Invoke-Native "Zipping the bundle" $python @($zipScript, $bundleFolder, $zipPath)

# -- 6. manifest -----------------------------------------------------------------

$zipInfo = Get-Item $zipPath
$sha256 = (Get-FileHash -Algorithm SHA256 $zipPath).Hash.ToLowerInvariant()
$url = "$($ReleaseUrlBase.TrimEnd('/'))/trainer-v$version/$zipName"

# The same four keys as TrainerBundle, camelCase. Written by hand rather than
# ConvertTo-Json so the shape is visible here and 5.1's formatting cannot
# change it.
$manifest = @"
{
  "version": "$version",
  "url": "$url",
  "sha256": "$sha256",
  "bytes": $($zipInfo.Length)
}
"@
[System.IO.File]::WriteAllText($manifestPath, $manifest, $utf8)

if (-not $KeepWork) {
    Remove-Item -Recurse -Force $workFolder
}

$megabytes = [math]::Round($zipInfo.Length / 1MB, 1)
Write-Host ""
Write-Host "Wrote $zipPath ($megabytes MB)"
Write-Host "      $manifestPath"
Write-Host "SHA-256 $sha256"
Write-Host ""
Write-Host "To release it, so that Train's 'Install training runtime' finds it:"
Write-Host "  gh release create trainer-v$version `"$zipPath`" `"$manifestPath`" --title `"Training runtime $version`""
Write-Host "The manifest must be on the newest release of the repo: TrainerRuntime reads releases/latest/download/trainer-manifest.json."
