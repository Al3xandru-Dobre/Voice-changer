# Setup / verify the Windows audio test rig for Milestone 1 Stages 05-06
# (virtual-cable loopback + latency measurement), fresh out of the box.
#
#   powershell -ExecutionPolicy Bypass -File tools\setup_test_rig.ps1
#
# What it does:
#   1. checks the toolchain (cmake required; python required for the rig
#      scripts; git only warned about)
#   2. configures, builds and runs the tests (fluxvoice, fv_probe)
#   3. checks the VB-CABLE driver endpoints; installs the driver from a
#      cached zip in tools\_dl\ or downloads it (UAC prompt)
#   4. optionally checks/installs Voicemeeter Banana the same way, for the
#      second virtual-cable pair used by the Stage 06 two-cable method
#   5. configures the Voicemeeter VAIO pairs as plain cables
#      (tools\vm_route.py: verified writes, refuses elevated Voicemeeter) and
#      verifies them end-to-end (fv_probe tone round-trip; silence fails)
#
# Options:
#   -NoBuild          skip configure/build/ctest (a build must already exist)
#   -NoInstall        check and report only; never launch an installer
#   -SkipVoicemeeter  VB-CABLE only; skip the second cable pair
#   -VBCableUrl / -VoicemeeterUrl   override the installer download URLs
#
# Lessons baked in (Documents/Bug Report/):
#   BUG-001  install drivers BEFORE running measurements; device indices
#            shift when the device graph changes, so scripts select devices
#            by name (--input-name/--output-name), never by stale indices.
#   BUG-002  run Voicemeeter non-elevated, or Remote API writes are silently
#            dropped. This script never launches Voicemeeter elevated.
#
# VB-Audio asks for a reboot after driver installs; the script continues
# anyway and reports anything that needs the reboot. Downloads are cached in
# tools\_dl\ (git-ignored).

param(
    [switch]$NoBuild,
    [switch]$NoInstall,
    [switch]$SkipVoicemeeter,
    [string]$VBCableUrl = 'https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack45.zip',
    [string]$VoicemeeterUrl = 'https://download.vb-audio.com/Download_MT128/VoicemeeterSetup_v2130.zip'
)

$ErrorActionPreference = 'Stop'
try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12 } catch {}

$Repo = Split-Path -Parent $PSScriptRoot
$Dl = Join-Path $Repo 'tools\_dl'
$Failures = 0

function Step([string]$Name) { Write-Host "`n=== $Name ===" -ForegroundColor Cyan }
function Ok([string]$Msg)    { Write-Host "  [ OK ] $Msg" -ForegroundColor Green }
function Warn([string]$Msg)  { Write-Host "  [warn] $Msg" -ForegroundColor Yellow }
function Fail([string]$Msg)  { Write-Host "  [FAIL] $Msg" -ForegroundColor Red; $script:Failures++ }

function Test-Tool([string]$Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Get-Python {
    # Finds the interpreter, stashing the command array in $PythonCmd and the
    # version string in $PythonVersion. Returns $true when found.
    foreach ($exe in @('py', 'python', 'python3')) {
        if (-not (Test-Tool $exe)) { continue }
        $v = if ($exe -eq 'py') { & $exe -3 --version 2>$null } else { & $exe --version 2>$null }
        if ($v) {
            $script:PythonCmd = if ($exe -eq 'py') { @($exe, '-3') } else { @($exe) }
            $script:PythonVersion = "$v"
            return $true
        }
    }
    return $false
}

function Invoke-Python([string]$Script, [string[]]$ScriptArgs = @()) {
    # Child output goes to the console (Out-Host); only the exit code is
    # returned, so callers can show messages and still test $LASTEXITCODE.
    $exe = $script:PythonCmd[0]
    $rest = @($script:PythonCmd | Select-Object -Skip 1)
    & $exe @rest $Script @ScriptArgs | Out-Host
    return $LASTEXITCODE
}

function Get-Devices([string]$FvProbe) {
    # Endpoint snapshot; indices shift on driver installs, so callers must
    # match by name substring, never by remembered index (BUG-001).
    return (& $FvProbe --list-devices 2>&1 | Out-String)
}

function Confirm-Install([string]$What) {
    if ($NoInstall) { Warn "$What is missing; -NoInstall given, not installing"; return $false }
    $a = Read-Host "  $What is missing. Install now (UAC prompt)? [y/N]"
    return ($a -eq 'y' -or $a -eq 'Y')
}

function Get-CachedOrDownload([string]$Url, [string]$ZipName) {
    if (-not (Test-Path $Dl)) { New-Item -ItemType Directory -Path $Dl | Out-Null }
    $zip = Join-Path $Dl $ZipName
    if (Test-Path $zip) { Ok "using cached $ZipName"; return $zip }
    Write-Host "  downloading $Url ..."
    Invoke-WebRequest -Uri $Url -OutFile $zip -UseBasicParsing
    return $zip
}

function Invoke-SetupExe([string]$Dir, [string]$Pattern, [string]$Label) {
    $exe = Get-ChildItem -Path $Dir -Recurse -Filter '*.exe' -ErrorAction SilentlyContinue |
           Where-Object { $_.Name -match $Pattern } | Select-Object -First 1
    if (-not $exe) { Fail "no setup executable matching '$Pattern' found under $Dir"; return }
    Write-Host "  running $($exe.FullName) elevated (accept the UAC prompt) ..."
    Start-Process -FilePath $exe.FullName -Verb RunAs -Wait
    Warn "$Label installed; VB-Audio recommends a reboot before measuring"
}

function Test-WavLoud([string]$Path) {
    # True when the 16-bit mono PCM WAV fv_probe writes has a sample above
    # roughly -40 dBFS. fv_probe writes a fixed 44-byte header.
    if (-not (Test-Path $Path)) { return $false }
    $b = [System.IO.File]::ReadAllBytes($Path)
    $peak = 0
    for ($i = 44; $i + 1 -lt $b.Length; $i += 2) {
        $s = [Math]::Abs([BitConverter]::ToInt16($b, $i))
        if ($s -gt $peak) { $peak = $s }
    }
    return ($peak / 32768.0) -gt 0.01
}

# ---------------------------------------------------------------------------
# 1. Toolchain
# ---------------------------------------------------------------------------
Step 'Toolchain'
if (Test-Tool cmake) { Ok "cmake $((cmake --version)[0])" }
else { Fail 'cmake not found (winget install Kitware.CMake; or install Visual Studio)'; exit 1 }

$Python = Get-Python
if ($Python) { Ok "python $($script:PythonVersion)" }
else { Warn 'python not found (winget install Python.Python.3.12); needed for vm_route.py / analyzers' }

if (Test-Tool git) { Ok 'git present' } else { Warn 'git not found' }

# ---------------------------------------------------------------------------
# 2. Build + tests
# ---------------------------------------------------------------------------
$FvProbe = $null
if (-not $NoBuild) {
    Step 'Build + tests'
    if (-not (Test-Path (Join-Path $Repo 'build\CMakeCache.txt'))) {
        cmake -S $Repo -B (Join-Path $Repo 'build')
        if ($LASTEXITCODE -ne 0) { Fail 'cmake configure failed (C++ toolchain installed?)'; exit 1 }
    }
    cmake --build (Join-Path $Repo 'build') --config Release
    if ($LASTEXITCODE -ne 0) { Fail 'build failed'; exit 1 }
    ctest --test-dir (Join-Path $Repo 'build') -C Release --output-on-failure
    if ($LASTEXITCODE -ne 0) { Fail 'ctest reported failures' } else { Ok 'ctest passed' }
}
foreach ($p in @('build\Release\fv_probe.exe', 'build\fv_probe.exe')) {
    $cand = Join-Path $Repo $p
    if (Test-Path $cand) { $FvProbe = $cand; break }
}
if ($FvProbe) { Ok "fv_probe: $FvProbe" }
else { Fail 'fv_probe.exe not found under build\ (run without -NoBuild once)'; exit 1 }

# ---------------------------------------------------------------------------
# 3. VB-CABLE (cable 1: the kernel-level loopback pair)
# ---------------------------------------------------------------------------
Step 'VB-CABLE driver'
$devices = Get-Devices $FvProbe
if ($devices -match 'CABLE Input' -and $devices -match 'CABLE Output') {
    Ok 'VB-CABLE endpoints present (CABLE Input / CABLE Output)'
} else {
    Warn 'VB-CABLE endpoints not found'
    if (Confirm-Install 'VB-CABLE') {
        $zip = Get-CachedOrDownload $VBCableUrl 'VBCABLE_Driver_Pack45.zip'
        $dst = Join-Path $Dl 'vbcable'
        if (-not (Test-Path $dst)) { Expand-Archive -Path $zip -DestinationPath $dst -Force }
        Invoke-SetupExe $dst 'VBCABLE_Setup' 'VB-CABLE'
        $devices = Get-Devices $FvProbe
        if ($devices -match 'CABLE Input') { Ok 'VB-CABLE endpoints present after install' }
        else { Fail 'VB-CABLE endpoints still missing (reboot?)' }
    }
}

# ---------------------------------------------------------------------------
# 4. Voicemeeter Banana (cable 2: the VAIO pairs; needed for Stage 06)
# ---------------------------------------------------------------------------
if (-not $SkipVoicemeeter) {
    Step 'Voicemeeter Banana'
    $vmDll = $null
    foreach ($root in @(${env:ProgramFiles(x86)}, $env:ProgramFiles)) {
        if (-not $root) { continue }
        $cand = Join-Path $root 'VB\Voicemeeter\VoicemeeterRemote64.dll'
        if (Test-Path $cand) { $vmDll = $cand; break }
    }
    if ($vmDll) { Ok "Voicemeeter installed ($vmDll)" }
    else {
        Warn 'Voicemeeter not found'
        if (Confirm-Install 'Voicemeeter Banana') {
            $zip = Get-CachedOrDownload $VoicemeeterUrl 'VoicemeeterSetup.zip'
            $dst = Join-Path $Dl 'voicemeeter'
            if (-not (Test-Path $dst)) { Expand-Archive -Path $zip -DestinationPath $dst -Force }
            Invoke-SetupExe $dst 'Setup' 'Voicemeeter'
            foreach ($root in @(${env:ProgramFiles(x86)}, $env:ProgramFiles)) {
                if (-not $root) { continue }
                $cand = Join-Path $root 'VB\Voicemeeter\VoicemeeterRemote64.dll'
                if (Test-Path $cand) { $vmDll = $cand; break }
            }
            if ($vmDll) { Ok 'Voicemeeter installed' } else { Fail 'Voicemeeter still missing (reboot?)' }
        }
    }
}

# ---------------------------------------------------------------------------
# 5. Configure + verify the Voicemeeter VAIO pairs
# ---------------------------------------------------------------------------
if (-not $SkipVoicemeeter) {
    Step 'VAIO pair routing + route check'
    if ($Python) {
        $vmRoute = Join-Path $Repo 'tools\vm_route.py'
        $rc = Invoke-Python $vmRoute
        if ($rc -ne 0) {
            Fail "vm_route.py failed (exit $rc; 2 = not installed/running or integrity mismatch, 1 = a write was dropped; see its message above)"
        } else {
            Ok 'VAIO pairs routed and verified: Voicemeeter Input -> Out B1, AUX Input -> Out B2'
            # Tone round-trip: fv_probe plays into the VAIO pair and records
            # the far end; silence means a dead route (BUG-001).
            $wav = Join-Path $env:TEMP 'vm_route_tone.wav'
            & $FvProbe tone --output-name 'Voicemeeter Input' --input-name 'Voicemeeter Out B1' --seconds 4 $wav | Out-Host
            if ($LASTEXITCODE -ne 0) { Fail 'fv_probe tone run failed' }
            elseif (-not (Test-WavLoud $wav)) {
                Fail 'Voicemeeter Out B1 stayed silent during the tone; routing verified correct, so see BUG-001 (wedge hypothesis / stale endpoints - re-run --list-devices)'
            } else { Ok 'route check passed: tone survives the round trip' }
        }
    } else {
        Warn 'python missing; configure manually: python tools\vm_route.py'
    }
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
Step 'Summary'
if ($Failures -eq 0) {
    Write-Host @'
Rig ready. Stage 05/06 commands (see Documents/Milestone 1/):

  build\Release\fluxvoice --list-devices        # pick mic / cable indices
  build\Release\fluxvoice --input <mic> --output <CABLE Input index>
  fv_probe record --input-name "CABLE Output" --seconds 20 voice.wav
  fv_probe latency --output-name "CABLE Input" --input-name "CABLE Output" --count 60

  two-cable latency method (Stage 06): cable 1 = VB-CABLE, cable 2 = the
  Voicemeeter VAIO pair configured above ("Voicemeeter Input" / "Voicemeeter
  Out B1"); fluxvoice itself still selects devices by index.

Reminders:
  - Re-run this script (or at least --list-devices) after installing or
    removing ANY audio driver: indices shift (BUG-001).
  - Keep Voicemeeter non-elevated or remote writes are silently dropped
    (BUG-002).
'@ -ForegroundColor Green
    exit 0
}
Write-Host "$Failures check(s) failed; see [FAIL] lines above." -ForegroundColor Red
exit 1
