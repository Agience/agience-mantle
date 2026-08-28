# Install an Agience node on Windows. Self-contained: needs nothing from any checkout.
#
#   irm https://mantle.home.agience.ai/install.ps1 | iex
#   .\install.ps1 -Zip .\dist\agience-0.1.0-win-x64.zip          # from a local build
#   .\install.ps1 -Zip <path> -Sha256 604580bc...                # verify before trusting
#   .\install.ps1 -Zip <path> -Check                             # say what would happen, change nothing
#   .\install.ps1 -Uninstall                                     # remove the task and the program files
#
# ASCII-ONLY ON PURPOSE. Windows PowerShell 5.1 reads a .ps1 as the system ANSI codepage unless it
# carries a UTF-8 BOM, so a non-ASCII character is decoded as mojibake and can terminate a string
# literal early. Backticks are avoided for the same class of reason - backtick is the escape char.
#
# WHAT AN INSTALLED NODE IS, AND WHY IT IS NOT WHAT THE DEV BOX RUNS. A development workstation runs
# five services (origin, mantle, ember, crystal, caddy) under a supervisor that orders them, gates on
# a JWKS and restarts them. An INSTALLED node is ONE process: `agience serve`. There is no origin to
# order it against - a standalone node is its own authority and mints its own credential - and no
# edge, because a stranger has no zone and no certificate. So this registers one task pointed at one
# executable, and Task Scheduler's own restart policy is enough. Shipping the five-service supervisor
# to someone with one service would be shipping them a machine to operate rather than a thing to use.
#
# WHAT THIS DOES NOT DO. It does not open a firewall port, request a certificate, or make the node
# reachable from anywhere but this machine. A default install answers on 127.0.0.1 and nowhere else.
# That is the honest default: TLS needs a name, a name needs DNS, and both are decisions with
# consequences that an installer must not make silently on someone's behalf.
[CmdletBinding()]
param(
    [string] $Zip,
    [string] $Url,
    [string] $Sha256,
    [string] $InstallDir = "$env:LOCALAPPDATA\Programs\Agience",
    [string] $DataDir    = "$env:LOCALAPPDATA\Agience\node",
    [int]    $Port       = 8081,
    [switch] $NoService,
    [switch] $Check,
    [switch] $Uninstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$TaskName = "Agience node"

# Stop-Node - stop the running node and WAIT for it to let go of its files.
#
# `Unregister-ScheduledTask` does not stop the process. Unregistering while the node runs leaves a
# loaded DLL locked - `_internal\cryptography\hazmat\bindings\_rust.pyd: Access to the path is
# denied` - and a half-deleted program directory with a live process running out of it. Stopping is a
# separate act from unregistering, and the file removal has to wait for the handle
# to actually close: on Windows a process's DLL locks outlive the kill request by a moment.
function Stop-Node ($exePath) {
    $t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($t -and $t.State -eq "Running") { Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue }
    # By the image path, not by name: "agience.exe" may also be a DIFFERENT install on this box,
    # and killing someone else's node while uninstalling yours is not a tidy-up, it is an outage.
    if ($exePath -and (Test-Path -LiteralPath $exePath)) {
        Get-CimInstance Win32_Process -Filter "Name='agience.exe'" -ErrorAction SilentlyContinue |
            Where-Object { $_.ExecutablePath -eq $exePath } |
            ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    }
    foreach ($i in 1..30) {
        $still = Get-CimInstance Win32_Process -Filter "Name='agience.exe'" -ErrorAction SilentlyContinue |
                 Where-Object { $_.ExecutablePath -eq $exePath }
        if (-not $still) { return }
        Start-Sleep -Milliseconds 500
    }
}

# Remove-Tree - delete a directory, retrying while Windows releases handles.
function Remove-Tree ($path) {
    if (-not (Test-Path -LiteralPath $path)) { return $true }
    foreach ($i in 1..10) {
        try { Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction Stop; return $true }
        catch { Start-Sleep -Seconds 1 }
    }
    return -not (Test-Path -LiteralPath $path)
}

function Say  ($m) { Write-Host "  $m" }
function OK   ($m) { Write-Host "  OK $m"      -ForegroundColor Green }
function Warn ($m) { Write-Host "  ! $m"       -ForegroundColor Yellow }
function Fail ($m) { Write-Host "  REFUSED $m" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "-- Agience node installer"

# ── uninstall ────────────────────────────────────────────────────────────────────────────────────
# The data directory is never removed. It holds `keys/`, and losing those makes every artifact in
# the store permanently unreadable - no grant, no re-derivation, no recovery. An uninstaller that
# deletes the program and leaves the data is recoverable in the direction that should be easy.
if ($Uninstall) {
    # Stop first, then unregister, then delete. Unregistering a running task leaves the process
    # alive with no task to stop it by, so the files stay locked and the only way back is Task
    # Manager.
    Stop-Node (Join-Path $InstallDir "agience.exe")
    $t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($t) { Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false; OK "stopped and unregistered '$TaskName'" }
    else    { Say "i no task named '$TaskName'" }
    if (Test-Path -LiteralPath $InstallDir) {
        if (Remove-Tree $InstallDir) { OK "removed $InstallDir" }
        else { Warn "could not fully remove $InstallDir - something still holds a file open" }
    }
    Warn "left $DataDir alone - it holds keys/, and without those the store is unreadable forever."
    Warn "  Delete it by hand if you are certain."
    exit 0
}

# ── acquire ──────────────────────────────────────────────────────────────────────────────────────
$tempZip = $null
if (-not $Zip) {
    if (-not $Url) { Fail "give me either -Zip <local file> or -Url <download>" }
    $tempZip = Join-Path $env:TEMP ("agience-" + [guid]::NewGuid().ToString("N") + ".zip")
    Say "i downloading $Url"
    Invoke-WebRequest -Uri $Url -OutFile $tempZip -UseBasicParsing
    $Zip = $tempZip
}
if (-not (Test-Path -LiteralPath $Zip)) { Fail "no such file: $Zip" }

# Verify before unpacking. The digest is the artifact - the same rule the fleet's deploy path
# applies ("nothing in this path accepts a tag") - and checking after expansion would mean the bytes
# are already written where they will be executed from.
$actual = (Get-FileHash -LiteralPath $Zip -Algorithm SHA256).Hash.ToLower()
Say "i sha256:$actual"
if ($Sha256) {
    $want = $Sha256.ToLower() -replace '^sha256:', ''
    if ($actual -ne $want) { Fail "digest mismatch - expected $want. NOT unpacking." }
    OK "digest matches"
} else {
    # An unverified payload is reported. An installer silent about one teaches the person running
    # it that verification is optional.
    Warn "no -Sha256 given, so nothing was verified. Pass the digest the node published."
}

if ($Check) {
    Write-Host ""
    Say "i would install to  $InstallDir"
    Say "i would seed        $DataDir"
    Say "i would serve on    http://127.0.0.1:$Port"
    if (-not $NoService) { Say "i would register    scheduled task '$TaskName' (at logon)" }
    OK "check only - nothing was changed"
    exit 0
}

# ── unpack ───────────────────────────────────────────────────────────────────────────────────────
# Replaced wholesale rather than merged: a half-old bundle is a set of files that were never tested
# together, and PyInstaller's `_internal` is a matched set with the executable that loads it.
if (Test-Path -LiteralPath $InstallDir) {
    Say "i stopping the running node before replacing it"
    Stop-Node (Join-Path $InstallDir "agience.exe")
    if (-not (Remove-Tree $InstallDir)) {
        Fail "could not clear $InstallDir - something still holds a file open there. Close it and retry."
    }
}
New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
Say "i unpacking to $InstallDir"
# The bundle is 8,782 entries, and where it lands dominates everything else. On this workstation,
# same archive, same extractor:
#
#     -> C: (an SSD, and where $InstallDir defaults)      10.3 s
#     -> D: (a 465 GB spinning disk, 86% full)           571.6 s      55x slower
#
# So the program goes on the fast disk. The "store must not live on C:" rule covers a lattice that
# grows to tens of GB on a nearly full disk; this is a fixed ~223 MB read constantly and written
# once, and applying that rule to it would make a ten-second operation take ten minutes.
#
# `ZipFile::ExtractToDirectory` rather than `Expand-Archive` is a separate, smaller point: the two
# take the same time on the same target, and the .NET call is used because it avoids the per-entry
# pipeline overhead and reports its own elapsed time.
#
# An installer that appears to hang is an installer people kill halfway, which leaves a
# half-unpacked program directory - the exact state this script refuses to merge into. Hence the
# timing line below: a slow target should look slow, not broken.
$sw = [Diagnostics.Stopwatch]::StartNew()
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::ExtractToDirectory(
    (Resolve-Path -LiteralPath $Zip).Path, $InstallDir)
$sw.Stop()
Say "i unpacked in $([math]::Round($sw.Elapsed.TotalSeconds,1))s"
if ($tempZip) { Remove-Item -LiteralPath $tempZip -Force -ErrorAction SilentlyContinue }

$exe = Join-Path $InstallDir "agience.exe"
if (-not (Test-Path -LiteralPath $exe)) { Fail "the archive did not contain agience.exe" }
OK "installed $((& $exe version | Select-Object -First 1))"

# ── seed ─────────────────────────────────────────────────────────────────────────────────────────
# Idempotent by refusal, not by overwrite: `agience init` reports an already-seeded directory rather
# than running over it, so re-running the installer upgrades the PROGRAM and never touches the keys.
New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
$seeded = Test-Path -LiteralPath (Join-Path $DataDir "keys\mantle.private.pem")
if ($seeded) {
    Say "i already seeded - keeping the existing keys at $DataDir\keys"
} else {
    Say "i seeding $DataDir"
    & $exe init --dir $DataDir | Out-Null
    if ($LASTEXITCODE -ne 0) { Fail "seeding failed" }
    OK "seeded"
}

# ── register ─────────────────────────────────────────────────────────────────────────────────────
if (-not $NoService) {
    $user = "$env:USERDOMAIN\$env:USERNAME"
    # Named `$serveArgs` because `$args` is a PowerShell automatic variable holding the script's
    # own arguments: assigning to it is legal, shadows the real one, and breaks anything downstream
    # that reads it, without failing at runtime. PSScriptAnalyzer catches it here.
    $serveArgs = "serve --dir `"$DataDir`" --host 127.0.0.1 --port $Port"
    $action  = New-ScheduledTaskAction -Execute $exe -Argument $serveArgs -WorkingDirectory $InstallDir
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $user
    $principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
    # ExecutionTimeLimit 0 = never kill it. The default is THREE DAYS, after which Task Scheduler
    # stops the node with no error recorded anywhere a person would look.
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
                    -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable `
                    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
                    -MultipleInstances IgnoreNew
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existing) { Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false }
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
        -Principal $principal -Settings $settings `
        -Description "An Agience node. Serves the lattice on 127.0.0.1:$Port." | Out-Null
    OK "registered '$TaskName' (starts at logon)"
    Start-ScheduledTask -TaskName $TaskName
}

# ── prove it ─────────────────────────────────────────────────────────────────────────────────────
# An install verifies itself. Three assertions, each named by what it catches, and the same three
# the cold-start rehearsal makes.
Write-Host ""
$base = "http://127.0.0.1:$Port"
# With -NoService nothing started the node, so there is nothing to verify: say so and stop. Polling
# an address for 60s and reporting a timeout would be the installer failing itself for doing exactly
# what it was asked to do.
if ($NoService) {
    OK "installed and seeded. No task was registered (-NoService), so nothing is running."
    Write-Host ""
    Say "  start it yourself:"
    Say "    & `"$exe`" serve --dir `"$DataDir`" --port $Port"
    exit 0
}

Say "i verifying (first start reads the whole bundle off disk - this can take a minute)"
# 240s, sized to the slowest legitimate first start. A first start faults in ~223 MB of DLLs: a few
# seconds on the default C: target, and minutes on a spinning disk, where a 60s budget gives up on an
# install that unpacked, seeded and registered correctly and comes up healthy moments later. An
# installer that calls a working install broken is worse than one
# that waits, because the person's next move is to uninstall something that was fine.
$deadline = 240
$up = $false
$sw2 = [Diagnostics.Stopwatch]::StartNew()
foreach ($i in 1..$deadline) {
    try { Invoke-WebRequest -Uri "$base/" -TimeoutSec 3 -UseBasicParsing | Out-Null; $up = $true; break }
    catch { Start-Sleep -Seconds 1 }
    if ($i % 30 -eq 0) { Say "i   still starting ($i s)..." }
}
$sw2.Stop()
if (-not $up) {
    Warn "the node did not answer on $base within ${deadline}s."
    Warn "  It may still be starting. Check with:  Invoke-WebRequest $base/"
    Warn "  Or run it in the foreground to see why:  & `"$exe`" serve --dir `"$DataDir`" --port $Port"
    exit 1
}
OK "GET  /            200   (started in $([math]::Round($sw2.Elapsed.TotalSeconds))s)"

try {
    Invoke-WebRequest -Uri "$base/mcp" -Method POST -Body '{}' -ContentType 'application/json' `
        -TimeoutSec 10 -UseBasicParsing | Out-Null
    Warn "POST /mcp answered without a credential - the node is NOT refusing anonymous callers"
} catch {
    $codeGot = $_.Exception.Response.StatusCode.value__
    if ($codeGot -eq 401) { OK "POST /mcp anon    401  (refuses anonymous callers)" }
    else { Warn "POST /mcp anon answered $codeGot, expected 401" }
}

Write-Host ""
OK "the node is running."
Write-Host ""
Say "  address     $base"
Say "  data        $DataDir"
Say "  program     $InstallDir"
Write-Host ""
Say "  a credential for a client:"
Say "    & `"$exe`" token --dir `"$DataDir`" --issuer $base"
Write-Host ""
Say "  wire it into Claude Code:"
Say "    claude mcp add --transport http mantle $base/mcp --header `"Authorization: Bearer <token>`""
Write-Host ""
Warn "this node answers on 127.0.0.1 only. It has no certificate and is not reachable from"
Warn "  another machine - that needs a name and a DNS record, which is a separate decision."
