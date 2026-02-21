$ErrorActionPreference = "Stop"

$profilePath = $PROFILE
$profileDir = Split-Path -Parent $profilePath

if (-not (Test-Path $profileDir)) {
    New-Item -ItemType Directory -Path $profileDir -Force | Out-Null
}

if (-not (Test-Path $profilePath)) {
    New-Item -ItemType File -Path $profilePath -Force | Out-Null
}

$markerStart = "# >>> logchange alias >>>"
$markerEnd = "# <<< logchange alias <<<"
$profileContent = [System.IO.File]::ReadAllText($profilePath)

if (-not $profileContent.Contains($markerStart)) {
    $block = @"
$markerStart
function Invoke-ChangeLog {
    param(
        [Parameter(ValueFromRemainingArguments = `$true)]
        [string[]]`$Args
    )

    `$scriptPath = Join-Path (Get-Location) 'append_change_log.ps1'
    if (-not (Test-Path `$scriptPath)) {
        `$scriptPath = 'E:\research_code\Agentic-Musrenbang-DSS\append_change_log.ps1'
    }

    if (-not (Test-Path `$scriptPath)) {
        Write-Error 'append_change_log.ps1 tidak ditemukan di folder aktif maupun path default.'
        return
    }

    & `$scriptPath @Args
}

Set-Alias -Name logchange -Value Invoke-ChangeLog -Scope Global
$markerEnd
"@

    [System.IO.File]::AppendAllText($profilePath, "`r`n$block`r`n")
    Write-Output "[OK] Alias block ditambahkan ke profile: $profilePath"
} else {
    Write-Output "[INFO] Alias block sudah ada di profile: $profilePath"
}

. $profilePath
$cmd = Get-Command logchange -ErrorAction SilentlyContinue
if ($null -eq $cmd) {
    Write-Output "[WARN] logchange belum aktif di sesi ini. Buka terminal baru atau jalankan: . `$PROFILE"
} else {
    Write-Output "[OK] logchange aktif: $($cmd.CommandType)"
}
