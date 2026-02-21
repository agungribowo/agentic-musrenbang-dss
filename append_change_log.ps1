param(
    [Parameter(Mandatory = $false)]
    [string]$Summary = "Update session",

    [Parameter(Mandatory = $false)]
    [string]$Scope = "General",

    [Parameter(Mandatory = $false)]
    [string]$Files = "",

    [Parameter(Mandatory = $false)]
    [string]$Details = "",

    [Parameter(Mandatory = $false)]
    [string]$VerifyCommand = "",

    [Parameter(Mandatory = $false)]
    [string]$VerifyResult = "",

    [Parameter(Mandatory = $false)]
    [string]$Notes = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
$logPath = Join-Path $repoRoot "change.md"
$now = Get-Date
$timestamp = $now.ToString("o")
$today = $now.ToString("yyyy-MM-dd")

if (-not (Test-Path $logPath)) {
    @"
# Change Log (Append-Only)

Dokumen ini **tidak di-overwrite**. Setiap sesi kerja menambahkan entry baru di bagian paling bawah.

## Format Entry

```markdown
## YYYY-MM-DD

### Session YYYY-MM-DDTHH:mm:sszzz
- Ringkasan: ...
- Scope: ...
- File Changed:
  - file_a.py
- Detail:
  - Perubahan penting
- Verification:
  - Command: ...
  - Result: ...
- Notes: ...
```
"@ | Set-Content -Path $logPath -Encoding UTF8
}

$content = Get-Content -Path $logPath -Raw

if ($content -notmatch "(?m)^##\s+$today\s*$") {
    Add-Content -Path $logPath -Value "`r`n## $today`r`n"
}

$fileLines = @()
if ($Files.Trim()) {
    $fileLines = $Files.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ }
}

$detailLines = @()
if ($Details.Trim()) {
    $detailLines = $Details.Split(';') | ForEach-Object { $_.Trim() } | Where-Object { $_ }
}

$entry = @()
$entry += ""
$entry += "### Session $timestamp"
$entry += "- Ringkasan: $Summary"
$entry += "- Scope: $Scope"
$entry += "- File Changed:"
if ($fileLines.Count -gt 0) {
    foreach ($line in $fileLines) {
        $entry += "  - $line"
    }
} else {
    $entry += "  - (tidak diisi)"
}
$entry += "- Detail:"
if ($detailLines.Count -gt 0) {
    foreach ($line in $detailLines) {
        $entry += "  - $line"
    }
} else {
    $entry += "  - (tidak diisi)"
}
$entry += "- Verification:"
$entry += "  - Command: $VerifyCommand"
$entry += "  - Result: $VerifyResult"
$entry += "- Notes: $Notes"

Add-Content -Path $logPath -Value ($entry -join "`r`n")

Write-Output "[OK] Entry ditambahkan ke change.md"
Write-Output "Timestamp: $timestamp"
