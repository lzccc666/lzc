[CmdletBinding()]
param(
    [string]$CodeDir = 'D:\study\Python.Project\lzc\code',
    [string]$MaterialsDir = 'D:\study\Python.Project\lzc\thesis_materials'
)
$ErrorActionPreference = 'Stop'
$Repository = 'https://github.com/lzccc666/lzc.git'

function Invoke-GitChecked {
    param([string[]]$GitArguments)
    $result = & git @GitArguments
    if ($LASTEXITCODE -ne 0) { throw "Git failed: $($GitArguments -join ' ')" }
    return $result
}

if ($env:OS -ne 'Windows_NT') { throw 'Run this script on the target Windows computer.' }
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'Git for Windows is required.' }
$CodeDir = [IO.Path]::GetFullPath($CodeDir).TrimEnd('\')
$MaterialsDir = [IO.Path]::GetFullPath($MaterialsDir).TrimEnd('\')
if ($CodeDir -eq $MaterialsDir -or $MaterialsDir.StartsWith($CodeDir + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'MaterialsDir must be separate from CodeDir.'
}
if (-not (Test-Path -LiteralPath $CodeDir)) {
    $null = New-Item -ItemType Directory -Force -Path (Split-Path -Parent $CodeDir)
    Invoke-GitChecked -GitArguments @('-c','core.autocrlf=false','clone','--config','core.autocrlf=false',
        '--config','core.longpaths=true',$Repository,$CodeDir)
} elseif (-not (Test-Path -LiteralPath (Join-Path $CodeDir '.git'))) {
    if (@(Get-ChildItem -LiteralPath $CodeDir -Force).Count -gt 0) {
        throw 'CodeDir already contains non-repository files; nothing was overwritten.'
    }
    Invoke-GitChecked -GitArguments @('-c','core.autocrlf=false','clone','--config','core.autocrlf=false',
        '--config','core.longpaths=true',$Repository,$CodeDir)
}
$origin = (Invoke-GitChecked -GitArguments @('-C',$CodeDir,'remote','get-url','origin') | Out-String).Trim()
if ($origin.TrimEnd('/') -notin @($Repository, 'https://github.com/lzccc666/lzc')) {
    throw 'Existing repository has a different origin; nothing was changed.'
}
$Package = Join-Path $CodeDir 'thesis_materials'
if (-not (Test-Path -LiteralPath (Join-Path $Package 'manifest.json'))) {
    throw 'This checkout does not contain the writing materials. Update the repository deliberately, then rerun.'
}
$manifest = Get-Content -LiteralPath (Join-Path $Package 'manifest.json') -Encoding UTF8 -Raw | ConvertFrom-Json
foreach ($entry in $manifest.files) {
    $path = Join-Path $Package $entry.path
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing package file: $($entry.path)" }
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) {
        throw "Package hash differs: $($entry.path)"
    }
}
$sourceRecords = Get-Content -LiteralPath (Join-Path $Package $manifest.code_inventory) -Encoding UTF8 -Raw | ConvertFrom-Json
foreach ($entry in $sourceRecords) {
    $path = Join-Path $CodeDir $entry.path
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing source: $($entry.path)" }
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) {
        throw "Source differs from the indexed version: $($entry.path). Existing edits were preserved."
    }
}
if (Test-Path -LiteralPath $MaterialsDir) {
    $existingManifest = Join-Path $MaterialsDir 'manifest.json'
    if (-not (Test-Path -LiteralPath $existingManifest)) { throw 'MaterialsDir exists without our manifest; preserved.' }
    if ((Get-FileHash -LiteralPath $existingManifest).Hash -ne (Get-FileHash -LiteralPath (Join-Path $Package 'manifest.json')).Hash) {
        throw 'MaterialsDir contains a different version; preserved.'
    }
    foreach ($entry in $manifest.files) {
        $path = Join-Path $MaterialsDir $entry.path
        if (-not (Test-Path -LiteralPath $path)) { throw "Existing materials are incomplete: $($entry.path)" }
        if ((Get-FileHash -LiteralPath $path).Hash.ToLowerInvariant() -ne $entry.sha256) {
            throw "Existing material was edited: $($entry.path). Nothing was overwritten."
        }
    }
} else {
    $null = New-Item -ItemType Directory -Force -Path (Split-Path -Parent $MaterialsDir)
    Copy-Item -LiteralPath $Package -Destination $MaterialsDir -Recurse
}
foreach ($entry in $manifest.files) {
    $copied = Join-Path $MaterialsDir $entry.path
    if ((Get-FileHash -LiteralPath $copied).Hash.ToLowerInvariant() -ne $entry.sha256) {
        throw "Copied file verification failed: $($entry.path)"
    }
}
$head = (Invoke-GitChecked -GitArguments @('-C',$CodeDir,'rev-parse','HEAD') | Out-String).Trim()
$receipt = [ordered]@{
    windows_verified = $true
    code_directory = $CodeDir
    materials_directory = $MaterialsDir
    git_version = (Invoke-GitChecked -GitArguments @('--version') | Out-String).Trim()
    git_commit = $head
    source_baseline_commit = $manifest.source_baseline_commit
    source_files_verified = @($sourceRecords).Count
    material_files_verified = @($manifest.files).Count
    created_utc = [DateTime]::UtcNow.ToString('o')
    experiments_started = $false
    goal_resumed = $false
}
$receiptPath = Join-Path $MaterialsDir 'WINDOWS_SETUP_RECEIPT.json'
if (Test-Path -LiteralPath $receiptPath) {
    $receiptPath = Join-Path $MaterialsDir ('WINDOWS_SETUP_RECEIPT_' + [Guid]::NewGuid().ToString('N') + '.json')
}
$receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $receiptPath -Encoding UTF8
Write-Host "Code ready: $CodeDir"
Write-Host "Writing materials ready: $MaterialsDir"
Write-Host "Verified commit: $head"
Write-Host "Receipt: $receiptPath"
