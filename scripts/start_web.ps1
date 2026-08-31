<#
P0-OPS-01: Web service startup script.

- Loads whitelisted KEY=VALUE pairs from the project root .env by default;
  variables already set in the current process environment always win.
- Parsing is pure text processing: no dot-sourcing, no Invoke-Expression.
- DEEPSEEK_API_KEY is only reported as set/unset, never printed.
- Checks .venv Python, embedding manifest, Qdrant meta.json and the API key,
  then starts Uvicorn in the foreground from the project root.
- -CheckOnly runs the checks only: exit 0 when everything passes, else 1.
#>
[CmdletBinding()]
param(
    # Uvicorn bind address (passed to --host).
    [string]$BindHost = '127.0.0.1',
    [ValidateRange(1, 65535)]
    [int]$Port = 8000,
    [string]$EnvFile = '',
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'

# Whitelist: exactly the env vars the application reads.
$whitelist = @(
    'DEEPSEEK_API_KEY',
    'DEEPSEEK_INTENT_MODEL',
    'DEEPSEEK_MODEL',
    'INTENT_CONFIDENCE_THRESHOLD',
    'MIN_RERANK_SCORE',
    'RAG_P0_ONLINE',
    'RAG_P0_HEAVY'
)

function Get-DotEnvValue {
    param([string]$Line)
    $v = $Line.Trim()
    if ($v.Length -ge 2) {
        $first = $v[0]
        $last = $v[$v.Length - 1]
        if (($first -eq '"' -and $last -eq '"') -or ($first -eq "'" -and $last -eq "'")) {
            return $v.Substring(1, $v.Length - 2)
        }
    }
    return $v
}

# Project root = parent of scripts/; never depends on the caller's cwd.
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $EnvFile) { $EnvFile = Join-Path $projectRoot '.env' }

$envFileLoaded = $false
if (Test-Path -LiteralPath $EnvFile -PathType Leaf) {
    foreach ($line in (Get-Content -LiteralPath $EnvFile -Encoding UTF8)) {
        if ($line -match '^\s*(?:#|;)') { continue }
        $idx = $line.IndexOf('=')
        if ($idx -lt 1) { continue }
        $key = $line.Substring(0, $idx).Trim()
        if ($key -cmatch '^[A-Z][A-Z0-9_]*$') {
            if ($whitelist -ccontains $key) {
                # Process env wins: only set when currently absent.
                if ($null -eq [Environment]::GetEnvironmentVariable($key)) {
                    [Environment]::SetEnvironmentVariable(
                        $key, (Get-DotEnvValue $line.Substring($idx + 1)), 'Process')
                }
            } else {
                Write-Host "[SKIP] $key is not in the whitelist, ignored."
            }
        }
    }
    $envFileLoaded = $true
}

$failures = New-Object System.Collections.Generic.List[string]
function Report-Check {
    param([bool]$Ok, [string]$Name, [string]$Detail)
    if ($Ok) {
        Write-Host ("[ OK ] {0} ({1})" -f $Name, $Detail)
    } else {
        Write-Host ("[FAIL] {0} ({1})" -f $Name, $Detail)
        $failures.Add($Name) | Out-Null
    }
}

if ($envFileLoaded) {
    Write-Host ("[ OK ] env file loaded: {0} (process env has priority)" -f $EnvFile)
} else {
    Write-Host ("[INFO] env file not found, using process environment only: {0}" -f $EnvFile)
}

# Check 1: .venv Python.
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
Report-Check (Test-Path -LiteralPath $pythonExe -PathType Leaf) 'venv_python' $pythonExe

# Check 2: embedding manifest.
$manifest = $null
$manifestPath = Join-Path $projectRoot 'output\embedding_manifest.json'
if (Test-Path -LiteralPath $manifestPath -PathType Leaf) {
    try { $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json } catch { $manifest = $null }
}
$manifestOk = ($null -ne $manifest) `
    -and $manifest.PSObject.Properties.Name -contains 'model_id' -and "$($manifest.model_id)".Trim() `
    -and $manifest.PSObject.Properties.Name -contains 'embedding_dimension' -and ($manifest.embedding_dimension -is [int]) -and $manifest.embedding_dimension -gt 0 `
    -and $manifest.PSObject.Properties.Name -contains 'chunk_count' -and ($manifest.chunk_count -is [int]) -and $manifest.chunk_count -gt 0 `
    -and $manifest.PSObject.Properties.Name -contains 'chunk_ids' -and @($manifest.chunk_ids).Count -eq $manifest.chunk_count
Report-Check $manifestOk 'embedding_manifest' $manifestPath

# Check 3: Qdrant meta.json (collection present, Cosine, size matches manifest).
$collectionName = 'rag_rules_bge_small_zh_v1_5'
$qdrantVectors = $null
$metaPath = Join-Path $projectRoot 'output\qdrant_storage\meta.json'
if (Test-Path -LiteralPath $metaPath -PathType Leaf) {
    try {
        $meta = Get-Content -LiteralPath $metaPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $collection = $meta.collections.$collectionName
        if ($null -ne $collection -and $null -ne $collection.vectors) { $qdrantVectors = $collection.vectors }
    } catch { $qdrantVectors = $null }
}
Report-Check ($null -ne $qdrantVectors) 'qdrant_collection' ("{0} :: {1}" -f $metaPath, $collectionName)

$dimensionOk = $manifestOk -and ($null -ne $qdrantVectors) `
    -and ($qdrantVectors.size -is [int]) -and ($qdrantVectors.size -eq $manifest.embedding_dimension) `
    -and ("$($qdrantVectors.distance)" -eq 'Cosine')
Report-Check $dimensionOk 'dimension_match' ("manifest={0} qdrant={1}" -f $(if ($manifestOk) { $manifest.embedding_dimension } else { '?' }), $(if ($qdrantVectors) { $qdrantVectors.size } else { '?' }))

# Check 4: API key presence only - never print the value.
$apiKey = [Environment]::GetEnvironmentVariable('DEEPSEEK_API_KEY')
Report-Check (-not [string]::IsNullOrWhiteSpace($apiKey)) 'deepseek_api_key' 'value not printed'

if ($CheckOnly) {
    if ($failures.Count -eq 0) {
        Write-Host 'CHECK ONLY: all checks passed.'
        exit 0
    }
    Write-Host ("CHECK ONLY: failed checks: {0}" -f ($failures -join ', '))
    exit 1
}

if ($failures.Count -gt 0) {
    Write-Host ("Startup aborted, failed checks: {0}" -f ($failures -join ', '))
    exit 1
}

Write-Host ("Starting uvicorn on http://{0}:{1}/ (Ctrl+C to stop)" -f $BindHost, $Port)
Set-Location -LiteralPath $projectRoot
& $pythonExe -m uvicorn customer_service_rag.api:app --host $BindHost --port $Port
exit $LASTEXITCODE
