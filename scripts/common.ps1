Set-StrictMode -Version Latest

$Script:RepoRoot = Split-Path -Parent $PSScriptRoot
$Script:RuntimeDir = Join-Path $PSScriptRoot '.runtime'
$Script:LogsDir = Join-Path $Script:RuntimeDir 'logs'
$Script:StateFile = Join-Path $Script:RuntimeDir 'process-state.json'
$Script:BackendPort = 8000
$Script:FrontendPort = 5173

function Get-RepoRoot {
    return $Script:RepoRoot
}

function Ensure-RuntimeDirectories {
    foreach ($path in @($Script:RuntimeDir, $Script:LogsDir)) {
        if (-not (Test-Path $path)) {
            New-Item -ItemType Directory -Path $path | Out-Null
        }
    }
}

function New-DefaultState {
    return [ordered]@{
        backend = $null
        frontend = $null
    }
}

function Load-State {
    Ensure-RuntimeDirectories

    $state = New-DefaultState
    if (-not (Test-Path $Script:StateFile)) {
        return $state
    }

    $raw = Get-Content -Path $Script:StateFile -Raw
    if (-not $raw.Trim()) {
        return $state
    }

    try {
        $loaded = $raw | ConvertFrom-Json
    } catch {
        return $state
    }

    if ($loaded.PSObject.Properties.Name -contains 'backend') {
        $state.backend = $loaded.backend
    }

    if ($loaded.PSObject.Properties.Name -contains 'frontend') {
        $state.frontend = $loaded.frontend
    }

    return $state
}

function Save-State {
    param(
        [Parameter(Mandatory = $true)]
        [object]$State
    )

    Ensure-RuntimeDirectories

    if (-not $State.backend -and -not $State.frontend) {
        if (Test-Path $Script:StateFile) {
            Remove-Item -Path $Script:StateFile -Force
        }
        return
    }

    $State | ConvertTo-Json -Depth 5 | Set-Content -Path $Script:StateFile -Encoding UTF8
}

function Resolve-Targets {
    param(
        [string[]]$Target
    )

    if (-not $Target -or $Target.Count -eq 0 -or $Target -contains 'all') {
        return @('backend', 'frontend')
    }

    $resolved = @()
    foreach ($item in $Target) {
        if ($item -and $item -notin $resolved) {
            $resolved += $item
        }
    }

    return $resolved
}

function Test-ProcessRunning {
    param(
        $ProcessId
    )

    if (-not $ProcessId) {
        return $false
    }

    try {
        Get-Process -Id ([int]$ProcessId) -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Get-ListeningProcessId {
    param(
        [Parameter(Mandatory = $true)]
        [int]$Port
    )

    try {
        return (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop | Select-Object -First 1 -ExpandProperty OwningProcess)
    } catch {
        return $null
    }
}

function Get-VenvPythonPath {
    $pythonPath = Join-Path $Script:RepoRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path $pythonPath)) {
        throw "Python virtual environment not found at '$pythonPath'. Run scripts\setup.ps1 first."
    }

    return $pythonPath
}

function Get-FrontendRunnerPath {
    $vitePath = Join-Path $Script:RepoRoot 'frontend\node_modules\.bin\vite.cmd'
    if (-not (Test-Path $vitePath)) {
        throw "Frontend runner not found at '$vitePath'. Run scripts\setup.ps1 first."
    }

    return $vitePath
}

function New-ProcessRecord {
    param(
        [Parameter(Mandatory = $true)]
        [System.Diagnostics.Process]$Process,
        [Parameter(Mandatory = $true)]
        [string]$StdoutPath,
        [Parameter(Mandatory = $true)]
        [string]$StderrPath,
        [Parameter(Mandatory = $true)]
        [int]$Port,
        [Parameter(Mandatory = $true)]
        [string]$CommandLine
    )

    return [pscustomobject]@{
        pid = $Process.Id
        stdout = $StdoutPath
        stderr = $StderrPath
        port = $Port
        command = $CommandLine
        startedAt = (Get-Date).ToString('o')
    }
}

function Remove-LogFiles {
    param(
        [Parameter(Mandatory = $true)]
        [string]$StdoutPath,
        [Parameter(Mandatory = $true)]
        [string]$StderrPath
    )

    foreach ($path in @($StdoutPath, $StderrPath)) {
        if (Test-Path $path) {
            Remove-Item -Path $path -Force
        }
    }
}

function Start-BackendProcess {
    param(
        [Parameter(Mandatory = $true)]
        [object]$State
    )

    $trackedPid = $null
    if ($null -ne $State.backend -and $State.backend.PSObject.Properties.Name -contains 'pid') {
        $trackedPid = $State.backend.pid
    }

    if (Test-ProcessRunning $trackedPid) {
        Write-Host "Backend is already running with PID $trackedPid."
        return $State
    }

    $portPid = Get-ListeningProcessId -Port $Script:BackendPort
    if ($portPid) {
        Write-Host "Port $($Script:BackendPort) is already in use by PID $portPid. Backend start skipped."
        $State.backend = [pscustomobject]@{
            pid = $portPid
            stdout = $null
            stderr = $null
            port = $Script:BackendPort
            command = 'external'
            startedAt = $null
        }
        return $State
    }

    Ensure-RuntimeDirectories
    $stdoutPath = Join-Path $Script:LogsDir 'backend.stdout.log'
    $stderrPath = Join-Path $Script:LogsDir 'backend.stderr.log'
    Remove-LogFiles -StdoutPath $stdoutPath -StderrPath $stderrPath

    $pythonPath = Get-VenvPythonPath
    $workingDirectory = Join-Path $Script:RepoRoot 'backend'
    $arguments = @('-m', 'uvicorn', 'api.main:app', '--host', '127.0.0.1', '--port', "$($Script:BackendPort)")

    $process = Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $workingDirectory -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
    $State.backend = New-ProcessRecord -Process $process -StdoutPath $stdoutPath -StderrPath $stderrPath -Port $Script:BackendPort -CommandLine 'python -m uvicorn api.main:app --host 127.0.0.1 --port 8000'

    Write-Host "Started backend on http://127.0.0.1:$($Script:BackendPort) (PID $($process.Id))."
    Write-Host "Stopping this backend process tree also kills any application-started ML runs because they execute inside the backend process."
    return $State
}

function Start-FrontendProcess {
    param(
        [Parameter(Mandatory = $true)]
        [object]$State
    )

    $trackedPid = $null
    if ($null -ne $State.frontend -and $State.frontend.PSObject.Properties.Name -contains 'pid') {
        $trackedPid = $State.frontend.pid
    }

    if (Test-ProcessRunning $trackedPid) {
        Write-Host "Frontend is already running with PID $trackedPid."
        return $State
    }

    $portPid = Get-ListeningProcessId -Port $Script:FrontendPort
    if ($portPid) {
        Write-Host "Port $($Script:FrontendPort) is already in use by PID $portPid. Frontend start skipped."
        $State.frontend = [pscustomobject]@{
            pid = $portPid
            stdout = $null
            stderr = $null
            port = $Script:FrontendPort
            command = 'external'
            startedAt = $null
        }
        return $State
    }

    Ensure-RuntimeDirectories
    $stdoutPath = Join-Path $Script:LogsDir 'frontend.stdout.log'
    $stderrPath = Join-Path $Script:LogsDir 'frontend.stderr.log'
    Remove-LogFiles -StdoutPath $stdoutPath -StderrPath $stderrPath

    $vitePath = Get-FrontendRunnerPath
    $workingDirectory = Join-Path $Script:RepoRoot 'frontend'
    $arguments = @('--host', '127.0.0.1', '--port', "$($Script:FrontendPort)")

    $process = Start-Process -FilePath $vitePath -ArgumentList $arguments -WorkingDirectory $workingDirectory -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
    $State.frontend = New-ProcessRecord -Process $process -StdoutPath $stdoutPath -StderrPath $stderrPath -Port $Script:FrontendPort -CommandLine 'vite --host 127.0.0.1 --port 5173'

    Write-Host "Started frontend on http://127.0.0.1:$($Script:FrontendPort) (PID $($process.Id))."
    return $State
}

function Stop-ManagedTarget {
    param(
        [Parameter(Mandatory = $true)]
        [object]$State,
        [Parameter(Mandatory = $true)]
        [ValidateSet('backend', 'frontend')]
        [string]$Name
    )

    $entry = $State.$Name
    $processId = $null
    if ($null -ne $entry -and $entry.PSObject.Properties.Name -contains 'pid' -and $entry.pid) {
        $processId = [int]$entry.pid
    }

    if (-not (Test-ProcessRunning $processId)) {
        $port = if ($Name -eq 'backend') { $Script:BackendPort } else { $Script:FrontendPort }
        $processId = Get-ListeningProcessId -Port $port
    }

    if (-not $processId) {
        Write-Host "$Name is not running."
        $State.$Name = $null
        return $State
    }

    & taskkill /PID $processId /T /F 2>$null | Out-Null
    Write-Host "Stopped $Name process tree (PID $processId)."
    $State.$Name = $null
    return $State
}