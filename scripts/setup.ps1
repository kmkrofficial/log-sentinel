param(
    [ValidateSet('frontend', 'mlcore', 'backend-mlcore', 'all')]
    [string]$Profile
)

Set-StrictMode -Version Latest

. (Join-Path $PSScriptRoot 'common.ps1')

function Get-PythonVersionTag {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonPath
    )

    return (& $PythonPath -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
}

function Get-Python311Executable {
    try {
        $pyLauncher = Get-Command py -ErrorAction Stop
        $pyExecutable = (& $pyLauncher.Source -3.11 -c "import sys; print(sys.executable)" 2>$null).Trim()
        if ($pyExecutable) {
            return $pyExecutable
        }
    } catch {
    }

    try {
        $pythonCommand = Get-Command python -ErrorAction Stop
        if ((Get-PythonVersionTag -PythonPath $pythonCommand.Source) -eq '3.11') {
            return $pythonCommand.Source
        }
    } catch {
    }

    throw 'Python 3.11 was not found. Install Python 3.11 and rerun this script.'
}

function Resolve-SetupProfile {
    param(
        [string]$RequestedProfile
    )

    if ($RequestedProfile) {
        return $RequestedProfile
    }

    Write-Host 'Choose what to prepare:'
    Write-Host '  [1] frontend only'
    Write-Host '  [2] mlcore only'
    Write-Host '  [3] backend and mlcore only'
    Write-Host '  [4] all 3'

    while ($true) {
        $choice = Read-Host 'Enter 1, 2, 3, or 4'
        switch ($choice) {
            '1' { return 'frontend' }
            '2' { return 'mlcore' }
            '3' { return 'backend-mlcore' }
            '4' { return 'all' }
            default { Write-Host 'Invalid choice. Try again.' }
        }
    }
}

function Ensure-VirtualEnvironment {
    $repoRoot = Get-RepoRoot
    $venvPath = Join-Path $repoRoot '.venv'
    $venvPython = Join-Path $venvPath 'Scripts\python.exe'

    $python311 = Get-Python311Executable
    $baseVersion = Get-PythonVersionTag -PythonPath $python311
    if ($baseVersion -ne '3.11') {
        throw "Expected Python 3.11, but resolved '$baseVersion'."
    }

    Write-Host "Verified base Python interpreter: $python311 ($baseVersion)."

    if (-not (Test-Path $venvPython)) {
        Write-Host 'Creating virtual environment at .venv...'
        & $python311 -m venv $venvPath
    } else {
        Write-Host 'Using existing virtual environment at .venv.'
    }

    $venvVersion = Get-PythonVersionTag -PythonPath $venvPython
    if ($venvVersion -ne '3.11') {
        throw "The virtual environment is using Python $venvVersion. Remove .venv and rerun this script with Python 3.11 available."
    }

    Write-Host "Verified virtual environment Python version: $venvVersion."
    return $venvPython
}

function Get-NpmCommandPath {
    foreach ($candidate in @('npm.cmd', 'npm')) {
        try {
            return (Get-Command $candidate -ErrorAction Stop).Source
        } catch {
        }
    }

    throw 'npm was not found. Install Node.js and rerun this script.'
}

function Install-PythonDependencies {
    param(
        [Parameter(Mandatory = $true)]
        [string]$VenvPython
    )

    $requirementsPath = Join-Path (Get-RepoRoot) 'backend\requirements.txt'
    Write-Host "Installing Python dependencies from $requirementsPath..."
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install -r $requirementsPath
}

function Install-AndBuild-Frontend {
    $npmPath = Get-NpmCommandPath
    $frontendDir = Join-Path (Get-RepoRoot) 'frontend'

    Write-Host 'Installing frontend dependencies...'
    Push-Location $frontendDir
    try {
        & $npmPath install
        Write-Host 'Building frontend...'
        & $npmPath run build
    } finally {
        Pop-Location
    }
}

$selectedProfile = Resolve-SetupProfile -RequestedProfile $Profile
$venvPython = Ensure-VirtualEnvironment

switch ($selectedProfile) {
    'frontend' {
        Install-AndBuild-Frontend
    }
    'mlcore' {
        Install-PythonDependencies -VenvPython $venvPython
    }
    'backend-mlcore' {
        Install-PythonDependencies -VenvPython $venvPython
    }
    'all' {
        Install-PythonDependencies -VenvPython $venvPython
        Install-AndBuild-Frontend
    }
}

Write-Host "Setup complete for profile '$selectedProfile'."