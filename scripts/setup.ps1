param(
    [ValidateSet('frontend', 'mlcore', 'backend-mlcore', 'all')]
    [string]$Profile,
    [switch]$InstallFlashAttention,
    [switch]$DownloadModels,
    [switch]$DownloadDatasets,
    [switch]$ForceProvision
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

function Get-YesNoChoice {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Prompt,
        [bool]$Default = $false
    )

    $defaultToken = if ($Default) { 'Y/n' } else { 'y/N' }

    while ($true) {
        $inputValue = Read-Host "$Prompt [$defaultToken]"
        if (-not $inputValue) {
            return $Default
        }

        switch ($inputValue.Trim().ToLowerInvariant()) {
            'y' { return $true }
            'yes' { return $true }
            'n' { return $false }
            'no' { return $false }
            default { Write-Host 'Please answer yes or no.' }
        }
    }
}

function Get-HuggingFaceCliPath {
    foreach ($candidate in @('huggingface-cli.exe', 'huggingface-cli')) {
        try {
            return (Get-Command $candidate -ErrorAction Stop).Source
        } catch {
        }
    }

    return $null
}

function Install-FlashAttentionWheel {
    param(
        [Parameter(Mandatory = $true)]
        [string]$VenvPython
    )

    $wheelPath = Get-ChildItem -Path (Join-Path (Get-RepoRoot) 'mlcore') -Filter 'flash_attn_3-*.whl' -File |
        Sort-Object Name -Descending |
        Select-Object -First 1

    if (-not $wheelPath) {
        Write-Host 'A local FlashAttention 3 wheel was not found in mlcore. Skipping wheel install.'
        return
    }

    Write-Host "Installing FlashAttention 3 wheel from $($wheelPath.FullName)..."
    & $VenvPython -m pip install --force-reinstall $wheelPath.FullName
}

function Invoke-BootstrapProvision {
    param(
        [Parameter(Mandatory = $true)]
        [string]$VenvPython,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $repoRoot = Get-RepoRoot
    $bootstrapPath = Join-Path $repoRoot 'scripts\bootstrap.py'
    if (-not (Test-Path $bootstrapPath)) {
        throw "Bootstrap script was not found at $bootstrapPath."
    }

    Push-Location $repoRoot
    try {
        & $VenvPython $bootstrapPath @Arguments
    } finally {
        Pop-Location
    }
}

function Download-HuggingFaceModels {
    param(
        [Parameter(Mandatory = $true)]
        [string]$VenvPython,
        [switch]$Force
    )

    if (-not $env:HF_TOKEN) {
        throw 'HF_TOKEN must be configured in the backend environment before downloading the gated Llama model.'
    }

    $arguments = @('--models')
    if ($Force) {
        $arguments += '--force'
    }
    Invoke-BootstrapProvision -VenvPython $VenvPython -Arguments $arguments
}

function Download-LogDatasets {
    param(
        [Parameter(Mandatory = $true)]
        [string]$VenvPython,
        [switch]$Force
    )

    $arguments = @('--datasets', 'all')
    if ($Force) {
        $arguments += '--force'
    }
    Invoke-BootstrapProvision -VenvPython $VenvPython -Arguments $arguments
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

$shouldInstallPythonDeps = $selectedProfile -in @('mlcore', 'backend-mlcore', 'all')
$shouldInstallFrontend = $selectedProfile -in @('frontend', 'all')

if (-not $PSBoundParameters.ContainsKey('InstallFlashAttention') -and $shouldInstallPythonDeps) {
    $InstallFlashAttention = Get-YesNoChoice -Prompt 'Install the local Flash Attention wheel if available?' -Default $false
}

if (-not $PSBoundParameters.ContainsKey('DownloadModels') -and $selectedProfile -in @('mlcore', 'backend-mlcore', 'all')) {
    $DownloadModels = Get-YesNoChoice -Prompt 'Download the required models using the server HF_TOKEN during setup?' -Default $false
}

if (-not $PSBoundParameters.ContainsKey('DownloadDatasets') -and $selectedProfile -in @('mlcore', 'backend-mlcore', 'all')) {
    $DownloadDatasets = Get-YesNoChoice -Prompt 'Download and checksum-verify all supported Zenodo datasets during setup?' -Default $false
}

switch ($selectedProfile) {
    'frontend' {
        Install-AndBuild-Frontend
    }
    'mlcore' {
        Install-PythonDependencies -VenvPython $venvPython
        if ($InstallFlashAttention) {
            Install-FlashAttentionWheel -VenvPython $venvPython
        }
        if ($DownloadModels) {
            Download-HuggingFaceModels -VenvPython $venvPython -Force:$ForceProvision
        }
        if ($DownloadDatasets) {
            Download-LogDatasets -VenvPython $venvPython -Force:$ForceProvision
        }
    }
    'backend-mlcore' {
        Install-PythonDependencies -VenvPython $venvPython
        if ($InstallFlashAttention) {
            Install-FlashAttentionWheel -VenvPython $venvPython
        }
        if ($DownloadModels) {
            Download-HuggingFaceModels -VenvPython $venvPython -Force:$ForceProvision
        }
        if ($DownloadDatasets) {
            Download-LogDatasets -VenvPython $venvPython -Force:$ForceProvision
        }
    }
    'all' {
        Install-PythonDependencies -VenvPython $venvPython
        if ($InstallFlashAttention) {
            Install-FlashAttentionWheel -VenvPython $venvPython
        }
        if ($DownloadModels) {
            Download-HuggingFaceModels -VenvPython $venvPython -Force:$ForceProvision
        }
        if ($DownloadDatasets) {
            Download-LogDatasets -VenvPython $venvPython -Force:$ForceProvision
        }
        Install-AndBuild-Frontend
    }
}

Write-Host "Setup complete for profile '$selectedProfile'."