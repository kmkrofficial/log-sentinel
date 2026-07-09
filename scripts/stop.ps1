param(
    [Parameter(Position = 0)]
    [ValidateSet('backend', 'frontend', 'all')]
    [string[]]$Target = @('all')
)

Set-StrictMode -Version Latest

. (Join-Path $PSScriptRoot 'common.ps1')

$targets = Resolve-Targets -Target $Target
$state = Load-State

foreach ($name in $targets) {
    $state = Stop-ManagedTarget -State $state -Name $name
}

Save-State -State $state