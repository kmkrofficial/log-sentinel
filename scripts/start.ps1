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
    switch ($name) {
        'backend' {
            $state = Start-BackendProcess -State $state
        }
        'frontend' {
            $state = Start-FrontendProcess -State $state
        }
    }
}

Save-State -State $state