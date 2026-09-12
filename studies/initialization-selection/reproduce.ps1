param([string]$DestinationName = 'rfm-init-selection-reproduction')
$ErrorActionPreference = 'Stop'
if ($DestinationName -notmatch '^rfm-init-selection-[a-zA-Z0-9-]+$') { throw 'Use a sibling study directory name beginning rfm-init-selection-.' }
$studySource = $PSScriptRoot
$outputsRoot = Split-Path -Parent $studySource
$studyTarget = Join-Path $outputsRoot $DestinationName
if (Test-Path -LiteralPath $studyTarget) { throw "Refusing to overwrite $studyTarget" }
New-Item -ItemType Directory -Path $studyTarget | Out-Null
foreach ($name in @('src','vendor','LICENSE','NOTICE.md','protocol.md','plan.json','requirements.txt')) {
    Copy-Item -LiteralPath (Join-Path $studySource $name) -Destination $studyTarget -Recurse
}
$linuxTarget = (wsl -d Ubuntu-24.04 -- wslpath -a $studyTarget).Trim()
if ($LASTEXITCODE -ne 0) { throw 'wslpath failed' }
$researchPython = '/home/zs1430/.venvs/llm-training-lab/bin/python'
foreach ($stage in @('preflight.py','benchmark.py freeze','benchmark.py run','evaluate.py','verify.py','analyze.py','figure_readability.py')) {
    $stageArgs = $stage.Split(' ')
    $scriptPath = "$linuxTarget/src/$($stageArgs[0])"
    if ($stageArgs.Count -gt 1) {
        wsl -d Ubuntu-24.04 -- $researchPython $scriptPath $stageArgs[1]
    } else {
        wsl -d Ubuntu-24.04 -- $researchPython $scriptPath
    }
    if ($LASTEXITCODE -ne 0) { throw "Stage failed: $stage; partial results retained in $studyTarget" }
}
Write-Output "Reproduction complete: $studyTarget/facts.json"
