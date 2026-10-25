# Build using the already installed original image after verifying its identity.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$baseImage = 'umdetree/signal-detector:0.2.1'
$expectedBaseId = 'sha256:38d12791e62a026ac1cbeb0064e4ddea155e36168c9ec854fd35e65d919c7ef2'
$actualBaseId = docker image inspect $baseImage --format '{{.Id}}'
if ($LASTEXITCODE -ne 0 -or $actualBaseId -ne $expectedBaseId) {
    throw "Required original 0.2.1 image is missing or differs from $expectedBaseId"
}
Push-Location -LiteralPath $projectRoot
try {
    docker build --pull=false --provenance=false --progress=plain --build-arg "DETECTOR_BASE=$baseImage" -f docker/Dockerfile -t weitaoccc/signal-recognition:0.1.1 .
    if ($LASTEXITCODE -ne 0) { throw 'Integrated image build failed.' }
} finally {
    Pop-Location
}
