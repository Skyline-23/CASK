param(
    [string]$OutputZip = (Join-Path $PSScriptRoot "cask_arxiv_source.zip"),
    [switch]$Verify,
    [switch]$KeepStaging
)

$ErrorActionPreference = "Stop"

$paperDir = Resolve-Path $PSScriptRoot
$repoRoot = Resolve-Path (Join-Path $paperDir "..")
$assetDir = Join-Path $repoRoot "docs\assets"
$stagingRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("cask_arxiv_submit_" + [guid]::NewGuid().ToString("N"))
$figDir = Join-Path $stagingRoot "figures"

function Write-Utf8NoBom {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Value
    )
    $encoding = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($Path, $Value, $encoding)
}

function Require-File {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required file not found: $Path"
    }
}

try {
    New-Item -ItemType Directory -Path $figDir -Force | Out-Null

    $mainAuthor = Join-Path $paperDir "main_author.tex"
    $mainShared = Join-Path $paperDir "main_shared.tex"
    $content = Join-Path $paperDir "content.tex"
    $references = Join-Path $paperDir "references.bib"
    $style = Join-Path $paperDir "cask_arxiv.sty"

    @($mainAuthor, $mainShared, $content, $references, $style) | ForEach-Object {
        Require-File -Path $_
    }
    if (-not (Test-Path -LiteralPath $assetDir -PathType Container)) {
        throw "Figure asset directory not found: $assetDir"
    }

    Copy-Item -LiteralPath $mainAuthor -Destination (Join-Path $stagingRoot "main.tex") -Force
    Copy-Item -LiteralPath $content -Destination (Join-Path $stagingRoot "content.tex") -Force
    Copy-Item -LiteralPath $references -Destination (Join-Path $stagingRoot "references.bib") -Force
    Copy-Item -LiteralPath $style -Destination (Join-Path $stagingRoot "cask_arxiv.sty") -Force

    $sharedText = Get-Content -LiteralPath $mainShared -Raw
    $sharedText = $sharedText.Replace("\graphicspath{{../docs/assets/}}", "\graphicspath{{figures/}}")
    Write-Utf8NoBom -Path (Join-Path $stagingRoot "main_shared.tex") -Value $sharedText

    $figures = Get-ChildItem -LiteralPath $assetDir -Filter "*.pdf" -File | Sort-Object Name
    if ($figures.Count -eq 0) {
        throw "No PDF figures found in: $assetDir"
    }
    $figures | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $figDir -Force
    }

    if (Test-Path -LiteralPath $OutputZip) {
        Remove-Item -LiteralPath $OutputZip -Force
    }
    Compress-Archive -Path (Join-Path $stagingRoot "*") -DestinationPath $OutputZip -Force

    if ($Verify) {
        $verifyRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("cask_arxiv_verify_" + [guid]::NewGuid().ToString("N"))
        New-Item -ItemType Directory -Path $verifyRoot -Force | Out-Null
        try {
            Expand-Archive -Path $OutputZip -DestinationPath $verifyRoot -Force
            Push-Location $verifyRoot
            latexmk -g -pdf -interaction=nonstopmode -halt-on-error main.tex
            if ($LASTEXITCODE -ne 0) {
                throw "latexmk verification failed with exit code $LASTEXITCODE"
            }
        }
        finally {
            Pop-Location
            Remove-Item -LiteralPath $verifyRoot -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    Write-Host "Created arXiv source package: $OutputZip"
    Write-Host "Included $($figures.Count) PDF figures from docs/assets."
}
finally {
    if ($KeepStaging) {
        Write-Host "Kept staging directory: $stagingRoot"
    }
    else {
        Remove-Item -LiteralPath $stagingRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}
