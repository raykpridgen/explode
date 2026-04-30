param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("cyl", "pli")]
    [string]$Dataset
)

switch ($Dataset) {
    "cyl" {
        $rangeStart = 1600
        $rangeEnd = 2100
        $baseUrl = "https://oceans11.lanl.gov/heat/cyl/cx241203_fp16_full"
        $outputDir = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "../data/cyl"))
    }
    "pli" {
        $rangeStart = 1600
        $rangeEnd = 1800
        $baseUrl = "https://oceans11.lanl.gov/heat/pli/pli240420"
        $outputDir = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "../data/cyl"))
    }
}

if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

Push-Location $outputDir
try {
    $urlsFile = Join-Path $outputDir "urls.txt"
    if (Test-Path $urlsFile) {
        Remove-Item $urlsFile -Force
    }

    foreach ($i in $rangeStart..$rangeEnd) {
        $id = "{0:D5}" -f $i
        $target = "$baseUrl/id$id/"

        # Parse wget spider output for discovered URLs, matching the original shell scripts.
        $matches = & wget -r -np -nd --spider $target 2>&1 |
            ForEach-Object { "$_" } |
            Where-Object { $_ -like "--*" } |
            ForEach-Object {
                $parts = $_ -split "\s+"
                if ($parts.Length -ge 3) {
                    $parts[2]
                }
            }

        if ($matches) {
            $matches | Add-Content -Path $urlsFile
        }
    }

    Write-Host "DOWNLOAD START"
    & aria2c -i $urlsFile -j 8 -x 8 -s 8 --continue=true --max-tries=0
}
finally {
    Pop-Location
}
