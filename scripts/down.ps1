# Configuration
$baseUrl = "https://oceans11.lanl.gov/heat/"
$targets = @("cyl/", "pli/")
$outputRoot = Get-Location # Downloads to current folder

foreach ($target in $targets) {
    $targetUrl = $baseUrl + $target
    Write-Host "Scanning $targetUrl..." -ForegroundColor Cyan
    
    # Create the local base folder (cyl/ or pli/)
    $localTargetDir = Join-Path $outputRoot $target.TrimEnd('/')
    if (!(Test-Path $localTargetDir)) { New-Item -ItemType Directory -Path $localTargetDir }

    # Fetch the index page of the subfolder
    $response = Invoke-WebRequest -Uri $targetUrl -UseBasicParsing
    
    # Filter for links that look like 'idXXXXX/'
    $idLinks = $response.Links | Where-Object { $_.href -match '^id\d+/' } | Select-Object -ExpandProperty href

    foreach ($id in $idLinks) {
        $idUrl = $targetUrl + $id
        $localIdDir = Join-Path $localTargetDir $id.TrimEnd('/')
        
        if (!(Test-Path $localIdDir)) { New-Item -ItemType Directory -Path $localIdDir }
        
        Write-Host "  Downloading from $id..." -ForegroundColor Gray
        
        # Get the files inside the idXXXXX/ folder
        $idPage = Invoke-WebRequest -Uri $idUrl -UseBasicParsing
        $files = $idPage.Links | Where-Object { $_.href -match '\.npz$' } | Select-Object -ExpandProperty href

        foreach ($file in $files) {
            $fileUrl = $idUrl + $file
            $destPath = Join-Path $localIdDir $file
            
            if (!(Test-Path $destPath)) {
                Write-Host "    -> $file"
                Invoke-WebRequest -Uri $fileUrl -OutFile $destPath
            } else {
                Write-Host "    -> $file (Skipped: Already exists)" -ForegroundColor DarkYellow
            }
        }
    }
}
Write-Host "Download Complete!" -ForegroundColor Green