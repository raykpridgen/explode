# Configuration
$baseUrl = "https://oceans11.lanl.gov/heat/"
$targets = @("cyl/", "pli/")
$outputRoot = Get-Location

foreach ($target in $targets) {
    $targetUrl = $baseUrl + $target
    Write-Host "`nScanning: $targetUrl" -ForegroundColor Cyan
    
    # Create local base folder
    $localTargetDir = Join-Path $outputRoot $target.TrimEnd('/')
    if (!(Test-Path $localTargetDir)) { New-Item -ItemType Directory -Path $localTargetDir | Out-Null }

    try {
        $html = (Invoke-WebRequest -Uri $targetUrl -UseBasicParsing).Content
        # Match anything that looks like href="id0000x/"
        $idFolders = [regex]::Matches($html, 'href="(id\d+/)"') | ForEach-Object { $_.Groups[1].Value }

        if ($idFolders.Count -eq 0) {
            Write-Host "  No 'id' folders found. Check if the URL is accessible." -ForegroundColor Red
            continue
        }

        foreach ($id in $idFolders) {
            $idUrl = $targetUrl + $id
            $localIdDir = Join-Path $localTargetDir $id.TrimEnd('/')
            if (!(Test-Path $localIdDir)) { New-Item -ItemType Directory -Path $localIdDir | Out-Null }

            Write-Host "  Entering $id" -ForegroundColor Gray
            
            # Fetch files inside the ID folder
            $idHtml = (Invoke-WebRequest -Uri $idUrl -UseBasicParsing).Content
            $files = [regex]::Matches($idHtml, 'href="([^"]+\.npz)"') | ForEach-Object { $_.Groups[1].Value }

            foreach ($file in $files) {
                $fileUrl = $idUrl + $file
                $destPath = Join-Path $localIdDir $file
                
                if (!(Test-Path $destPath)) {
                    Write-Host "    Downloading: $file" -ForegroundColor White
                    Invoke-WebRequest -Uri $fileUrl -OutFile $destPath
                } else {
                    Write-Host "    Skipped: $file (Exists)" -ForegroundColor DarkYellow
                }
            }
        }
    } catch {
        Write-Host "  Error accessing $targetUrl : $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host "`nFinished processing all chunks." -ForegroundColor Green