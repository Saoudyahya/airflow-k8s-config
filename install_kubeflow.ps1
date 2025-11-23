$maxRetries = 50
$retryCount = 0

while ($retryCount -lt $maxRetries) {
    Write-Host "Attempting Kubeflow installation (Attempt $($retryCount + 1))..."
    
    # Run Kustomize and apply
    kustomize build example | kubectl apply -f -
    
    # Check if the last command succeeded
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Installation command finished successfully!" -ForegroundColor Green
        break
    }
    
    Write-Host "Resources not ready yet. Retrying in 10 seconds..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10
    $retryCount++
}