# setup_git.ps1
# This script initializes git, adds the remote, and pushes your code to GitHub.

Write-Host "🚀 Initializing Git Repository..." -ForegroundColor Cyan

# 1. Initialize
if (!(Test-Path .git)) {
    git init
    Write-Host "✅ Git initialized." -ForegroundColor Green
} else {
    Write-Host "ℹ️ Git already initialized." -ForegroundColor Yellow
}

# 2. Add Remote
$remoteUrl = "https://github.com/Bhargavk055/chest-xray-explainable-ai.git"
$existingRemote = git remote get-url origin 2>$null
if ($existingRemote -eq $null) {
    git remote add origin $remoteUrl
    Write-Host "✅ Remote added: $remoteUrl" -ForegroundColor Green
} else {
    git remote set-url origin $remoteUrl
    Write-Host "✅ Remote updated: $remoteUrl" -ForegroundColor Green
}

# 3. Commit and Push
Write-Host "📦 Adding files and committing..." -ForegroundColor Cyan
git add .
git commit -m "Initial commit: Chest X-ray Explainable AI pipeline"

Write-Host "📤 Pushing to GitHub (main branch)..." -ForegroundColor Cyan
git branch -M main
git push -u origin main

Write-Host "`n✨ DONE! Refresh your GitHub page to see the code." -ForegroundColor Green
pause
