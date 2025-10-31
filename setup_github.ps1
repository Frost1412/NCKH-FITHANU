# GitHub Repository Setup Script
# This script initializes git, creates initial commit, and pushes to GitHub
# Usage: .\setup_github.ps1

Write-Host "Setting up GitHub repository..." -ForegroundColor Cyan
Write-Host ""

# Check if git is installed
try {
    git --version | Out-Null
} catch {
    Write-Host "Git is not installed. Please install Git first." -ForegroundColor Red
    Write-Host "Download from: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}

# Copy GitHub README to root
Write-Host "Setting up README..." -ForegroundColor Cyan
Copy-Item -Path "GITHUB_README.md" -Destination "README_GITHUB.md" -Force
Write-Host "README prepared" -ForegroundColor Green

# Initialize git repository
Write-Host ""
Write-Host "Initializing Git repository..." -ForegroundColor Cyan
if (Test-Path ".git") {
    Write-Host "Git repository already exists" -ForegroundColor Yellow
} else {
    git init
    Write-Host "Git initialized" -ForegroundColor Green
}

# Add all files
Write-Host ""
Write-Host "Adding files..." -ForegroundColor Cyan
git add .
Write-Host "Files staged" -ForegroundColor Green

# Create initial commit
Write-Host ""
Write-Host "Creating initial commit..." -ForegroundColor Cyan
git commit -m "Initial commit: Speech Emotion Recognition MVP" -m "FastAPI backend with Wav2Vec2 and DistilRoBERTa fusion for online classroom emotion analysis"

Write-Host "Initial commit created" -ForegroundColor Green

# Instructions for GitHub
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Go to GitHub: https://github.com/new" -ForegroundColor White
Write-Host "2. Create a new repository named: NCKH-FITHANU" -ForegroundColor White
Write-Host "3. Do NOT initialize with README, .gitignore, or license" -ForegroundColor White
Write-Host "4. Run these commands:" -ForegroundColor White
Write-Host ""
Write-Host "   git branch -M main" -ForegroundColor Green
Write-Host "   git remote add origin https://github.com/YOUR_USERNAME/NCKH-FITHANU.git" -ForegroundColor Green
Write-Host "   git push -u origin main" -ForegroundColor Green
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Repository settings recommendations:" -ForegroundColor Yellow
Write-Host "  - Description: AI-powered Speech Emotion Recognition for Online Classrooms" -ForegroundColor White
Write-Host "  - Topics: speech-emotion-recognition, ai, transformer, wav2vec2, fastapi, python" -ForegroundColor White
Write-Host "  - License: MIT" -ForegroundColor White
Write-Host ""
Write-Host "Your local repository is ready!" -ForegroundColor Green
Write-Host ""
