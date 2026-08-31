# PowerShell 1-Click Deployment Script for Google Cloud Run
param(
    [string]$ProjectId = "bandhu-agentic-2026",
    [string]$Region = "us-central1",
    [string]$ServiceName = "bandhu-agent"
)

Write-Host "🚀 Starting Bandhu Google Cloud Run Deployment..." -ForegroundColor Cyan

# 1. Verify gcloud CLI
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    Write-Error "Google Cloud SDK (gcloud) is not installed or not in PATH."
    exit 1
}

# 2. Verify GEMINI_API_KEY is set
if (-not $env:GEMINI_API_KEY) {
    Write-Error "GEMINI_API_KEY environment variable is not set. Set it before deploying:`n  `$env:GEMINI_API_KEY = 'your-key-here'"
    exit 1
}

# 3. Set Active GCP Project
Write-Host "📌 Setting GCP Project to: $ProjectId" -ForegroundColor Yellow
gcloud config set project $ProjectId

# 4. Build & Deploy directly from source via Cloud Build
Write-Host "📦 Building and Deploying to Google Cloud Run ($Region)..." -ForegroundColor Yellow
gcloud run deploy $ServiceName `
    --source . `
    --region $Region `
    --platform managed `
    --allow-unauthenticated `
    --memory 2Gi `
    --cpu 2 `
    --set-env-vars "GCP_PROJECT_ID=$ProjectId,ENVIRONMENT=production,GEMINI_MODEL=gemini-3.7-flash,GEMINI_FALLBACK_MODEL=gemini-3.5-flash-lite,GEMINI_API_KEY=$($env:GEMINI_API_KEY),USE_SQLITE_FALLBACK=true"

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n❌ Cloud Run Deployment Failed." -ForegroundColor Red
    Write-Host "💡 If billing is not linked, link billing at: https://console.cloud.google.com/billing/linkedaccount?project=$ProjectId" -ForegroundColor Yellow
    exit $LASTEXITCODE
}

Write-Host "`n✅ Bandhu Successfully Deployed to Google Cloud Run!" -ForegroundColor Green
