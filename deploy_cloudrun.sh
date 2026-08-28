#!/usr/bin/env bash
set -e

PROJECT_ID=${1:-"bandhu-agentic-demo"}
REGION=${2:-"us-central1"}
SERVICE_NAME=${3:-"bandhu-agent"}

echo "🚀 Deploying Bandhu to Google Cloud Run..."
gcloud config set project "$PROJECT_ID"

gcloud run deploy "$SERVICE_NAME" \
    --source . \
    --region "$REGION" \
    --platform managed \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --set-env-vars "GCP_PROJECT_ID=$PROJECT_ID,ENVIRONMENT=production,USE_SQLITE_FALLBACK=false"

echo "✅ Bandhu Successfully Deployed to Google Cloud Run!"
