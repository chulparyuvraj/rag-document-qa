#!/bin/bash
# ─────────────────────────────────────────────────────────────
# azure-deploy.sh
# Deploys the RAG FastAPI app to Azure App Service.
#
# Prerequisites:
#   - Azure CLI installed & logged in (az login)
#   - Docker installed locally
#   - .env file configured
#
# Usage:
#   chmod +x deploy/azure-deploy.sh
#   ./deploy/azure-deploy.sh
# ─────────────────────────────────────────────────────────────

set -e

# ── Config — edit these ───────────────────────────────────────
RESOURCE_GROUP="rag-qa-rg"
LOCATION="eastus"
APP_NAME="rag-document-qa"
ACR_NAME="ragdocumentqaacr"         # Azure Container Registry name
APP_SERVICE_PLAN="rag-qa-plan"
SKU="S2"                            # Standard S2: 2 cores, 3.5GB RAM
# ─────────────────────────────────────────────────────────────

echo "🚀 Starting Azure deployment for $APP_NAME"

# 1. Create resource group
echo "Creating resource group..."
az group create --name $RESOURCE_GROUP --location $LOCATION

# 2. Create Azure Container Registry
echo "Creating Azure Container Registry..."
az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME \
  --sku Basic \
  --admin-enabled true

# 3. Get ACR credentials
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query "passwords[0].value" -o tsv)
ACR_SERVER="${ACR_NAME}.azurecr.io"

# 4. Build and push Docker image
echo "Building Docker image..."
docker build -t $APP_NAME:latest .
docker tag $APP_NAME:latest $ACR_SERVER/$APP_NAME:latest

echo "Pushing to ACR..."
docker login $ACR_SERVER -u $ACR_NAME -p $ACR_PASSWORD
docker push $ACR_SERVER/$APP_NAME:latest

# 5. Create App Service Plan
echo "Creating App Service Plan..."
az appservice plan create \
  --name $APP_SERVICE_PLAN \
  --resource-group $RESOURCE_GROUP \
  --sku $SKU \
  --is-linux

# 6. Create Azure Blob Storage for PDFs
echo "Creating Storage Account..."
STORAGE_ACCOUNT="${APP_NAME//-/}storage"
az storage account create \
  --name $STORAGE_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --sku Standard_LRS

STORAGE_CONN=$(az storage account show-connection-string \
  --name $STORAGE_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --query connectionString -o tsv)

az storage container create \
  --name "research-papers" \
  --connection-string "$STORAGE_CONN"

# 7. Create Web App from container
echo "Creating Web App..."
az webapp create \
  --resource-group $RESOURCE_GROUP \
  --plan $APP_SERVICE_PLAN \
  --name $APP_NAME \
  --deployment-container-image-name $ACR_SERVER/$APP_NAME:latest

# 8. Configure environment variables
echo "Setting environment variables..."
az webapp config appsettings set \
  --resource-group $RESOURCE_GROUP \
  --name $APP_NAME \
  --settings \
    AZURE_STORAGE_CONNECTION_STRING="$STORAGE_CONN" \
    AZURE_CONTAINER_NAME="research-papers" \
    EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2" \
    BASE_LLM="mistralai/Mistral-7B-Instruct-v0.2" \
    WEBSITES_PORT=8000

# 9. Configure container registry auth
az webapp config container set \
  --resource-group $RESOURCE_GROUP \
  --name $APP_NAME \
  --docker-custom-image-name $ACR_SERVER/$APP_NAME:latest \
  --docker-registry-server-url https://$ACR_SERVER \
  --docker-registry-server-user $ACR_NAME \
  --docker-registry-server-password $ACR_PASSWORD

echo ""
echo "✅ Deployment complete!"
echo "   App URL: https://${APP_NAME}.azurewebsites.net"
echo "   Health:  https://${APP_NAME}.azurewebsites.net/health"
echo "   Docs:    https://${APP_NAME}.azurewebsites.net/docs"
