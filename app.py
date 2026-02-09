from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from azure.storage.blob import BlobServiceClient
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import io
import os

app = FastAPI()

# Allow your React app to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
transform = None
ASL_CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
               'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

class ASLEfficientNet(nn.Module):
    """EfficientNet-B3 - matches your uploaded model"""
    def __init__(self, num_classes=29):
        super(ASLEfficientNet, self).__init__()
        
        self.model = models.efficientnet_b3(weights=None)
        
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)


@app.on_event("startup")
async def load_model():
    global model, transform
    
    print("Downloading model from Azure...")
    
    # Get connection string from environment variable
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    
    # Download model
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(
        container="models",
        blob="deep_model.pth"
    )
    
    # Save to temp file
    with open("/tmp/model.pth", "wb") as f:
        download_stream = blob_client.download_blob()
        f.write(download_stream.readall())
    
    print("Loading model...")
    
    # Load checkpoint
    checkpoint = torch.load("/tmp/model.pth", map_location="cpu")
    
    # Initialize model
    model = ASLEfficientNet(num_classes=29)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Set up preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print("Model loaded successfully!")


@app.get("/")
def root():
    return {"message": "ASL API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Preprocess
        input_tensor = transform(image).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_idx = probabilities.max(1)
            
            # Top 5
            top5_prob, top5_idx = probabilities.topk(5, dim=1)
        
        # Convert to letter
        predicted_class = predicted_idx.item()
        predicted_letter = ASL_CLASSES[predicted_class]
        
        return {
            "predicted_class": predicted_class,
            "predicted_letter": predicted_letter,
            "confidence": confidence.item(),
            "top5_predictions": [
                {
                    "class": int(top5_idx[0][i]),
                    "letter": ASL_CLASSES[int(top5_idx[0][i])],
                    "confidence": float(top5_prob[0][i])
                }
                for i in range(5)
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))