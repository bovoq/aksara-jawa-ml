from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
from PIL import Image
import io
from helper.predict import predict_from_images
from helper.config import get_allowed_origins

app = FastAPI(
    title="Aksara Jawa API",
    description="API for Aksara Jawa",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to Aksara Jawa API"}

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    try:
        images = []
        for file in files:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            images.append(image)

        prediction = predict_from_images(images)
        return {"prediction": prediction}
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
