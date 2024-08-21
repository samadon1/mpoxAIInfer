from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import uuid
import cv2
from inference import get_model
import supervision as sv
import numpy as np
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)
SAVE_DIR = "annotated_images"
os.makedirs(SAVE_DIR, exist_ok=True)
app.mount("/annotated_images", StaticFiles(directory=SAVE_DIR), name="annotated_images")
model = get_model(model_id="monkeypox-project-cnn/3", api_key="g4taHRQJRP1us1B2dBfO")
@app.post("/infer/")
async def infer_image(file: UploadFile = File(...)):
    image_data = await file.read()
    image_np = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    results = model.infer(image, confidence=0.3, iou_threshold=0.5)[0]
    detections = sv.Detections.from_inference(results)
    confidence_threshold = 0.5
    valid_detections = detections.confidence > confidence_threshold
    bounding_box_annotator = sv.BoxAnnotator(thickness=2, color=sv.Color.YELLOW)
    annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
    image_filename = f"{uuid.uuid4()}.jpg"
    image_path = os.path.join(SAVE_DIR, image_filename)
    cv2.imwrite(image_path, annotated_image)
    image_url = f"http://127.0.0.1:8000/{SAVE_DIR}/{image_filename}"
    status = ''
    symptoms = "Rash, fever, sore throat, headache, muscle aches, back pain, low energy, swollen lymph nodes"
    contagiousness = "Yes"
    severity = "Low to moderate"
    # status = "Monkey Pox" if np.any(valid_detections) else "Clear"
    if np.any(valid_detections):
        status = "Monkey Pox"
    else:
        status = "Clear"
    return JSONResponse(content={"image_url": image_url, "status": status, 
                                 "symptoms": symptoms, "contagiousness": contagiousness, 
                                 "severity": severity})
