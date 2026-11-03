
from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI(
    title="ObjectDetect API",
    description="API de detecção de objetos utilizando YOLO",
    version="1.0.0"
)

# Modelo pré-treinado
model = YOLO("yolo11n.pt")

# Classes selecionadas para o projeto
ALLOWED_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus",
    "truck", "dog", "cat", "bird", "bottle",
    "cup", "chair", "laptop", "book", "backpack"
}


@app.get("/")
def home():
    return {"message": "ObjectDetect API funcionando"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Formato inválido. Envie JPG ou PNG."
        )

    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Não foi possível processar a imagem."
        )

    results = model(image)

    detections = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            if class_name not in ALLOWED_CLASSES:
                continue

            coordinates = box.xyxy[0].tolist()

            detections.append({
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "box": {
                    "x1": round(coordinates[0], 2),
                    "y1": round(coordinates[1], 2),
                    "x2": round(coordinates[2], 2),
                    "y2": round(coordinates[3], 2)
                }
            })

    return {
        "filename": file.filename,
        "total_detections": len(detections),
        "detections": detections
    }
