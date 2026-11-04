
from ultralytics import YOLO
from pathlib import Path
import csv

MODEL_PATH = "yolo11n.pt"
IMAGE_DIR = Path("dataset/images/val2017")

IMAGES = [
    "000000021167.jpg",
    "000000518326.jpg",
    "000000013774.jpg",
    "000000377635.jpg",
    "000000511647.jpg",
    "000000158744.jpg",
    "000000317024.jpg",
    "000000574297.jpg",
    "000000091654.jpg",
    "000000403565.jpg",
]

ALLOWED_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus",
    "truck", "dog", "cat", "bird", "bottle",
    "cup", "chair", "laptop", "book", "backpack"
}

model = YOLO(MODEL_PATH)

rows = []

for filename in IMAGES:
    image_path = IMAGE_DIR / filename

    results = model.predict(
        source=str(image_path),
        conf=0.25,
        verbose=False
    )

    detections = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            if class_name in ALLOWED_CLASSES:
                detections.append((class_name, confidence))

    classes = sorted(set(name for name, _ in detections))

    mean_confidence = (
        sum(conf for _, conf in detections) / len(detections)
        if detections else 0
    )

    rows.append({
        "imagem": filename,
        "total_deteccoes": len(detections),
        "classes": ", ".join(classes) if classes else "Nenhuma",
        "confianca_media": round(mean_confidence, 4)
    })

    print(
        f"{filename} | "
        f"Detecções: {len(detections)} | "
        f"Classes: {', '.join(classes) if classes else 'Nenhuma'} | "
        f"Confiança média: {mean_confidence:.4f}"
    )

with open("tests/validation_results.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "imagem",
            "total_deteccoes",
            "classes",
            "confianca_media"
        ]
    )

    writer.writeheader()
    writer.writerows(rows)

print("\nValidação concluída!")
print("Resultados salvos em tests/validation_results.csv")
