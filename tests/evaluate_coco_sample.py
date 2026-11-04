
from ultralytics import YOLO
from pathlib import Path
from pycocotools.coco import COCO
from collections import defaultdict
import numpy as np

IMAGE_DIR = Path("dataset/images/val2017")
ANNOTATIONS = "dataset/annotations/instances_val2017.json"

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

IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.25

model = YOLO("yolo11n.pt")
coco = COCO(ANNOTATIONS)

# Mapeamento entre nomes COCO e IDs do modelo
model_class_ids = {
    name: class_id
    for class_id, name in model.names.items()
    if name in ALLOWED_CLASSES
}

coco_categories = coco.loadCats(coco.getCatIds())
coco_class_ids = {
    category["name"]: category["id"]
    for category in coco_categories
    if category["name"] in ALLOWED_CLASSES
}


def calculate_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0


total_tp = 0
total_fp = 0
total_fn = 0

class_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

for filename in IMAGES:
    image_id = int(Path(filename).stem)
    image_path = IMAGE_DIR / filename

    # Anotações reais da imagem
    annotation_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=False)
    annotations = coco.loadAnns(annotation_ids)

    ground_truth = []

    for annotation in annotations:
        category = coco.loadCats(annotation["category_id"])[0]["name"]

        if category not in ALLOWED_CLASSES:
            continue

        x, y, width, height = annotation["bbox"]

        ground_truth.append({
            "class": category,
            "box": [x, y, x + width, y + height],
            "matched": False
        })

    # Inferência
    results = model.predict(
        source=str(image_path),
        conf=CONF_THRESHOLD,
        verbose=False
    )

    predictions = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            if class_name not in ALLOWED_CLASSES:
                continue

            predictions.append({
                "class": class_name,
                "box": box.xyxy[0].tolist(),
                "confidence": float(box.conf[0])
            })

    # Maior confiança primeiro
    predictions.sort(key=lambda item: item["confidence"], reverse=True)

    image_tp = 0
    image_fp = 0

    for prediction in predictions:
        best_match = None
        best_iou = 0

        for target in ground_truth:
            if target["matched"] or target["class"] != prediction["class"]:
                continue

            iou = calculate_iou(prediction["box"], target["box"])

            if iou > best_iou:
                best_iou = iou
                best_match = target

        if best_match is not None and best_iou >= IOU_THRESHOLD:
            best_match["matched"] = True
            image_tp += 1
            class_metrics[prediction["class"]]["tp"] += 1
        else:
            image_fp += 1
            class_metrics[prediction["class"]]["fp"] += 1

    image_fn = sum(1 for target in ground_truth if not target["matched"])

    for target in ground_truth:
        if not target["matched"]:
            class_metrics[target["class"]]["fn"] += 1

    total_tp += image_tp
    total_fp += image_fp
    total_fn += image_fn

    print(
        f"{filename}: TP={image_tp}, FP={image_fp}, FN={image_fn}"
    )

precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0
recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0
f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0
)

print("\n===== RESULTADO GERAL =====")
print(f"TP: {total_tp}")
print(f"FP: {total_fp}")
print(f"FN: {total_fn}")
print(f"Precisão: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

print("\n===== RESULTADOS POR CLASSE =====")

for class_name, metrics in sorted(class_metrics.items()):
    print(
        f"{class_name}: "
        f"TP={metrics['tp']} | "
        f"FP={metrics['fp']} | "
        f"FN={metrics['fn']}"
    )
