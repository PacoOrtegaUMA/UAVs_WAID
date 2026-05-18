from ultralytics import YOLO
import os
from pathlib import Path
import numpy as np

# ---------------- CONFIG ----------------

MODEL_PATH = "/home/fozamorano/WAID/ModelsWAID/Model_11s_4.pt"

WAIDPLUS_ROOT = "/home/fozamorano/WAID/WAIDplus"
IMG_DIR   = os.path.join(WAIDPLUS_ROOT, "images", "test")
LABEL_DIR = os.path.join(WAIDPLUS_ROOT, "labels", "test")

USE_GPU = True
DEVICE = 2 if USE_GPU else "cpu"

OLD_CLASS_IDS = [0, 1, 2, 3, 4, 5]
CLASS_NAMES = ["sheep","cattle","seal","camelus","kiang","zebra"]

CONF_THRES = 0.01   # bajo para construir bien la curva PR
IOU_THRESHES = np.linspace(0.5, 0.95, 10)  # para mAP50-95

# ---------------- UTILIDADES ----------------

def load_gt_boxes(label_path):
    """Lista de (cls, x1,y1,x2,y2) con coords normalizadas [0,1]."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            if cls not in OLD_CLASS_IDS:
                continue
            xc, yc, w, h = map(float, parts[1:5])
            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2
            boxes.append((cls, x1, y1, x2, y2))
    return boxes

def iou_xyxy(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter + 1e-9
    return inter / union

def compute_ap(rec, prec):
    """AP como área bajo la curva P-R (interp estilo COCO sencillo)."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap

# ---------------- MAIN ----------------

def main():
    print("Cargando modelo:", MODEL_PATH)
    model = YOLO(MODEL_PATH)

    exts = {".jpg",".jpeg",".png"}
    img_paths = [
        os.path.join(IMG_DIR, f)
        for f in sorted(os.listdir(IMG_DIR))
        if Path(f).suffix.lower() in exts
    ]
    print("Imágenes test WAIDPlus:", len(img_paths))

    # Predicciones
    results = model.predict(
        img_paths,
        imgsz=640,
        device=DEVICE,
        conf=CONF_THRES,
        verbose=False,
    )

    # Para cada clase: lista de predicciones (img_id, conf, bbox) y GTs
    preds = {cid: [] for cid in OLD_CLASS_IDS}
    gts   = {cid: {} for cid in OLD_CLASS_IDS}  # cid -> {img_id: [bboxes]}

    for img_id, (img_path, r) in enumerate(zip(img_paths, results)):
        img_name = os.path.basename(img_path)
        label_path = os.path.join(
            LABEL_DIR, os.path.splitext(img_name)[0] + ".txt"
        )

        # GT
        gt_boxes = load_gt_boxes(label_path)
        for (cls, x1, y1, x2, y2) in gt_boxes:
            gts.setdefault(cls, {})
            gts[cls].setdefault(img_id, [])
            gts[cls][img_id].append((x1, y1, x2, y2))

        # Predicciones
        if r.boxes is not None:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls not in OLD_CLASS_IDS:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxyn[0].tolist()  # normalizado
                preds[cls].append((img_id, conf, (x1, y1, x2, y2)))

    # mAP por clase
    ap50 = {}
    ap5095 = {}

    for cid in OLD_CLASS_IDS:
        # ordenar predicciones por confianza descendente
        cls_preds = sorted(preds[cid], key=lambda x: -x[1])
        if len(cls_preds) == 0:
            ap50[cid] = 0.0
            ap5095[cid] = 0.0
            continue

        # total de GT de esta clase
        npos = sum(len(boxes) for boxes in gts[cid].values()) if cid in gts else 0
        if npos == 0:
            ap50[cid] = 0.0
            ap5095[cid] = 0.0
            continue

        aps = []

        for iou_th in IOU_THRESHES:
            tp = np.zeros(len(cls_preds))
            fp = np.zeros(len(cls_preds))
            matched = {img_id: np.zeros(len(gts[cid].get(img_id, [])))
                       for img_id in gts[cid].keys()}

            for i, (img_id, conf, bbox_pred) in enumerate(cls_preds):
                gt_boxes = gts[cid].get(img_id, [])
                if len(gt_boxes) == 0:
                    fp[i] = 1
                    continue

                ious = np.array([iou_xyxy(bbox_pred, gt) for gt in gt_boxes])
                j = ious.argmax()
                best_iou = ious[j]

                if best_iou >= iou_th and matched[img_id][j] == 0:
                    tp[i] = 1
                    matched[img_id][j] = 1
                else:
                    fp[i] = 1

            fp_cum = np.cumsum(fp)
            tp_cum = np.cumsum(tp)
            rec = tp_cum / npos
            prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

            ap = compute_ap(rec, prec)
            aps.append(ap)

        ap50[cid] = aps[0]                # IoU=0.5
        ap5095[cid] = float(np.mean(aps)) # media 0.5:0.95

    print("\n=== AP por clase (WAID en WAIDPlus, solo viejas) ===")
    for cid in OLD_CLASS_IDS:
        print(f"{cid} {CLASS_NAMES[cid]:8s} -> "
              f"AP50: {ap50[cid]:.4f}  AP50-95: {ap5095[cid]:.4f}")

    mAP50 = float(np.mean(list(ap50.values())))
    mAP5095 = float(np.mean(list(ap5095.values())))
    print("\n=== mAP (media de clases viejas) ===")
    print("mAP50:", round(mAP50, 4))
    print("mAP50-95:", round(mAP5095, 4))

if __name__ == "__main__":
    main()
