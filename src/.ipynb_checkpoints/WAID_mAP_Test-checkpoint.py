import os
import cv2
import glob
from ultralytics import YOLO
import csv
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

def calcular_ap50(resultados_globales, fn_globales, clase, intervalo):
    """
    Calcula AP@50 para (clase, intervalo).
    Si no hay GT en ese intervalo para esa clase, devuelve (np.nan, False).
    """
    # Predicciones de esa clase e intervalo
    preds = [r for r in resultados_globales if r["cls"]==clase and r["intervalo"]==intervalo]
    preds = sorted(preds, key=lambda x: -x["conf"])

    # Contar TP en ese intervalo (sirve para saber si hay GT)
    tp_count = sum(1 for r in preds if r["status"]=="TP")
    # Contar FN en ese intervalo
    fn_count = sum(1 for fn in fn_globales.get(clase, []) if fn["intervalo"]==intervalo)

    # Total GT en ese intervalo para esa clase
    gt_total = tp_count + fn_count
    if gt_total == 0:
        return np.nan, False  # no hay GT → AP no definido

    # Acumulado para curva PR
    tp, fp = 0, 0
    precisiones, recalls = [], []
    for p in preds:
        if p["status"] == "TP":
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / gt_total if gt_total > 0 else 0.0
        precisiones.append(precision)
        recalls.append(recall)

    if not recalls:
        # No hubo predicciones, pero sí GT → AP=0 por convención
        return 0.0, True

    # Interpolación (COCO-like): máxima precisión para cada nivel de recall
    recalls = np.array(recalls)
    precisiones = np.array(precisiones)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = np.max(precisiones[recalls >= t]) if np.any(recalls >= t) else 0.0
        ap += p
    ap /= 101.0
    return ap, True

def calcular_map50_por_intervalo_filtrado(resultados_globales, fn_globales, intervalos=[0,1,10,100]):
    """
    Devuelve:
      - mAP@50 por intervalo EXCLUYENDO clases sin GT en ese intervalo
      - mAP@50 por intervalo INCLUYENDO todas las clases (para comparar)
      - AP@50 por clase e intervalo (con NaN cuando no hay GT)
    """
    clases = sorted(set(r["cls"] for r in resultados_globales))
    ap_por_clase_intervalo = {cls: {} for cls in clases}
    map_excl = {}
    map_incl = {}

    for intervalo in range(1, len(intervalos)):
        aps_validos = []
        aps_todos = []
        for cls in clases:
            ap, has_gt = calcular_ap50(resultados_globales, fn_globales, cls, intervalo)
            ap_por_clase_intervalo[cls][intervalo] = ap
            # Para mAP incluyendo todo (NaN -> 0 por convención previa)
            aps_todos.append(0.0 if (ap is np.nan or (isinstance(ap, float) and math.isnan(ap))) else ap)
            # Para mAP excluyendo clases sin GT
            if has_gt:
                aps_validos.append(ap)

        map_incl[intervalo] = float(np.mean(aps_todos)) if aps_todos else 0.0
        map_excl[intervalo] = float(np.mean(aps_validos)) if aps_validos else 0.0

    return map_excl, map_incl, ap_por_clase_intervalo


    
        
def guardar_predicciones_csv(resultados, filename="predicciones.csv"):
    """
    Guarda todas las predicciones (TP/FP) en un CSV.
    """
    campos = ["imagen","cls","conf","status","iou","intervalo","pct","box"]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        for r in resultados:
            fila = {k:r[k] for k in campos}
            writer.writerow(fila)

# ---------------------------
# Guardar FN en CSV
# ---------------------------
def guardar_fn_csv(fn_globales, filename="fn_por_clase.csv"):
    """
    Guarda todos los FN por clase en un CSV.
    """
    campos = ["imagen","cls","intervalo","pct","box"]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        for cls, fns in fn_globales.items():
            for fn in fns:
                fila = {"imagen":fn["imagen"], "cls":cls,
                        "intervalo":fn["intervalo"], "pct":fn["pct"],
                        "box":fn["box"]}
                writer.writerow(fila)
# ---------------------------
# Función: cargar GT
# ---------------------------
def cargar_gt(imagen_base, images_dir="../WAID/images/test/", labels_dir="../WAID/labels/test/", img_ext=".jpg"):
    """
    Carga las anotaciones GT de una imagen en formato YOLO y devuelve
    una lista de diccionarios con cajas en píxeles y su clase.
    """
    img_path = os.path.join(images_dir, imagen_base + img_ext)
    label_path = os.path.join(labels_dir, imagen_base + ".txt")

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {img_path}")
    H, W = img.shape[:2]

    gts = []
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                cls, cx, cy, w, h = map(float, line.split()[:5])
                cls = int(cls)
                x1 = (cx - w/2) * W
                y1 = (cy - h/2) * H
                x2 = (cx + w/2) * W
                y2 = (cy + h/2) * H
                gts.append({"box":[x1,y1,x2,y2], "cls":cls})
    else:
        print("No hay GT para esta imagen.")

    return gts, (W, H)

# ---------------------------
# Función: cargar predicciones
# ---------------------------
def cargar_preds(imagen_base, model, images_dir="../WAID/images/test/", img_ext=".jpg",
                 conf=0.01, iou=0.7, max_det=300):
    """
    Obtiene las predicciones del modelo para una imagen y devuelve
    una lista de diccionarios con cajas en píxeles, clase y confianza.
    """
    img_path = os.path.join(images_dir, imagen_base + img_ext)
    results = model.predict(img_path, conf=conf, iou=iou, max_det=max_det, verbose=False)[0]

    preds = []
    for b in results.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        cls = int(b.cls.item())
        confv = float(b.conf.item())
        preds.append({"box":[x1,y1,x2,y2], "cls":cls, "conf":confv})
    return preds

# ---------------------------
# Función: IoU
# ---------------------------
def iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = max(0, box1[2]-box1[0]) * max(0, box1[3]-box1[1])
    area2 = max(0, box2[2]-box2[0]) * max(0, box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter/union if union>0 else 0

# ---------------------------
# Función: asignar intervalo
# ---------------------------
def asignar_intervalo(pct, intervalos=[0,100]):
    """
    Devuelve el índice del intervalo en el que cae el porcentaje.
    Ejemplo: pct=5 -> devuelve 2 porque está en [1,10).
    """
    for i in range(1, len(intervalos)):
        if intervalos[i-1] <= pct < intervalos[i]:
            return i
    return len(intervalos)

# ---------------------------
# Función: evaluar predicciones con FN e intervalos
# ---------------------------
def evaluar_predicciones_con_fn_intervalos(imagen_base, model, iou_threshold=0.5,
                                           images_dir="../WAID/images/test/", labels_dir="../WAID/labels/test/",
                                           img_ext=".jpg", conf=0.01, iou_nms=0.7, max_det=300,
                                           intervalos=[0,100]):
    gts, (W,H) = cargar_gt(imagen_base, images_dir, labels_dir, img_ext)
    preds = cargar_preds(imagen_base, model, images_dir, img_ext, conf=conf, iou=iou_nms, max_det=max_det)

    img_area = W * H
    gt_available = set(range(len(gts)))
    resultados = []

    for p in sorted(preds, key=lambda x: -x["conf"]):
        best_iou, best_gt = 0, -1
        for gi in gt_available:
            gt = gts[gi]
            if gt["cls"] != p["cls"]:
                continue
            val = iou(p["box"], gt["box"])
            if val > best_iou:
                best_iou, best_gt = val, gi

        if best_iou >= iou_threshold and best_gt != -1:
            # TP → intervalo según área del GT
            gt_box = gts[best_gt]["box"]
            gt_area = max(0,(gt_box[2]-gt_box[0])) * max(0,(gt_box[3]-gt_box[1]))
            pct = (gt_area/img_area)*100 if img_area>0 else 0
            intervalo = asignar_intervalo(pct, intervalos)
            resultados.append({
                "imagen": imagen_base,
                "cls": p["cls"], "conf": p["conf"], "box": p["box"],
                "status": "TP", "iou": best_iou, "intervalo": intervalo, "pct": pct
            })
            gt_available.remove(best_gt)
        else:
            # FP → intervalo según área de la predicción
            pred_box = p["box"]
            pred_area = max(0,(pred_box[2]-pred_box[0])) * max(0,(pred_box[3]-pred_box[1]))
            pct = (pred_area/img_area)*100 if img_area>0 else 0
            intervalo = asignar_intervalo(pct, intervalos)
            resultados.append({
                "imagen": imagen_base,
                "cls": p["cls"], "conf": p["conf"], "box": p["box"],
                "status": "FP", "iou": best_iou, "intervalo": intervalo, "pct": pct
            })

    # FN por clase
    fn_por_clase = {}
    for gi in gt_available:
        gt = gts[gi]
        cls = gt["cls"]
        gt_box = gt["box"]
        gt_area = max(0,(gt_box[2]-gt_box[0])) * max(0,(gt_box[3]-gt_box[1]))
        pct = (gt_area/img_area)*100 if img_area>0 else 0
        intervalo = asignar_intervalo(pct, intervalos)
        fn_por_clase.setdefault(cls, []).append({
            "imagen": imagen_base,
            "box": gt_box, "intervalo": intervalo, "pct": pct
        })

    return resultados, fn_por_clase

# ---------------------------
# Función: procesar dataset completo
# ---------------------------
def procesar_dataset(model, images_dir="../WAID/images/test/", labels_dir="../WAID/labels/test/",
                     img_ext=".jpg", iou_threshold=0.5, conf=0.01, iou_nms=0.7, max_det=300,
                     intervalos=[0,100]):
    """
    Recorre todas las imágenes del dataset y concatena resultados y FN por clase.
    """
    image_files = glob.glob(os.path.join(images_dir, "*" + img_ext))
    resultados_globales = []
    fn_globales = {}

    total_imgs = len(image_files)

    # enumerate da (idx, img_path)
    for idx, img_path in enumerate(image_files, start=1):
        imagen_base = os.path.splitext(os.path.basename(img_path))[0]

        # Esto sobreescribe la misma línea en consola
        print(f"Procesando {idx}/{total_imgs}: {imagen_base}", end="\r")
        
        resultados, fn_por_clase = evaluar_predicciones_con_fn_intervalos(
            imagen_base, model,
            iou_threshold=iou_threshold,
            images_dir=images_dir, labels_dir=labels_dir, img_ext=img_ext,
            conf=conf, iou_nms=iou_nms, max_det=max_det,
            intervalos=intervalos
        )

        resultados_globales.extend(resultados)

        for cls, fns in fn_por_clase.items():
            if cls not in fn_globales:
                fn_globales[cls] = []
            fn_globales[cls].extend(fns)

    return resultados_globales, fn_globales


# ---------------------------
# Ejemplo de uso
# ---------------------------
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
    MODELS_DIR = os.path.join(PROJECT_ROOT, "ModelsWAID")

    # Dispositivo / variante (ajusta esta variable)
    #Device = "isolda"  # por ejemplo

    LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Nombres de CSV
    Name_mAP = "WAID_mAP_intervalo_1.csv"
    Name_AP  = "WAID_AP_clase_intervalo_1.csv"

    Modelos = ["11s_4", "11n_4", "11m_4", "11l_4", "11x_4"]

    IntVal = [0, 0.1, 1, 10, 100]
    confi = 0.1

    for Modelo in Modelos:
        model_path = os.path.join(MODELS_DIR, f"WAID_{Modelo}.pt")
        print(f"\n=== Evaluando modelo {model_path} ===")

        model = YOLO(model_path)

        resultados_por_umbral = {}
        for thr in np.arange(0.5, 1.0, 0.05):
            print(f"\nProcesando dataset con IoU={thr:.2f}")
            resultados_globales, fn_globales = procesar_dataset(
                model,
                intervalos=IntVal,
                iou_threshold=thr,
                conf=confi
            )
            resultados_por_umbral[thr] = {
                "resultados": resultados_globales,
                "fn": fn_globales
            }
            print()
            print(f"Total predicciones procesadas: {len(resultados_globales)}")
            print(f"Total FN clases: {len(fn_globales)}")

        # --- Calcular métricas ---
        Medida = 0.5
        resultados_globales = resultados_por_umbral[Medida]["resultados"]
        fn_globales = resultados_por_umbral[Medida]["fn"]

        map_excl, map_incl, ap_ci = calcular_map50_por_intervalo_filtrado(
            resultados_globales, fn_globales, intervalos=IntVal
        )

        clases = sorted(set(r["cls"] for r in resultados_por_umbral[0.5]["resultados"]))
        ap_ci_50_95 = {cls: {} for cls in clases}
        map_por_intervalo = {}

        for intervalo in range(1, len(IntVal)):
            aps_validos = []
            for cls in clases:
                ap_vals = []
                for thr in resultados_por_umbral.keys():  # 0.50 ... 0.95
                    resultados_globales = resultados_por_umbral[thr]["resultados"]
                    fn_globales = resultados_por_umbral[thr]["fn"]
                    ap, has_gt = calcular_ap50(resultados_globales, fn_globales, cls, intervalo)
                    if has_gt:
                        ap_vals.append(ap)
                if ap_vals:
                    ap_mean = np.mean(ap_vals)
                    ap_ci_50_95[cls][intervalo] = ap_mean
                    aps_validos.append(ap_mean)
                else:
                    ap_ci_50_95[cls][intervalo] = np.nan
            map_por_intervalo[intervalo] = np.mean(aps_validos) if aps_validos else np.nan

        # --- Guardar resultados en CSV (append) ---
        from pandas.errors import EmptyDataError

        # 1. mAP por intervalo
        df_map = pd.DataFrame([
            {
                "Intervalo": intervalo,
                "mAP@50": map_excl.get(intervalo, None),
                "mAP@50-95": map_por_intervalo.get(intervalo, None),
                "Modelo": Modelo
            }
            for intervalo in sorted(set(map_excl) | set(map_por_intervalo))
        ])

        file_map = os.path.join(LOGS_DIR, Name_mAP)
        if os.path.exists(file_map):
            try:
                df_old = pd.read_csv(file_map)
                if "Modelo" in df_old.columns:
                    df_old = df_old[df_old["Modelo"] != Modelo]
                df_map = pd.concat([df_old, df_map], ignore_index=True)
            except EmptyDataError:
                pass

        df_map.to_csv(file_map, index=False)
        print(f"Resultados añadidos a '{file_map}'.")

        # 2. AP por clase e intervalo
        rows = []
        for cls, ints in ap_ci_50_95.items():
            for intervalo, ap95 in ints.items():
                ap50 = ap_ci.get(cls, {}).get(intervalo, np.nan)
                rows.append({
                    "Clase": cls,
                    "Intervalo": intervalo,
                    "AP@50": ap50,
                    "AP@50-95": ap95,
                    "Modelo": Modelo
                })

        df_ap = pd.DataFrame(rows)
        file_ap = os.path.join(LOGS_DIR, Name_AP)

        if os.path.exists(file_ap):
            try:
                df_old = pd.read_csv(file_ap)
                if "Modelo" in df_old.columns:
                    df_old = df_old[df_old["Modelo"] != Modelo]
                df_ap = pd.concat([df_old, df_ap], ignore_index=True)
            except EmptyDataError:
                pass

        df_ap.to_csv(file_ap, index=False)
        print(f"Resultados añadidos a '{file_ap}'.")
