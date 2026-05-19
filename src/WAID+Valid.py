from ultralytics import YOLO
import os, csv
import numpy as np

DEVICE="Isolda"

# GPU
if DEVICE=="Isolda":
    USE_GPU = True     
    GPU_ID = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    TRAIN_DEVICE = GPU_ID 
    print("Using GPU:", GPU_ID)
else:
    print("Using CPU")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Directorio de logs
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
VAL_LOG_DIR = os.path.join(LOG_DIR, "ValWAID+")
os.makedirs(VAL_LOG_DIR, exist_ok=True)

DATA_YAML  = os.path.join(BASE_DIR, "waidplus.yaml")
IMG_SIZE   = 640
BATCH      = 8

# --- COMPROBACIÓN DE RUNS ---
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")
if not os.path.exists(RUNS_DIR):
    raise FileNotFoundError(f"La carpeta base de resultados no existe: {RUNS_DIR}")

# --- NUEVA CARPETA ESPECÍFICA PARA VALIDACIÓN ---
VAL_ROOT_DIR = os.path.join(RUNS_DIR, "ValWAID+")
os.makedirs(VAL_ROOT_DIR, exist_ok=True)

LEVELS_WAIDPLUS = [1, 2, 3, 4]
MAX_BLOCKS = 100 

def run_val_for_block(level_inc, block_id):
    block_tag = f"L{level_inc}_B{block_id}"
    # Buscamos el modelo en la carpeta original de entrenamiento
    model_dir = os.path.join(RUNS_DIR, "waidplus", f"waidplus_11n4_{block_tag}")
    model_path = os.path.join(model_dir, "weights", "best.pt")

    if not os.path.exists(model_path):
        print(f"[SKIP] No existe best.pt para {block_tag} en {model_path}")
        return

    # El nombre de la carpeta de salida (ej: ValWAID+/Val_L1_B1)
    val_run_name = f"Val_{block_tag}"
    out_csv = os.path.join(VAL_LOG_DIR, f"class_{block_tag}.csv")

    print(f"\n=== VAL {block_tag} ===")
    model = YOLO(model_path)

    metrics = model.val(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=TRAIN_DEVICE, 
        split="test",
        verbose=False,
        project=VAL_ROOT_DIR,  # Guardamos dentro de runs/ValWAID+
        name=val_run_name,     
    )

    # ... (resto del procesamiento de métricas igual)
    p, r, map50_global, map5095_global = metrics.mean_results()
    names = metrics.names
    
    per_class_results = []
    for cid in range(len(names)):
        pc, rc, ap50c, ap5095c = metrics.class_result(cid)
        per_class_results.append((cid, names[cid], ap50c, ap5095c))

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_name", "mAP50", "mAP50-95"])
        for cid, cname, ap50c, ap5095c in per_class_results:
            w.writerow([cid, cname, float(ap50c), float(ap5095c)])
        w.writerow([])
        w.writerow(["GLOBAL", "all", float(map50_global), float(map5095_global)])

    print("Guardado CSV en:", out_csv)
    print("Resultados en:", os.path.join(VAL_ROOT_DIR, val_run_name))
    try:
        # metrics.confusion_matrix.matrix es (nc x nc)
        cm_np = np.array(metrics.confusion_matrix.matrix, dtype=float)

        cm_csv_path = os.path.join(VAL_LOG_DIR, f"confmat_{block_tag}.csv")
        np.savetxt(cm_csv_path, cm_np, delimiter=",")
        print("Guardada matriz de confusión en:", cm_csv_path)
    except Exception as e:
        print("[WARN] No se pudo guardar confmat para", block_tag, "->", e)
        

if __name__ == "__main__":
    for level_inc in LEVELS_WAIDPLUS:
        for block_id in range(1, MAX_BLOCKS+1):
            run_val_for_block(level_inc, block_id)