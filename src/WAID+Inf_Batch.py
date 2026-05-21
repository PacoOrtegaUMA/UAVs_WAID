import os
import glob
import time
import csv
import socket
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from datetime import datetime


# -------------------------------------------------------
# DEVICE
# -------------------------------------------------------

DEVICE = "Jetson"

if DEVICE == "Isolda":
    USE_GPU = True
    GPU_ID = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    TRAIN_DEVICE = 0   # índice relativo tras CUDA_VISIBLE_DEVICES
    print("Using GPU:", GPU_ID)
elif DEVICE="Jetson":
    TRAIN_DEVICE = 0
    print("Host:", DEVICE)
    print("CUDA disponible:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
else:
    TRAIN_DEVICE = "cpu"
    print("Using CPU")





# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

IMAGES_DIR = os.path.join(PROJECT_ROOT, "WAIDplus", "images", "test")
IMG_EXT = ".jpg"

WAIDPLUS_RUNS_DIR = os.path.join(PROJECT_ROOT, "runs", "waidplus")

LOG_DIR = os.path.join(PROJECT_ROOT, "logs", DEVICE)
os.makedirs(LOG_DIR, exist_ok=True)

CSV_PATH = os.path.join(LOG_DIR, "WAIDplus_batch_inference_times.csv")

MODELOS = ["11n4"]
LEVELS  = [1, 2, 3, 4]
LEVELS  = [3]

# Formatos a comparar
FORMATS = ["pt", "engine"]

# Batch sizes
BATCH_SIZES = [1, 2, 4, 8, 16,32]

MAX_IMGS = 386
IMG_SIZE = 640


# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------

def cargar_imagenes(images_dir, img_ext, max_imgs=None):
    files = sorted(glob.glob(os.path.join(images_dir, "*" + img_ext)))
    if max_imgs is not None:
        files = files[:max_imgs]
    return files


def precargar_tensores(image_files, img_size, device):
    """Carga todas las imágenes como tensor (N, 3, H, W) ya en el device."""
    torch_device = torch.device(f"cuda:{device}" if isinstance(device, int) else device)
    tensors = []
    total = len(image_files)
    for idx, p in enumerate(image_files, 1):
        print(f"  Precargando {idx}/{total}", end="\r")
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.resize(img, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        tensors.append(t)

    all_tensors = torch.stack(tensors).to(torch_device)
    print(f"\nTensores precargados en {torch_device}: {all_tensors.shape}")
    return all_tensors


def find_last_best_pt(runs_dir, modelo, level):
    """Busca el último bloque entrenado y devuelve la ruta al best.pt."""
    last_block = None
    for block_id in range(1, 101):
        run_name = f"waidplus_{modelo}_L{level}_B{block_id}"
        weights_path = os.path.join(runs_dir, run_name, "weights", "best.pt")
        if os.path.exists(weights_path):
            last_block = block_id

    if last_block is None:
        return None, None

    run_name = f"waidplus_{modelo}_L{level}_B{last_block}"
    return os.path.join(runs_dir, run_name, "weights", "best.pt"), last_block


def load_engine_if_exists(pt_path):
    """Carga el engine SOLO si ya existe (no lo genera)."""
    engine_path = pt_path.replace(".pt", ".engine")
    if not os.path.exists(engine_path):
        print(f"  [SKIP engine] No existe: {engine_path}")
        print("  Genera el engine antes con GenerarEngine.py")
        return None
    print(f"  Cargando engine ya compilado: {engine_path}")
    return engine_path


def medir_batch_inference_pt(model, all_tensors, batch_size):
    """Inferencia con modelo PyTorch usando model.model() directamente."""
    n = all_tensors.shape[0]
    batches = [all_tensors[i:i+batch_size] for i in range(0, n, batch_size)]
    num_batches = len(batches)

    with torch.no_grad():
        _ = model.model(batches[0])
    torch.cuda.synchronize()

    t_start = time.perf_counter()
    with torch.no_grad():
        for batch in batches:
            _ = model.model(batch)
    torch.cuda.synchronize()
    t_end = time.perf_counter()

    total_time = t_end - t_start
    return total_time, total_time / n, num_batches


def medir_batch_inference_engine(model, image_files, batch_size, img_size, device):
    """Inferencia con engine TensorRT usando model.predict() en lotes."""
    n = len(image_files)
    batches = [image_files[i:i+batch_size] for i in range(0, n, batch_size)]
    num_batches = len(batches)

    # Warmup
    _ = model.predict(batches[0], verbose=False, device=device, imgsz=img_size)
    torch.cuda.synchronize()

    t_start = time.perf_counter()
    for batch in batches:
        _ = model.predict(batch, verbose=False, device=device, imgsz=img_size)
    torch.cuda.synchronize()
    t_end = time.perf_counter()

    total_time = t_end - t_start
    return total_time, total_time / n, num_batches


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():
    image_files = cargar_imagenes(IMAGES_DIR, IMG_EXT, MAX_IMGS)
    print(f"Imágenes encontradas: {len(image_files)}")

    print("\nPrecargando tensores FP32 en GPU (para PyTorch)...")
    tensors_fp32 = precargar_tensores(image_files, IMG_SIZE, TRAIN_DEVICE)
    n_imgs = tensors_fp32.shape[0]

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f, delimiter=",")
        w.writerow([
            "Modelo", "Formato", "Level", "Ultimo_bloque", "Batch_size",
            "Num_imgs", "Num_batches",
            "Inicio", "Fin",
            "Tiempo_total_s", "Tiempo_medio_img_s", "imgs_por_segundo"
        ])

        for modelo in MODELOS:
            for level in LEVELS:
                pt_path, last_block = find_last_best_pt(WAIDPLUS_RUNS_DIR, modelo, level)

                if pt_path is None:
                    print(f"[SKIP] No se encontró bloque para {modelo} L{level}")
                    continue

                print(f"\n=========================================")
                print(f"=== Modelo: {modelo} L{level} (B{last_block}) ===")
                print(f"=========================================")
                print(f"    {pt_path}")

                for fmt in FORMATS:
                    print(f"\n--- Formato: {fmt} ---")

                    runner = None
                    try:
                        if fmt == "pt":
                            model = YOLO(pt_path)
                            model.to(f"cuda:{TRAIN_DEVICE}")
                            model.model.eval()
                            runner = model

                        elif fmt == "engine":
                            engine_path = load_engine_if_exists(pt_path)
                            if engine_path is None:
                                continue   # salta si no hay engine ya compilado
                            runner = YOLO(engine_path)

                    except Exception as e:
                        print(f"  No se pudo preparar {fmt}: {e}")
                        continue

                    for batch_size in BATCH_SIZES:
                        print(f"  Batch {batch_size:4d}", end=" ... ", flush=True)
                        
                        inicio = datetime.now()
                        
                        try:
                            if fmt == "pt":
                                total_t, mean_t, n_batches = medir_batch_inference_pt(
                                    runner, tensors_fp32, batch_size
                                )
                            else:
                                total_t, mean_t, n_batches = medir_batch_inference_engine(
                                    runner, image_files, batch_size, IMG_SIZE, TRAIN_DEVICE
                                )

                            fin = datetime.now()
                            imgs_per_sec = n_imgs / total_t
                            print(f"{mean_t*1000:.2f} ms/img  |  {imgs_per_sec:.1f} imgs/s")

                            w.writerow([
                                modelo, fmt, level, last_block, batch_size,
                                n_imgs, n_batches,
                                inicio.isoformat(sep=" ", timespec="seconds"),
                                fin.isoformat(sep=" ", timespec="seconds"),
                                round(total_t, 4),
                                round(mean_t, 6),
                                round(imgs_per_sec, 2)
                            ])
                            f.flush()

                        except torch.cuda.OutOfMemoryError:
                            fin = datetime.now()
                            print("OOM")
                            torch.cuda.empty_cache()
                            w.writerow([modelo, fmt, level, last_block, batch_size,
                                        n_imgs, "",
                                        inicio.isoformat(sep=" ", timespec="seconds"),
                                        fin.isoformat(sep=" ", timespec="seconds"),
                                        "", "", "OOM"])
                            f.flush()

                        except Exception as e:
                            fin = datetime.now()
                            print(f"ERROR: {e}")
                            w.writerow([modelo, fmt, level, last_block, batch_size,
                                        n_imgs, "",
                                        inicio.isoformat(sep=" ", timespec="seconds"),
                                        fin.isoformat(sep=" ", timespec="seconds"),
                                        "", "", f"ERROR: {e}"])
                            f.flush()
                        print("Dormido 10s")
                        time.sleep(10)
                    del runner
                    torch.cuda.empty_cache()
                    

    print(f"\nResultados guardados en: {CSV_PATH}")


if __name__ == "__main__":
    main()