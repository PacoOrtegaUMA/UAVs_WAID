
import os
import glob
import time
from datetime import datetime
from ultralytics import YOLO
import random
import csv
import math

# Parámetros
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



IMAGES_DIR = os.path.join(PROJECT_ROOT, "WAID", "images", "test")
MODELS_DIR = os.path.join(PROJECT_ROOT, "ModelsWAID")
LOGS_DIR   = os.path.join(PROJECT_ROOT, "logs", DEVICE, "Inferencia")

os.makedirs(LOGS_DIR, exist_ok=True)



IMG_EXT = ".jpg"
MODELOS = ["11n_1", "11s_1", "11m_1", "11l_1", "11x_1",
           "11n_2", "11s_2", "11m_2", "11l_2", "11x_2",
           "11n_3", "11s_3", "11m_3", "11l_3", "11x_3",
           "11n_4", "11s_4", "11m_4", "11l_4", "11x_4"]

MAX_IMGS = 100              # número máximo de imágenes (None = todas)
ALEATORIO = False           # True -> muestra aleatoria; False -> primeras N ordenadas
GUARDAR_DETALLE = True      # tiempos por imagen
GUARDAR_RESUMEN = True      # resumen por modelo
PAUSA_ENTRE_MODELOS = 6    # segundos



def cargar_imagenes(images_dir=IMAGES_DIR, img_ext=IMG_EXT,
                    max_imgs=None, aleatorio=False):
    image_files = glob.glob(os.path.join(images_dir, "*" + img_ext))
    image_files = sorted(image_files)

    if max_imgs is not None and max_imgs < len(image_files):
        if aleatorio:
            image_files = random.sample(image_files, k=max_imgs)
        else:
            image_files = image_files[:max_imgs]

    return image_files


def medir_tiempo_inferencia_modelo(model, image_files):
    tiempos = []

    for idx, img_path in enumerate(image_files, start=1):
        nombre = os.path.basename(img_path)
        print(f"[{idx}/{len(image_files)}] {nombre}", end="\r")

        t0 = time.perf_counter()
        _ = model.predict(img_path, verbose=False, device=TRAIN_DEVICE)
        t1 = time.perf_counter()

        tiempos.append((nombre, t1 - t0))

    print()
    return tiempos


def resumen_tiempos(tiempos):
    if not tiempos:
        return 0.0, 0.0, 0.0, 0.0
    vals = [t for _, t in tiempos]
    mean_t = sum(vals) / len(vals)
    var_t = sum((v - mean_t) ** 2 for v in vals) / len(vals)
    std_t = math.sqrt(var_t)
    min_t = min(vals)
    max_t = max(vals)
    return mean_t, std_t, min_t, max_t


def main():
    image_files = cargar_imagenes(
        max_imgs=MAX_IMGS,
        aleatorio=ALEATORIO
    )
    print(f"Imágenes a procesar: {len(image_files)}")

    # CSV detalle por imagen
    if GUARDAR_DETALLE:
        csv_detalle = os.path.join(LOGS_DIR, "Predic_t_detalle.csv")
        escribir_cabecera_det = not os.path.exists(csv_detalle)
        f_det = open(csv_detalle, "a", newline="", encoding="utf-8")
        w_det = csv.writer(f_det)
        if escribir_cabecera_det:
            w_det.writerow(["Modelo", "Imagen", "Tiempo_s"])
        f_det.flush()

    # CSV resumen por modelo
    if GUARDAR_RESUMEN:
        csv_resumen = os.path.join(LOGS_DIR, "Predic_t_resumen.csv")
        escribir_cabecera_res = not os.path.exists(csv_resumen)
        f_res = open(csv_resumen, "a", newline="", encoding="utf-8")
        w_res = csv.writer(f_res)
        if escribir_cabecera_res:
            w_res.writerow([
                "Modelo", "Inicio", "Fin",
                "Tiempo_medio_s", "Desv_tipica_s",
                "Tiempo_min_s", "Tiempo_max_s"
            ])
        f_res.flush()

    for modelo in MODELOS:
        print(f"\n=== Evaluando modelo {modelo} ===")
        model_path = os.path.join(MODELS_DIR, f"WAID_{modelo}.pt")
        print(f"Cargando: {model_path}")
        model = YOLO(model_path)

        inicio_dt = datetime.now()
        inicio_str = inicio_dt.isoformat(sep=" ", timespec="seconds")

        tiempos = medir_tiempo_inferencia_modelo(model, image_files)

        fin_dt = datetime.now()
        fin_str = fin_dt.isoformat(sep=" ", timespec="seconds")

        mean_t, std_t, min_t, max_t = resumen_tiempos(tiempos)

        print(f"Modelo {modelo}:")
        print(f"  Tiempo medio: {mean_t:.4f} s/imagen")
        print(f"  Desv. típica: {std_t:.6f} s")
        print(f"  Mínimo: {min_t:.4f} s")
        print(f"  Máximo: {max_t:.4f} s")

        if GUARDAR_DETALLE:
            for nombre_img, t in tiempos:
                w_det.writerow([modelo, nombre_img, t])
            f_det.flush()  # fuerza escritura de detalle

        if GUARDAR_RESUMEN:
            w_res.writerow([
                modelo, inicio_str, fin_str,
                mean_t, std_t, min_t, max_t
            ])
            f_res.flush()  # fuerza escritura de resumen

        print(f"Esperando {PAUSA_ENTRE_MODELOS} segundos antes del siguiente modelo...")
        time.sleep(PAUSA_ENTRE_MODELOS)

    if GUARDAR_DETALLE:
        f_det.close()
    if GUARDAR_RESUMEN:
        f_res.close()
    print("\nTiempos guardados en CSV.")


if __name__ == "__main__":
    main()

