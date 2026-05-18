from ultralytics import YOLO
import os, glob, yaml, random, csv
from datetime import datetime

# =========================
# CONFIG GLOBAL
# =========================

USE_GPU = True
GPU_ID = "2"
if USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    DEVICE = 2
    print("Using GPU:", GPU_ID)
else:
    DEVICE = "cpu"
    print("Using CPU")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
TIME_CSV = os.path.join(LOG_DIR, "waidplus_blocks_times.csv")

# Dataset WAIDPlus
BASE_YAML = os.path.join(BASE_DIR, "waidplus.yaml")

# Modelos WAID de partida
MODELS_WAID = [
    os.path.join(PROJECT_ROOT, "ModelsWAID", "WAID_11n_4.pt"),
    # os.path.join(PROJECT_ROOT, "ModelsWAID", "WAID_11s_4.pt"),
    # os.path.join(PROJECT_ROOT, "ModelsWAID", "WAID_11m_4.pt"),
    # os.path.join(PROJECT_ROOT, "ModelsWAID", "WAID_11l_4.pt"),
]

# Niveles de reentreno incremental
LEVELS_WAIDPLUS = [1, 2, 3, 4]

BLOCK_YAML_DIR = os.path.join(BASE_DIR, "blocks_yaml_waidplus")
os.makedirs(BLOCK_YAML_DIR, exist_ok=True)

BATCH_SIZE = 2
WORKERS = 2
MAX_BLOCKS = 100
BLOCK_SIZE = 2   # solo referencia

# Clases nuevas y antiguas (orden IMPORTA)
NEW_CLASSES_ORDER = [6, 7, 8, 9]      # crocodile, elephant, deer, horse
NEW_CLASSES_SET = set(NEW_CLASSES_ORDER)
OLD_CLASSES_ORDER = [0, 1, 2, 3, 4, 5]

# =========================
# HELPERS
# =========================

def load_time_rows():
    rows = {}
    if os.path.exists(TIME_CSV):
        with open(TIME_CSV, "r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows[r["RunId"]] = r
    return rows

def save_time_rows(rows):
    cols_set = set()
    for r in rows.values():
        cols_set.update(k for k in r.keys() if k != "RunId")

    block_cols = sorted(
        [c for c in cols_set if c.startswith("B")],
        key=lambda x: (int(x[1:].split("_")[0]), 0 if x.endswith("_Ini") else 1)
    )

    fieldnames = ["RunId"] + block_cols

    with open(TIME_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run_id, r in rows.items():
            row = {fn: "" for fn in fieldnames}
            row["RunId"] = run_id
            for k, v in r.items():
                if k in row:
                    row[k] = v
            writer.writerow(row)

def load_base_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_yaml(data, path):
    with open(path, "w") as f:
        yaml.safe_dump(data, f)

def get_train_images_from_yaml(cfg):
    root = cfg.get("path", "")
    train_spec = cfg["train"]

    if isinstance(train_spec, str) and train_spec.lower().endswith(".txt"):
        with open(train_spec, "r") as f:
            files = [line.strip() for line in f if line.strip()]
        return sorted(files)

    train_path = os.path.join(root, train_spec)
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(train_path, ext)))
    return sorted(files)

def create_block_yaml(base_cfg, block_images, tag):
    block_txt = os.path.join(BLOCK_YAML_DIR, f"{tag}_train.txt")
    with open(block_txt, "w") as f:
        for img in block_images:
            f.write(img + "\n")

    cfg_block = base_cfg.copy()
    cfg_block["train"] = block_txt

    block_yaml = os.path.join(BLOCK_YAML_DIR, f"{tag}.yaml")
    save_yaml(cfg_block, block_yaml)
    return block_yaml

def get_label_path_from_img(img_path):
    # Ajusta si tu estructura es distinta
    label_path = img_path.replace("/images/", "/labels/")
    label_path = os.path.splitext(label_path)[0] + ".txt"
    return label_path

def image_has_any_new_class(img_path, new_classes=NEW_CLASSES_SET):
    label_path = get_label_path_from_img(img_path)
    if not os.path.exists(label_path):
        return False
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cls_id = int(line.split()[0])
            if cls_id in new_classes:
                return True
    return False

def split_new_old_images(train_images, new_classes=NEW_CLASSES_SET):
    new_imgs = []
    old_imgs = []
    for img in train_images:
        if image_has_any_new_class(img, new_classes):
            new_imgs.append(img)
        else:
            old_imgs.append(img)
    return new_imgs, old_imgs

def get_main_class_from_label(img_path):
    """Devuelve la primera clase del txt (para agrupar)."""
    label_path = get_label_path_from_img(img_path)
    if not os.path.exists(label_path):
        return None
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cls_id = int(line.split()[0])
            return cls_id
    return None

def group_images_by_class(imgs, allowed_classes=None):
    """dict: clase -> lista de imágenes de esa clase."""
    groups = {}
    for img in imgs:
        cls_id = get_main_class_from_label(img)
        if cls_id is None:
            continue
        if allowed_classes is not None and cls_id not in allowed_classes:
            continue
        if cls_id not in groups:
            groups[cls_id] = []
        groups[cls_id].append(img)
    return groups

# =========================
# ENTRENAMIENTO INCREMENTAL WAIDPLUS
# =========================

def train_waidplus_incremental(model_path, level_inc):
    # Hiperparámetros por nivel de reentreno incremental
    if level_inc == 1:
        freeze = 20
        lr0 = 5e-4
        epochs = 10
    elif level_inc == 2:
        freeze = 10
        lr0 = 5e-4
        epochs = 30
    elif level_inc == 3:
        freeze = 5
        lr0 = 1e-3
        epochs = 40
    elif level_inc == 4:
        freeze = 0
        lr0 = 1e-3
        epochs = 50
    else:
        raise ValueError("Nivel incremental inválido")

    base_cfg = load_base_yaml(BASE_YAML)

    train_images = get_train_images_from_yaml(base_cfg)
    print("Total imágenes train WAIDPlus:", len(train_images))

    # 1) Barajar todo el train
    random.shuffle(train_images)

    # 2) Dividir en imágenes con clases nuevas vs solo antiguas
    new_class_imgs, old_class_imgs = split_new_old_images(train_images)
    print("Imágenes con clases nuevas (6,7,8,9):", len(new_class_imgs))
    print("Imágenes solo clases antiguas:", len(old_class_imgs))

    # 3) Agrupar nuevas y antiguas por clase
    new_by_class = group_images_by_class(new_class_imgs, allowed_classes=NEW_CLASSES_SET)
    print("Distribución de imágenes NUEVAS por clase:")
    for c in sorted(new_by_class.keys()):
        print(f"  clase {c}: {len(new_by_class[c])} imágenes")

    old_by_class = group_images_by_class(old_class_imgs, allowed_classes=set(OLD_CLASSES_ORDER))
    print("Distribución de imágenes ANTIGUAS por clase:")
    for c in sorted(old_by_class.keys()):
        print(f"  clase {c}: {len(old_by_class[c])} imágenes")

    # índices por clase
    new_class_idx = {c: 0 for c in NEW_CLASSES_ORDER}
    old_class_idx = {c: 0 for c in OLD_CLASSES_ORDER}

    # máximo de bloques por nuevas (rotando 6,7,8,9,...)
    max_blocks_by_new = 0
    tmp_new_idx = new_class_idx.copy()
    while True:
        avanzó = False
        for c in NEW_CLASSES_ORDER:
            if c not in new_by_class:
                continue
            if tmp_new_idx[c] < len(new_by_class[c]):
                tmp_new_idx[c] += 1
                max_blocks_by_new += 1
                avanzó = True
                if max_blocks_by_new >= MAX_BLOCKS:
                    break
        if not avanzó or max_blocks_by_new >= MAX_BLOCKS:
            break

    # máximo de bloques por antiguas (rotando 0..5)
    max_blocks_by_old = 0
    tmp_old_idx = old_class_idx.copy()
    while True:
        avanzó = False
        for c in OLD_CLASSES_ORDER:
            if c not in old_by_class:
                continue
            if tmp_old_idx[c] < len(old_by_class[c]):
                tmp_old_idx[c] += 1
                max_blocks_by_old += 1
                avanzó = True
                if max_blocks_by_old >= MAX_BLOCKS:
                    break
        if not avanzó or max_blocks_by_old >= MAX_BLOCKS:
            break

    num_blocks = min(MAX_BLOCKS, max_blocks_by_new, max_blocks_by_old)

    base_name = os.path.basename(model_path)   # p.ej. WAID_11n_4.pt
    tag_model = base_name.replace(".pt", "")   # WAID_11n_4
    tag_model = tag_model.replace("WAID_", "").replace("_", "")  # 11n4

    print("\n====================================")
    print("Modelo base WAID:", base_name)
    print("Nivel incremental WAIDPlus:", level_inc)
    print("Bloques posibles (por datos):", num_blocks)
    print("====================================")

    last_weights = model_path

    for block_id in range(1, num_blocks + 1):
        # === Imagen NUEVA: clases 6,7,8,9,6,7,... ===
        new_img = None
        start_new_cls = NEW_CLASSES_ORDER[(block_id - 1) % len(NEW_CLASSES_ORDER)]
        for offset in range(len(NEW_CLASSES_ORDER)):
            cls_to_try = NEW_CLASSES_ORDER[(NEW_CLASSES_ORDER.index(start_new_cls) + offset) % len(NEW_CLASSES_ORDER)]
            if cls_to_try not in new_by_class:
                continue
            idx_n = new_class_idx[cls_to_try]
            if idx_n < len(new_by_class[cls_to_try]):
                new_img = new_by_class[cls_to_try][idx_n]
                new_class_idx[cls_to_try] += 1
                break

        if new_img is None:
            print("No quedan imágenes NUEVAS disponibles en ninguna clase")
            break

        # === Imagen ANTIGUA: clases 0,1,2,3,4,5,0,1,... ===
        old_img = None
        start_old_cls = OLD_CLASSES_ORDER[(block_id - 1) % len(OLD_CLASSES_ORDER)]
        for offset in range(len(OLD_CLASSES_ORDER)):
            cls_to_try = OLD_CLASSES_ORDER[(OLD_CLASSES_ORDER.index(start_old_cls) + offset) % len(OLD_CLASSES_ORDER)]
            if cls_to_try not in old_by_class:
                continue
            idx_o = old_class_idx[cls_to_try]
            if idx_o < len(old_by_class[cls_to_try]):
                old_img = old_by_class[cls_to_try][idx_o]
                old_class_idx[cls_to_try] += 1
                break

        if old_img is None:
            print("No quedan imágenes ANTIGUAS disponibles en ninguna clase")
            break

        # Dataset acumulativo: todas las nuevas y antiguas usadas hasta ahora
        used_new_flat = []
        for c in NEW_CLASSES_ORDER:
            if c not in new_by_class:
                continue
            used_new_flat.extend(new_by_class[c][:new_class_idx[c]])

        used_old_flat = []
        for c in OLD_CLASSES_ORDER:
            if c not in old_by_class:
                continue
            used_old_flat.extend(old_by_class[c][:old_class_idx[c]])

        block_imgs = used_new_flat + used_old_flat

        print(f"\n=== BLOCK {block_id}/{num_blocks} ===")
        print("Num imágenes acumuladas en bloque:", len(block_imgs))
        print("  Nuevas:", len(used_new_flat), "Antiguas:", len(used_old_flat))

        RunId = f"waidplus_{tag_model}_L{level_inc}"
        TagRun = f"{RunId}_B{block_id}"
        block_yaml = create_block_yaml(base_cfg, block_imgs, TagRun)
        run_name = TagRun

        model = YOLO(last_weights)

        Ini = datetime.now()

        model.train(
            data=block_yaml,
            epochs=epochs,
            imgsz=640,
            batch=BATCH_SIZE,
            workers=WORKERS,
            device=DEVICE,
            lr0=lr0,
            freeze=freeze,
            amp=False,
            val=False,
            save=True,
            plots=False,
            seed=42,
            project=os.path.join(PROJECT_ROOT, "runs","waidplus"),
            name=run_name,
            exist_ok=True,
        )

        Fin = datetime.now()
        rows = load_time_rows()
        if RunId not in rows:
            rows[RunId] = {"RunId": RunId}
        rows[RunId][f"B{block_id}_Ini"] = Ini.isoformat(sep=" ")
        rows[RunId][f"B{block_id}_Fin"] = Fin.isoformat(sep=" ")
        save_time_rows(rows)

        last_weights = os.path.join(
            PROJECT_ROOT, "runs", run_name, "weights", "best.pt"
        )

    print("\nFin reentreno WAIDPlus para", base_name, "nivel", level_inc)
    print("Últimos pesos en:", last_weights)

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    for model_path in MODELS_WAID:
        if not os.path.exists(model_path):
            print("WARNING: no existe", model_path)
            continue
        for level_inc in LEVELS_WAIDPLUS:
            train_waidplus_incremental(model_path, level_inc)
