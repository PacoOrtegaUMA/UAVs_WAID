from ultralytics import YOLO
import os
import csv

# -------------------------------------------------------
# DEVICE
# -------------------------------------------------------

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

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_YAML = os.path.join(BASE_DIR, "waid.yaml")
MODELS_DIR = os.path.join(PROJECT_ROOT, "ModelsWAID")

LOGS_DIR = os.path.join(PROJECT_ROOT, "logs", "Waid_mAP")
os.makedirs(LOGS_DIR, exist_ok=True)

CSV_PATH_50 = os.path.join(LOGS_DIR, "WAID_map50_results.csv")
CSV_PATH_95 = os.path.join(LOGS_DIR, "WAID_map50_95_results.csv")
CSV_HEADER = ["model", "train1", "train2", "train3", "train4"]

CSV_PER_CLASS = os.path.join(LOGS_DIR, "WAID_map_per_class.csv")
PER_CLASS_HEADER = ["base_model", "train_level", "class_id",
                    "class_name", "map50", "map50_95"]

VAL_PROJECT = os.path.join(PROJECT_ROOT, "runs", "val_map50")
os.makedirs(VAL_PROJECT, exist_ok=True)

TAGS = ["11n", "11s", "11m", "11l", "11x"]
LEVELS = [1, 2, 3, 4]


# -------------------------------------------------------
# CSV helpers
# -------------------------------------------------------

def load_existing_rows(csv_path, header):
    rows_dict = {}
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for col in header:
                    if col not in row:
                        row[col] = ""
                model_name = row["model"]
                rows_dict[model_name] = row
    return rows_dict


def save_csv(rows_dict, csv_path, header):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for model_name in sorted(rows_dict.keys()):
            writer.writerow(rows_dict[model_name])


def append_per_class_rows(rows, csv_path, header):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():
    rows_50 = load_existing_rows(CSV_PATH_50, CSV_HEADER)
    rows_95 = load_existing_rows(CSV_PATH_95, CSV_HEADER)

    for tag in TAGS:
        base_name = "yolo%s.pt" % tag
        print("\n====================================")
        print("Base model (row):", base_name)
        print("====================================")

        if base_name not in rows_50:
            rows_50[base_name] = {
                "model": base_name,
                "train1": "",
                "train2": "",
                "train3": "",
                "train4": ""
            }

        if base_name not in rows_95:
            rows_95[base_name] = {
                "model": base_name,
                "train1": "",
                "train2": "",
                "train3": "",
                "train4": ""
            }

        row_50 = rows_50[base_name]
        row_95 = rows_95[base_name]

        for level in LEVELS:
            col_name = "train%d" % level     # columnas globales
            train_level_value = level        # 1..4 para CSV por clase

            tuned_name = "WAID_%s_%d.pt" % (tag, level)
            tuned_path = os.path.join(MODELS_DIR, tuned_name)

            print("\n  Level", level)
            print("  Tuned model:", tuned_path)

            if not os.path.exists(tuned_path):
                print("  Warning: tuned model not found, skipping.")
                continue

            try:
                model = YOLO(tuned_path)
            except Exception as e:
                print("  Error loading model:", e)
                continue

            try:
                metrics = model.val(
                    data=DATA_YAML,
                    split="test",
                    save_txt=True,
                    save_conf=True,
                    save=True,
                    project=VAL_PROJECT,
                    name="val_%s_%d" % (tag, level),
                    exist_ok=True,
                    device=TRAIN_DEVICE,
                    verbose=False
                )
            except Exception as e:
                print("  Error during val:", e)
                continue

            # métricas globales (p, r, map50, map50-95)
            try:
                _, _, map50_global, map5095_global = metrics.mean_results()
            except Exception as e:
                print("  Warning: could not read global metrics:", e)
                continue

            row_50[col_name] = "%.6f" % float(map50_global)
            row_95[col_name] = "%.6f" % float(map5095_global)
            print("  map50:     ", row_50[col_name])
            print("  map50-95: ", row_95[col_name])

            # --------- mAP por clase: AP50 y AP50-95 ----------
            per_class_rows = []
            try:
                names = metrics.names
                n_classes = len(names)
                for cid in range(n_classes):
                    pc, rc, ap50c, ap5095c = metrics.class_result(cid)
                    per_class_rows.append({
                        "base_model": base_name,
                        "train_level": train_level_value,  # 1..4
                        "class_id": cid,
                        "class_name": names[cid],
                        "map50": float(ap50c),
                        "map50_95": float(ap5095c)
                    })
            except Exception as e:
                print("  Warning: could not read per-class metrics:", e)
                per_class_rows = []

            if per_class_rows:
                append_per_class_rows(per_class_rows, CSV_PER_CLASS, PER_CLASS_HEADER)
                print("  Per-class metrics saved for this model/level.")

            save_csv(rows_50, CSV_PATH_50, CSV_HEADER)
            save_csv(rows_95, CSV_PATH_95, CSV_HEADER)

    save_csv(rows_50, CSV_PATH_50, CSV_HEADER)
    save_csv(rows_95, CSV_PATH_95, CSV_HEADER)
    print("\nDone. Results saved to:")
    print("  ", CSV_PATH_50)
    print("  ", CSV_PATH_95)
    print("  ", CSV_PER_CLASS)


if __name__ == "__main__":
    main()

