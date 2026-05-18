from ultralytics import YOLO
import os
import shutil
import time
import csv
import time
from datetime import datetime

# -------------------------------------------------------
# BASE DIR
# -------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
    
# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------



# Dataset
DATA_YAML = os.path.join(BASE_DIR, "waid.yaml")

# Fraction of training data (0-1)
DATA_FRACTION = 1.0   # Todas
#DATA_FRACTION = 0.0994 # ≈1000
#DATA_FRACTION = 0.00994 # ≈100

# Models
MODELS = [
    os.path.join(PROJECT_ROOT, "Models", "yolo11n.pt"),
    os.path.join(PROJECT_ROOT, "Models", "yolo11s.pt"),
    os.path.join(PROJECT_ROOT, "Models", "yolo11m.pt"),
    os.path.join(PROJECT_ROOT, "Models", "yolo11l.pt"),
    os.path.join(PROJECT_ROOT, "Models", "yolo11x.pt")
]
# Levels
#LEVELS = [1,2,3,4]
LEVELS = [1,2,3]
# Output
OUT_DIR = os.path.join(PROJECT_ROOT, "ModelsWAID")
os.makedirs(OUT_DIR, exist_ok=True)

# CSV

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

CSV_PATH = os.path.join(LOG_DIR , "training_times.csv")
CSV_HEADER = ["model", "train1", "train2", "train3", "train4"]

LOG_CSV = os.path.join(LOG_DIR , "training_log.csv")
LOG_HEADER = ["datetime","model","levels","fraction","epochs"]

# -------------------------------------------------------
# DEVICE
# -------------------------------------------------------

# GPU
USE_GPU = True       # True = GPU | False = CPU
GPU_ID = "2"

if USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    DEVICE = 2
    print("Using GPU:", GPU_ID)
else:
    DEVICE = "cpu"
    print("Using CPU")


# -------------------------------------------------------
# CSV
# -------------------------------------------------------

def save_csv(rows):

    with open(CSV_PATH, "w", newline="") as f:

        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()

        for model in sorted(rows.keys()):
            writer.writerow(rows[model])


def log_run(model, level, fraction, epochs):
    file_exists = os.path.exists(LOG_CSV)

    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(LOG_HEADER)

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model,
            f"[{level}]",
            fraction,
            epochs,
        ])

# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():

    rows = {}


    # Load CSV if exists
    if os.path.exists(CSV_PATH):

        with open(CSV_PATH, "r") as f:

            reader = csv.DictReader(f)

            for row in reader:

                for col in CSV_HEADER:
                    if col not in row:
                        row[col] = ""

                rows[row["model"]] = row


    # Train loop
    print("PROJECT_ROOT:", PROJECT_ROOT)
    for MODEL_PATH in MODELS:
        print("MODEL_PATH:", MODEL_PATH, "exists:", os.path.exists(MODEL_PATH))

        base_name = os.path.basename(MODEL_PATH)
        tag = base_name.replace(".pt", "").replace("yolo", "")

        now = datetime.now()
        
        print("\n====================================")
        print("Model:", base_name)
        print("Levels:", LEVELS)
        print("Fraction:", DATA_FRACTION)
        print("Hora actual:", now.strftime("%Y-%m-%d %H:%M:%S"))
        print("====================================")


        # CSV row
        if base_name not in rows:

            rows[base_name] = {
                "model": base_name,
                "train1": "",
                "train2": "",
                "train3": "",
                "train4": ""
            }


        row = rows[base_name]


        for level in LEVELS:

            print("\nTraining level:", level)


            out_name = "WAID_%s_%d.pt" % (tag, level)
            out_path = os.path.join(OUT_DIR, out_name)

            run_name = "waid_%s_%d" % (tag, level)

            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"MODEL_PATH no existe: {MODEL_PATH}")
            else:
                print("Usando modelo local:", MODEL_PATH)
            model = YOLO(MODEL_PATH, task="detect")


            # Hyperparams
            if level == 1:
                freeze = 20
                epochs = 10
                lr0 = 0.0005

            elif level == 2:
                freeze = 10
                epochs = 30
                lr0 = 0.001

            elif level == 3:
                freeze = 5
                epochs = 40
                lr0 = 0.001

            elif level == 4:
                freeze = 0
                epochs = 50
                lr0 = 0.001

            else:
                raise ValueError("Invalid level")


            # Train
            log_run(base_name, level,DATA_FRACTION,epochs)
            start = time.time()

            results = model.train(
                data=DATA_YAML,
                epochs=epochs,
                imgsz=640,
                batch=16,
                device=DEVICE,
                lr0=lr0,
                freeze=freeze,
                fraction=DATA_FRACTION, 
                seed=42,
                project=os.path.join(PROJECT_ROOT, "runs"), 
                name=run_name,
                exist_ok=True

            )


            total = time.time() - start
            avg = total / epochs

            print("Average epoch time:", round(avg, 3), "s")


            # Copy best model
            best = os.path.join(results.save_dir, "weights", "best.pt")
            if os.path.exists(best):
                shutil.copy(best, out_path)
                print("Saved:", out_path)
            else:
                print("Warning: best.pt not found at", best)


            # Update CSV
            row["train%d" % level] = "%.6f" % avg


            save_csv(rows)
            time.sleep(60) 


# -------------------------------------------------------

if __name__ == "__main__":
    main()