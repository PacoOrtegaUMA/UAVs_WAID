# UAVs_WAID

Fine-tuning and incremental learning pipeline for wildlife detection from UAV imagery using YOLO11 models on the WAID and WAIDPlus datasets.

---

## Project Structure

```
UAVs_WAID/
├── Models/              # Pretrained YOLO11 base models (yolo11n.pt, yolo11s.pt, ...)
├── ModelsWAID/          # Fine-tuned models on WAID (WAID_11n_1.pt, ...)
├── WAID/                # WAID dataset (images + labels, train/val/test splits)
├── WAIDplus/            # WAIDPlus dataset (4 new classes added)
├── logs/                # Training times, validation metrics CSVs
├── runs/                # Ultralytics training and validation runs
├── plots/               # Output PDF plots
└── src/                 # All Python scripts and YAML configs
```

---

## Datasets

### WAID — Wildlife Animal Identification Dataset
6 classes: sheep, cattle, seal, camelus, kiang, zebra.

### WAIDPlus
Extends WAID with 4 new classes (IDs 6–9): crocodile, elephant, deer, horse.

---

## Pipeline

### 1. WAID Fine-tuning — `WAID_Training.py`

Fine-tunes YOLO11 base models on WAID using 4 progressive levels that gradually unfreeze layers:

| Level | Frozen layers | Epochs | LR |
|-------|--------------|--------|----|
| 1 | 20 | 10 | 0.0005 |
| 2 | 10 | 30 | 0.001 |
| 3 | 5 | 40 | 0.001 |
| 4 | 0 | 50 | 0.001 |

**Output:** `ModelsWAID/WAID_<tag>_<level>.pt`, training times in `logs/<device>/Train/`

---

### 2. WAID Validation — `WAID_Validation.py`

Evaluates fine-tuned WAID models using Ultralytics `model.val()` on the test split.

**Output:** mAP50 and mAP50-95 per model and level, per-class metrics in `logs/Waid_mAP/`

---

### 3. WAID Custom Evaluation — `WAID_mAP_Intervalo.py`

Custom mAP evaluation that breaks down detections by object size relative to image area, using intervals `[0, 0.1, 1, 10, 100]%`.

Computes both mAP@50 and mAP@50-95 (averaged over IoU thresholds 0.50–0.95).

**Output:** `logs/WAID_mAP_intervalo_1.csv`, `logs/WAID_AP_clase_intervalo_1.csv`

---

### 4. Inference Speed Benchmark — `WAID_InferenceTime.py`

Measures inference time per image for all 20 trained models (5 sizes × 4 levels) on up to 100 test images.

**Output:** `logs/<device>/Predic_t_detalle.csv`, `logs/<device>/Predic_t_resumen.csv`

---

### 5. WAIDPlus Incremental Training — `WAIDPlus_Training.py`

Incrementally trains WAID models to learn the 4 new WAIDPlus classes while retaining knowledge of the original 6, using a block-by-block strategy that alternates new and old class images (anti-catastrophic-forgetting).

Each block accumulates all images seen so far (new + old), training up to 100 blocks per level.

**Output:** `runs/waidplus/waidplus_<tag>_L<level>_B<block>/weights/best.pt`, block times in `logs/waidplus_blocks_times.csv`

---

### 6. WAIDPlus Validation — `WAIDPlus_Validation.py`

Validates each trained block checkpoint on the WAIDPlus test split, saving per-class metrics and confusion matrices.

**Output:** `logs/ValWAID+/class_L<level>_B<block>.csv`, `logs/ValWAID+/confmat_<tag>.csv`

---

### 7. Results Plotting — `WAIDPlus_Plots.py`

Reads all validation CSVs and generates PDF plots of mAP evolution across blocks, separately for old and new classes.

**Output (PDFs in `plots/`):**
- `map50_old_by_block.pdf`
- `map50_new_by_block.pdf`
- `map5095_old_by_block.pdf`
- `map5095_new_by_block.pdf`
- `confmat_fullnorm_L<level>.pdf`

---

## Requirements

```bash
conda create -n WAID python=3.11
conda activate WAID
mamba install -c conda-forge pytorch torchvision torchaudio
mamba install -c conda-forge ultralytics
```

Tested with:
- Python 3.11
- PyTorch 2.10.0
- Ultralytics 8.x
- CUDA 12.9

---

## GPU Configuration

Scripts auto-detect the machine via `socket.gethostname()` and configure the GPU accordingly. To add a new machine, edit the `HOSTNAME` block at the top of each script:

```python
import socket
HOSTNAME = socket.gethostname()

if HOSTNAME == "Isolda":
    GPU_ID = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    TRAIN_DEVICE = 0
else:
    TRAIN_DEVICE = "cpu"
```

---

## Authors

Francisco Ortega — Universidad de Málaga
