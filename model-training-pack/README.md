**README for Training and Prediction Scripts**

This guide explains how to use the two main entry points in the `training_package`:

1. **Training pipeline** (`training_pipeline.py`)
2. **Prediction script** (`predict_mask_from_model.py`)

---

## Prerequisites

- Python 3.7+
- Install package dependencies:

  ```bash
  pip install .
  ```

---

## 1. Training Pipeline

Use `training_pipeline.py` to process raw images/labels and train a Random Forest model.

```bash
python -m training_package.training_pipeline \
  --input-dir /path/to/experiment_folder \
  --model-dir /path/to/experiment_folder/models \
  [-c CONFIG_INDEX] [-o CONFIG_OPTIONS]
```

- `--input-dir`, `-i`
  Path to your data folder. Must contain subfolders: `Images/` and `Labels/`.

- `--model-dir`, `-m`
  Destination folder for trained model. Will be created if it doesn’t exist (default: `./models`).

- `--config-index`, `-c`
  Base configuration: `0` (deep), `1` (light). Default: `1`.

- `--config-options`, `-o`
  JSON string for additional config overrides, e.g. `'{"n_estimators":200}'`.

**Output:**

- Processed training data (if configured) and saved model file (e.g. `rf_model.joblib`) in `--model-dir`.

---

## 2. Prediction Script

Use `predict_mask_from_model.py` to generate a mask from a `.czi` image using your trained model.

```bash
python -m training_package.predict_mask_from_model \
  --czi /path/to/Images/YourImage.czi \
  --model /path/to/models/rf_model.joblib \
  [--output /path/to/predicted_mask.png] [--no-viz]
```

- `--czi`, `-i`
  Path to the input `.czi` microscopy image.

- `--model`, `-m`
  Path to the trained `.joblib` model (default: `./models/rf_model.joblib`).

- `--output`, `-o`
  Where to save the predicted mask PNG (default: `./predicted_mask.png`).

- `--no-viz`
  Skip plotting the mask overlay.

**Output:**

- Binary mask image saved to the `--output` path.

---

## Examples

```bash
# Train with defaults:
pitcount-cfim-training -i ~/Desktop/my-exp/Images -m ~/Desktop/my-exp/models

# Predict and view:
python -m training_package.predict_mask_from_model -i ~/Desktop/my-exp/Images/sample.czi -m ~/Desktop/my-exp/models/rf_model.joblib
```

---

_For full configuration options and advanced usage, see the docstrings in each module._
