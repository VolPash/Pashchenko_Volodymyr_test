# Deforestation matching — Quick Start

This repository contains a pipeline to co-register and match Sentinel TCI (True Color Image) tiles using SuperPoint + SuperGlue, with additional GIS masking for deforestation zones.

**Dataset origin:** taken from Kaggle (see `dataset_load.py` for the exact Kaggle dataset slug and download logic).

---

## Models used (why SuperPoint + SuperGlue)

We chose **SuperPoint** (local feature detector / descriptor) together with **SuperGlue** (learned context-aware matcher) as the core feature-matching stack for this project because:

* **Robustness on remote-sensing images:** SuperPoint provides dense, repeatable keypoints and compact descriptors which work well on aerial and satellite imagery after moderate resizing.
* **Contextual matching:** SuperGlue learns to reason about global context and matches features more reliably than simple descriptor nearest-neighbour approaches, improving homography estimation and reducing spurious matches in areas with repetitive texture (fields, forests).
* **Pretrained, open-source weights:** Both models have high-quality pretrained weights publicly available (open access GitHub / model releases). Using these pretrained weights lets us avoid costly training and yields good zero-shot performance on co-registration tasks.

> **Note:** the project expects the model weights to be placed in `./models/`:
>
> * `models/superpoint_v1.pth`
> * `models/superglue_outdoor.pth`
>
> These weight files were downloaded from their public releases (open-access repositories) and included locally in the project. If you prefer to re-download them, place the `.pth` files into `./models/`.

---

## Prerequisites

* Python 3.9+ recommended
* GPU with CUDA is recommended for speed; CPU mode also supported
* Put model weights in `./models/`:

  * `models/superpoint_v1.pth`
  * `models/superglue_outdoor.pth`

---

## Install dependencies

Install packages listed in `requirements.txt` (this repository includes a minimal `requirements.txt` with package names only):

```bash
pip install -r requirements.txt
```


## High-level pipeline and order to run

Run scripts in this order (each script prints progress and output paths):

1. **Requirements** (install packages) — already done above.

2. **Download dataset from Kaggle**

   ```bash
   python dataset_load.py
   ```

   The script uses the Kaggle API. Provide your credentials by placing `kaggle.json` in `~/.kaggle/` or exporting `KAGGLE_USERNAME` and `KAGGLE_KEY` as environment variables.

3. **Prepare/convert dataset**

   ```bash
   python data_loader.py
   ```

   This script converts raw JP2 tiles to the local structure used by the pipeline, rescales images and extracts metadata.

4. **Feature extraction**

   ```bash
   python feature_extractor.py
   ```

   Runs SuperPoint → SuperGlue on selected pairs. The script expects model weights in `./models/`.

5. **Inference / Co-registration**

   ```bash
   python inference.py
   ```

   Performs final co-registration, RANSAC homography estimation and scoring. Produces visualizations.

All final outputs will be saved into:

```
./results/
```

Each script prints its results (saved file paths) to the console.

---

