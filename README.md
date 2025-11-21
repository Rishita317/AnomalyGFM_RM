# AnomalyGFM — Graph Foundation Model for Zero/Few-shot Anomaly Detection

Short: AnomalyGFM finds abnormal nodes in graphs (social, transaction, e‑commerce, etc.). This README gives explicit, reproducible setup and run steps for macOS (Intel & Apple Silicon) and Linux.

## Quick start (recommended, reproducible)

1. Clone repo and open project root:
   cd ~/Downloads/AnomalyGFM

2. Create a conda environment with Python 3.9 (recommended):
   conda create -n anomalygfm python=3.9 -y
   conda activate anomalygfm

3. Install lightweight Python deps:
   Create `requirements_fixed.txt` with:

   ```
   matplotlib>=3.5.0
   networkx>=2.6.3
   numpy>=1.21.6
   pandas>=1.3.5
   scikit-learn>=1.0.2
   scipy>=1.7.3
   tqdm
   ```

   Then:
   pip install -r requirements_fixed.txt

4. Install PyTorch (use official instructions for your platform):

   - CPU / Apple Silicon (conda):
     conda install pytorch torchvision torchaudio -c pytorch -y
   - GPU (Linux) — follow https://pytorch.org

5. Install DGL and graph libraries:

   - Try CPU wheel (works for many macOS setups):
     pip uninstall -y dgl
     pip install dgl -f https://data.dgl.ai/wheels/cpu/repo.html
   - If using PyG/torch-sparse/torch-geometric (required by some modules):
     conda install -c pyg -c conda-forge torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -y
   - If issues occur on macOS/arm64 see Troubleshooting section.

6. Place pretrained weights:
   mkdir -p pretrain
   cp few_shot/model_weights_abnormal300.pth pretrain/

## Datasets

- Download the aligned GAD datasets from:
  https://drive.google.com/drive/folders/1SSWgFRdth3U44_IMRnW775B1l-bjQATW?usp=sharing
- After unzip, place `.mat` files in repository root or `datasets/` (scripts check repo root by default).
  Example:
  mv ~/Downloads/gad_extracted/\*.mat ~/Downloads/AnomalyGFM/

Note for this fork (AnomalyGFM_RM)
---------------------------------
This fork intentionally excludes large dataset and model files to keep the repository lightweight and easy to clone. To run experiments locally you must download the datasets and pretrained weights and place them into the project. Follow these steps:

1. Download the dataset `.mat` files from the link above and put them in either the repository root or `datasets/`.

2. If you need pretrained weights, create `pretrain/` and place the `.pth` file(s) there, for example:

  mkdir -p pretrain
  cp /path/to/model_weights_abnormal300.pth pretrain/

3. Verify files are present before running scripts:

  ls -la *.mat datasets/*.mat || true
  ls -la pretrain || true

If you prefer not to keep datasets in the repo, you can keep them in a separate folder and update the scripts to point to that path. For reproducibility, note the Drive folder above where the original datasets are hosted.

Verify:
ls -la \*.mat
ls -la datasets/

## Where the run scripts live (use these exact commands)

Files are in subfolders — run from project root:

- Zero-shot inference
  python zero_shot/run_abnormal.py --dataset_train Facebook_svd --dataset_test yelp_svd

- Few-shot normal fine-tuning
  python few_shot/run_finetune_normal.py --dataset_train Facebook_svd --dataset_test elliptic_svd

- Few-shot abnormal fine-tuning
  python few_shot/run_finetune_abnormal.py --dataset_train Facebook_svd --dataset_test elliptic_svd

- Large-scale inference
  python large_scale/run_inference.py --dataset_test <dataset_name>

Or run from folder:
cd zero_shot && python run_abnormal.py --dataset_train Facebook_svd --dataset_test yelp_svd

## Minimal smoke-test (quick health check)

Create `scripts/smoke_test.py` with:

```python
import sys, os
import torch
ok = {"python": sys.version.split()[0], "torch": torch.__version__}
try:
    import dgl; ok["dgl"] = dgl.__version__
except Exception as e:
    ok["dgl_error"] = str(e)
print(ok)
print("Datasets present:", any(f.endswith(".mat") for f in os.listdir(".")))
```

Run:
python scripts/smoke_test.py

## Typical problems & fixes (macOS / Apple Silicon)

1. DGL graphbolt / C++ library errors

   - Install CPU wheel: pip install dgl -f https://data.dgl.ai/wheels/cpu/repo.html
   - If graphbolt still errors, run guarded imports (see below) or use Linux/x86 environment (Docker / remote machine).

2. torch_sparse / torch_geometric missing or binary mismatch

   - Use conda binary builds:
     conda install -c pyg -c conda-forge pytorch-sparse torch-geometric -y
   - Or install PyG wheels matching your torch version:
     TORCH_VER=$(python - <<'PY'
     import torch, re
     print(re.sub(r'\+.*','',torch.__version__))
     PY
     )
     pip install -f https://data.pyg.org/whl/torch-${TORCH_VER}.html torch-sparse torch-scatter torch-geometric

3. Missing pydantic / torchdata errors

   - conda install -c conda-forge pydantic
   - pip install "torchdata>=0.11.0"

4. If a compiled package fails due to ABI mismatch, prefer recreating the conda env with the exact Python version used by builders (3.9 recommended).

## Recommended repository improvements (short)

- Add `environment.yml` and `requirements_fixed.txt`
- Add `scripts/smoke_test.py` and `scripts/download_datasets.sh` (small dataset placer)
- Guard optional imports (dgl, torch_sparse, torch_geometric) to print friendly messages
- Provide a small toy dataset for onboarding

## Troubleshooting quick commands

- Check your working dir and python:
  pwd
  which python
  python -V

- Verify imports:
  python - <<'PY'
  import torch, sys
  print("python", sys.version)
  try:
  import dgl; print("dgl", dgl.**version**)
  except Exception as e:
  print("dgl error", e)
  PY

## How to run experiments reproducibly

- Use the conda env above
- Put datasets into repo root
- Copy pretrained weights into `pretrain/`
- Run scripts using the subpath commands above

## Contact / citation

If this repo is useful please cite:

```
@inproceedings{qiao2025anomalygfm, ... }
```
