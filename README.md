# Retention Time Prediction of Proteolytic Peptides by Transformer Learning
![Graphical Abstract2](https://github.com/user-attachments/assets/d17db877-5319-4e60-8dc8-11bd43a33526)

[![Tensorflow](https://img.shields.io/badge/Tensorflow-v2.20.0-FF6F00?logo=tensorflow&style=for-the-badge&logoSize=amd&logoColor=FDEE21)](https://www.tensorflow.org/) 
[![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.7.1-ECD53F?logo=scikitlearn&style=for-the-badge&logoSize=amd&logoColor=FDEE21)](https://scikit-learn.org/) 
[![Keras](https://img.shields.io/badge/Keras-v3.11.3-D00000?logo=keras&style=for-the-badge&logoSize=amd&logoColor=FDEE21)](https://keras.io/) 
[![SciPy](https://img.shields.io/badge/SciPy-v1.16.1-8CAAE6?logo=scipy&style=for-the-badge&logoSize=amd&logoColor=FDEE21)](https://scipy.org/) 
[![Pandas](https://img.shields.io/badge/pandas-v2.3.1-00AA45?logo=pandas&style=for-the-badge&logoSize=amd&logoColor=FDEE21)](https://pandas.pydata.org/) 
[![NumPy](https://img.shields.io/badge/NumPy-v2.2.6-897BFF?logo=numpy&style=for-the-badge&logoSize=amd&logoColor=FDEE21)](https://numpy.org/) 
[![Seaborn](https://img.shields.io/badge/Seaborn-v0.13.2-F1007E?logo=seaborn&style=for-the-badge&logoSize=amd&logoColor=FDEE21)](https://seaborn.pydata.org/)

# 1. Installation
```python
git clone https://github.com/Transformer-RT-Prediction/Peptide-RT-Prediction
cd Peptide-RT-Prediction
```
# 2. Datasets
## 2.1. Different LC types

Total 11 datasets including 3 LC conditions were utilized -

1. RPLC - `Exp. HeLa (unmod. and mod.)`; `DeepRT HeLa (unmod. and mod.)`; `DeepRT Yeast`; `DeepRT Atlantis Silica`; `DeepLC Xbridge`; `DeepLC Luna Silica`; `DeepLC Misc. (modified)`

2. HILIC - `DeepLC Luna HILIC`

3. SCX - `DeepLC SCX`

| Dataset | Number of Peptides | Peptide Seq Length |
|:--------|:-------------------:|:-------------------:|
| Exp. HeLa | 4,213 | 6–33 |
| Exp. HeLa (modified) | 704 | 6–27 |
| DeepRT HeLa | 3,413 | 6–50 |
| DeepRT HeLa (modified) | 2,243 | 7–50 |
| DeepRT Yeast | 14,361 | 6–38 |
| DeepRT Atlantis Silica | 39,091 | 6–49 |
| DeepLC Xbridge | 40,037 | 6–51 |
| DeepLC SCX | 30,471 | 6–49 |
| DeepLC Luna HILIC | 36,080 | 7–51 |
| DeepLC Luna Silica | 36,823 | 7–48 |
| DeepLC Misc. (modified) | 14,557 | 7–45 |

## 2.2. Dataset Structure (Input files)
Datasets of unmodified and modified peptides were prepared as the following formats -
### 2.2.1. Unmodified peptides
| Sequence        | RT    |
|:-----------------|:-------:|
| AAAAAAAAAAPAAAAATAPTTATATATAAQ | 67.3  |
| AAAAAAALQAK     | 20.42 |
| AAAEVAGQFVT  | 57.61 |
| AAAEVNQDYGLDPK  | 43.16 |
| AAALEAMK        | 23.54 |

### 2.2.2. Modified peptides
| Sequence | Modifications | RT |
|:---------|:--------------:|:---:|
| LLYEALVDCK | 1xCarbamidomethyl [C9] | 70.71 |
| IYYGGSVTGATCK | 1xCarbamidomethyl [C12] | 38.56 |
| ILYSQCGDVMR | 1xCarbamidomethyl [C6] | 47.68 |
| ISNASCTTNCLAPLAK | 2xCarbamidomethyl [C7; C11] | 56.89 |
| IINDNATYCR | 1xCarbamidomethyl [C9] | 25.69 |

#### Explanations -
`ILYSQCGDVMR | 1xCarbamidomethyl [C6]` indicates one carbamidomethyl modification on 6th postion of Cys residue.

`ISNASCTTNCLAPLAK | 2xCarbamidomethyl [C7; C11]`indicates two carbamidomethyl modifications on 7 and 11th positions of Cys residues.

#### Types of modifications -
1. Exp. HeLa: `Cys-Carbamidomethylation`, `Lys, Ser-Oxidation`, `Met-N-Term Acetylation`
2. DeepRT HeLa: `Lys, Tyr-Oxidation`, `Met, Thr-Phospho`
3. DeepLC Miscellaneous: `Lys, Tyr-Oxidation`, `Tyr-Phospho`, `Arg-Acetylation`, `Met-Propionyl`, `Pro-Succinyl`, `Lys-Biotin`, `Tyr-Butyryl`, `Met-Crotonyl`, `Pro-Deamidated`, `Met-Formyl`,`Lys-GG (Glycine-Glycine)`,`Met-Malonyl`, `Tyr-Methyl`, `Met-Nitro`





## RT prediction package – usage guide

This package refactors the original notebooks
`rt_pred_exp_hela1 (2).ipynb` and `rt_pred_exp_mod_hela1.ipynb`
into a reusable, folder‑based Python package.

### 1. Package layout (high level)

- `rt_pred/config/` – experiment configuration
  - `ExperimentConfig`, `hela_unmodified_config`, `hela_modified_config`
- `rt_pred/tokenizer/` – peptide tokenization utilities
  - `infer_alphabet`, `build_tokenizer`, `encode_sequence`
- `rt_pred/model/` – model architecture
  - `schedules.py` – `WarmupCosine`
  - `layers.py` – encoder and pooling layers
  - `builder.py` – `build_model_from_hp`
  - `utils.py` – `set_seed`
- `rt_pred/metrics/` – evaluation and RT unit helpers
  - `pearson_r`, `p95_width`, `residual_ci95`, `normalize_rt_units`
- `rt_pred/training/` – end‑to‑end training & evaluation
  - `make_ds`, `tune_hyperparams`, `run_cross_validation`,
    `train_one_file`, `run_experiment`
- `rt_pred/data/` – data loading & modification handling
  - `unmodified.py` – `load_tsv_unmodified`
  - `modified.py` – `load_tsv_modified` and helpers
- `rt_pred/experiments/` – ready‑to‑run experiment entry points
  - `hela_unmodified.py`
  - `hela_modified.py`

All public functions are available via the package:

```python
from rt_pred.config import hela_unmodified_config, hela_modified_config
from rt_pred.data import load_tsv_unmodified, load_tsv_modified
from rt_pred.training import run_experiment
from rt_pred.model import build_model_from_hp, set_seed
```

### 2. Running the Hela experiments from Python

Make sure your working directory is the project root
(`c:\Users\Mahin\Downloads\Transformer RT` on Windows).

#### 2.1 Unmodified Hela experiment

Assume your data file is at:

- `c:\Users\Mahin\Downloads\Transformer RT\hela_unmodified.txt`

You can run the full pipeline with:

```python
from rt_pred.experiments.hela_unmodified import main as run_unmodified

# save all outputs (metrics + predictions) into the current folder
run_unmodified(root=".")
```

This will:

- Load `./hela_unmodified.txt`
- Perform HP search (unless disabled in config)
- Run 5‑fold CV
- Train the final model
- Write CSVs (metrics, CV metrics, predictions, training curves)
  into `root` (here: `.`).

#### 2.2 Modified Hela experiment

Assume your data file is at:

- `c:\Users\Mahin\Downloads\Transformer RT\mod_hela_exp.csv`

Run:

```python
from rt_pred.experiments.hela_modified import main as run_modified

run_modified(root=".")
```

This will:

- Load `./mod_hela_exp.csv`
- Parse the `Modifications` column, map each modification type
  to a character from `MOD_CHAR_POOL`
- Produce modification‑aware sequences
- Train and evaluate the same model architecture as above
- Save all CSV outputs into `root`.

### 3. Using from Jupyter notebooks

Inside a notebook in the project root you can call:

```python
from rt_pred.experiments.hela_unmodified import main as run_unmodified
from rt_pred.experiments.hela_modified import main as run_modified

# run unmodified dataset
run_unmodified(root=".")

# run modified dataset
run_modified(root=".")
```

If your data lives in a different folder, just point `root` there:

```python
run_unmodified(root=r"C:\path\to\unmodified_data")
run_modified(root=r"C:\path\to\modified_data")
```

The code expects:

- `root / "hela_unmodified.txt"` for the unmodified experiment
- `root / "mod_hela_exp.csv"` for the modified experiment

### 4. Custom experiments and configuration

You can build your own config and run the generic pipeline directly.

```python
from pathlib import Path

from rt_pred.config import ExperimentConfig
from rt_pred.data import load_tsv_unmodified
from rt_pred.training import run_experiment

cfg = ExperimentConfig(
    root=Path(r"C:\my\rt\project"),
    files=["my_dataset.tsv"],      # file(s) inside root
    epochs=300,                    # override defaults if needed
    batch=128,
    hp_search=True,                # or False to skip HP search
)

run_experiment(cfg, load_df=load_tsv_unmodified)
```

For a modified dataset:

```python
from rt_pred.config import ExperimentConfig
from rt_pred.data import load_tsv_modified
from rt_pred.training import run_experiment

cfg = ExperimentConfig(
    root=Path(r"C:\my\rt\project"),
    files=["my_modified_rt.csv"],
)

run_experiment(cfg, load_df=load_tsv_modified)
```

You can also adjust the hyperparameter search grid by modifying
`cfg.hp_configs` (a list of dicts with keys
`name, D_MODEL, N_LAYERS, N_HEADS, D_FF, DROPOUT, BASE_LR`).

### 5. Advanced: direct access to model and tokenizer

If you need to use the model and tokenizer outside the full pipeline:

```python
from rt_pred.tokenizer import infer_alphabet, build_tokenizer, encode_sequence
from rt_pred.model import build_model_from_hp
from rt_pred.config import ExperimentConfig, DEFAULT_HP_CONFIGS

seqs = ["PEPTIDE", "ACDMK"]
alphabet = infer_alphabet(seqs)
tok = build_tokenizer(alphabet)
max_len = max(len(s) for s in seqs)
X = np.stack([encode_sequence(s, tok, max_len) for s in seqs])

cfg = ExperimentConfig(root=Path("."), files=[])
hp = DEFAULT_HP_CONFIGS[0]
steps_per_epoch = 1

model = build_model_from_hp(
    hp,
    max_len_with_cls=max_len + 1,
    vocab_size=max(tok.values()) + 1,
    steps_per_epoch=steps_per_epoch,
    epochs=cfg.epochs,
    cfg=cfg,
)
preds = model.predict(X)
```

### 6. Where outputs are saved

All experiment outputs are written into the directory specified
by `cfg.root` (or the `root` argument passed to the `main` functions):

- `*_cv_metrics.csv` – per‑fold cross‑validation metrics
- `*_test_predictions_cv.csv` – per‑fold CV predictions
- `*_train_history.csv` – training and validation loss per epoch
- `*_train_predictions.csv` – predictions on the training split
- `*_validation_predictions.csv` – predictions on the held‑out split
- `rt_transformer_metrics.csv` – summary metrics per file

