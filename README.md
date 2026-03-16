# <p align="center"> Retention Time (RT) Prediction of Proteolytic Peptides by Transformer Learning
![Graphical Abstract2](https://github.com/user-attachments/assets/d17db877-5319-4e60-8dc8-11bd43a33526)

<div align="center">
  
  [![Tensorflow](https://img.shields.io/badge/Tensorflow-v2.20.0-FF6F00?logo=tensorflow&style=for-the-badge&logoSize=amd&logoColor=FDEE21)](https://www.tensorflow.org/)
  [![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.7.1-ECD53F?logo=scikitlearn&style=for-the-badge&logoSize=amd&logoColor=FDEE21)](https://scikit-learn.org/)
  [![Keras](https://img.shields.io/badge/Keras-v3.11.3-D00000?logo=keras&style=for-the-badge&logoSize=amd&logoColor=FDEE21)](https://keras.io/)
  [![SciPy](https://img.shields.io/badge/SciPy-v1.16.1-8CAAE6?logo=scipy&style=for-the-badge&logoSize=amd&logoColor=FDEE21)](https://scipy.org/)
  [![Pandas](https://img.shields.io/badge/pandas-v2.3.1-00AA45?logo=pandas&style=for-the-badge&logoSize=amd&logoColor=FDEE21)](https://pandas.pydata.org/)
  [![NumPy](https://img.shields.io/badge/NumPy-v2.2.6-897BFF?logo=numpy&style=for-the-badge&logoSize=amd&logoColor=FDEE21)](https://numpy.org/)
  [![Seaborn](https://img.shields.io/badge/Seaborn-v0.13.2-F1007E?logo=seaborn&style=for-the-badge&logoSize=amd&logoColor=FDEE21)](https://seaborn.pydata.org/)

</div>

# 1. Installation
```python
git clone https://github.com/Transformer-RT-Prediction/Peptide-RT-Prediction
cd Peptide-RT-Prediction
```

# 2. Datasets
## 2.1. LC types
1. RPLC - `Exp. HeLa (unmod. and mod.)`; `DeepRT HeLa (unmod. and mod.)`; `DeepRT Yeast`; `DeepRT Atlantis Silica`; `DeepLC Xbridge`; `DeepLC Luna Silica`; `DeepLC Misc. (mod.)`
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
1. `ILYSQCGDVMR | 1xCarbamidomethyl [C6]` represents one carbamidomethyl modification on 6th postion of Cys residue.
2. `ISNASCTTNCLAPLAK | 2xCarbamidomethyl [C7; C11]`represents two carbamidomethyl modifications on 7 and 11th positions of Cys residues.

#### Types of modifications -
1. Exp. HeLa: `Cys-Carbamidomethylation`, `Lys, Ser-Oxidation`, `Met-N-Term Acetylation`
2. DeepRT HeLa: `Lys, Tyr-Oxidation`, `Met, Thr-Phospho`
3. DeepLC Misc.: `Lys, Tyr-Oxidation`, `Tyr-Phospho`, `Arg-Acetylation`, `Met-Propionyl`, `Pro-Succinyl`, `Lys-Biotin`, `Tyr-Butyryl`, `Met-Crotonyl`, `Pro-Deamidated`, `Met-Formyl`,`Lys-GG (Glycine-Glycine)`,`Met-Malonyl`, `Tyr-Methyl`, `Met-Nitro`

# 3. Usage

## 3.1. Package layout
The project is organized into the following modules -

`rt_pred/config/` – Experiment Configuration

Defines configuration objects and default experiment settings
  - `ExperimentConfig`,
  - `hela_unmodified_config`,
  - `hela_modified_config`


`rt_pred/tokenizer/` – Peptide Tokenization Utilities

Functions for building tokenizers and encoding peptide sequences
  - `infer_alphabet`,
  - `build_tokenizer`,
  - `encode_sequence`


`rt_pred/model/` – Model Architecture

Core model components and utilities
  - `schedules.py`
    – `WarmupCosine` - learning rate scheduler
  - `layers.py` – encoder and pooling layers
  - `builder.py`
    – `build_model_from_hp` - construct model from hyperparameters
  - `utils.py`
    – `set_seed` - ensure reproducible experiments


`rt_pred/metrics/` – Evaluation and RT unit helpers

Metrics and helper functions for retention time (RT) evaluation
  - `pearson_r`,
  - `p95_width`,
  - `residual_ci95`,
  - `normalize_rt_units`


- `rt_pred/training/` – End‑to‑End Training & Evaluation

Training pipelines and evaluation workflows
  - `make_ds`,
  - `tune_hyperparams`,
  - `run_cross_validation`,
  - `train_one_file`,
  - `run_experiment`


- `rt_pred/data/` – Data Loading and Modification Handling

Utilities for loading datasets and handling peptide modifications
  - `unmodified.py`
    – `load_tsv_unmodified`
  - `modified.py`
    – `load_tsv_modified`
    - modification parsing helpers


- `rt_pred/experiments/` – Ready-to-Run Experiment Entry Points

Scripts for running predefined experiments
  - `hela_unmodified.py`
  - `hela_modified.py`

All major functions can be imported directly from the package -

```python
from rt_pred.config import hela_unmodified_config, hela_modified_config
from rt_pred.data import load_tsv_unmodified, load_tsv_modified
from rt_pred.training import run_experiment
from rt_pred.model import build_model_from_hp, set_seed
```

## 3.2. Running the code

Make sure your working directory is the project root (e.g. `c:\Users\Mahin\Downloads\Transformer RT`).

### 3.2.1. Unmodified Hela experiment

Assume your unmodified dataset is located in the following directory -

- `Transformer RT\hela_unmodified.txt`

You can run the full pipeline with -

```python
from rt_pred.experiments.hela_unmodified import main as run_unmodified

# save all outputs (metrics + predictions) into the current folder
run_unmodified(root=".")
```

Outputs:

- Load `./hela_unmodified.txt`
- Perform Hyperparameter search
- Run 5‑fold cross-validation
- Train the final model
- Save evaluation metrics and predictions as CSV files in `root` (here: `.`).

### 3.2.2. Modified Hela experiment

Assume your modified dataset is located in this dirctory -

- `Transformer RT\mod_hela_exp.csv`

Run:

```python
from rt_pred.experiments.hela_modified import main as run_modified

run_modified(root=".")
```

This will:

- Load `./mod_hela_exp.csv`
- Parse the `Modifications` column, map each modification type to a character from `MOD_CHAR_POOL`
- Generate modification-derived sequences
- Train and evaluate the same model architecture as used in the unmodified dataset
- Save all CSV outputs into `root`.

# 4. Using from jupyter notebooks

You can run the code directly from jupyter notebook located in the project root -

```python
from rt_pred.experiments.hela_unmodified import main as run_unmodified
from rt_pred.experiments.hela_modified import main as run_modified

# Run the unmodified dataset
run_unmodified(root=".")

# Run the modified dataset
run_modified(root=".")
```

If your data is stored in a different directory, simply pass the path to the `root` argument -

```python
run_unmodified(root=r"C:\path\to\unmodified_data")
run_modified(root=r"C:\path\to\modified_data")
```

The code expects the following files in the specified `root` directory -

- `root / "hela_unmodified.txt"` for the unmodified dataset
- `root / "mod_hela_exp.csv"` for the modified dataset

# 5. Code customization and configuration

You can define your own configuration and run the generic training pipeline directly

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

### Running with a Modified Dataset
For datasets that include peptide modifications -

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

### Customizing the Hyperparameter search
You can modify the hyperparameter search grid by updating `cfg.hp_configs` 

This attribute is a list of dictionaries with the following keys -
`name, D_MODEL, N_LAYERS, N_HEADS, D_FF, DROPOUT, BASE_LR`

# 6. Output Files

The outputs are saved into the directory specified by `cfg.root` (or by the `root` argument passed to the `main` functions)

The following files are generated -
- `*_cv_metrics.csv` – per‑fold cross‑validation metrics
- `*_test_predictions_cv.csv` – predictions from each cross-validation fold
- `*_train_history.csv` – training and validation loss per epoch
- `*_train_predictions.csv` – predictions on the training split
- `*_validation_predictions.csv` – predictions on the validation split
- `rt_transformer_metrics.csv` – summary metrics for each dataset

# 7. Transformer Architecture
The main architecture is composed of 4 major stages -
## 1. Preprocessing of peptide sequences and PTMs -

   ![Preprocessing](https://github.com/user-attachments/assets/248eeae2-03c5-4910-bfbd-aef0150abc5c)

   
   In this step, the input peptide sequences are encoded to predefined alphabets by an alphabet decision module. The base alphabets of amino acids can be defined as
   ```python
   BASE_AA = {A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y}
   ```
   For PTM-containing peptides, numeric, lowercase, uppercase and special characters were utilized along with the base alphabets.
   ```python
   DP_ALPHABET = "ACDEFGHIKLMNPQRSTVWY1234*"
   ```
## 2. Conformer-lite econding

   ![Conformer stack](https://github.com/user-attachments/assets/7346315e-e30a-49f7-ae7c-a309fd03cf7a)
   
      
## 3. Hybrid pooling mechanism

   ![Hybrid Pooling Mechanism](https://github.com/user-attachments/assets/022a633c-94c0-43a4-b19e-5eb80651bc9b)


   
## 4. Regression head

   ![Regression Head](https://github.com/user-attachments/assets/8bf77fd3-cdac-4943-9540-bb8a19ec1dca)



