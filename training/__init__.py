from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

from ..config import ExperimentConfig
from ..metrics import pearson_r, p95_width, residual_ci95, normalize_rt_units
from ..model import build_model_from_hp, set_seed
from ..tokenizer import build_tokenizer, encode_sequence, infer_alphabet


DataFrameLoader = Callable[[Path], pd.DataFrame]


def make_ds(X, y=None, batch: int = 256, shuffle: bool = False) -> tf.data.Dataset:
    ds = (
        tf.data.Dataset.from_tensor_slices((X, y))
        if y is not None
        else tf.data.Dataset.from_tensor_slices(X)
    )
    if shuffle:
        ds = ds.shuffle(len(X), seed=42)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def tune_hyperparams(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_len_with_cls: int,
    vocab_size: int,
    cfg: ExperimentConfig,
) -> Dict:
    """
    Simple manual hyperparameter search over cfg.hp_configs.
    Returns the best hp dict.
    """
    print(f"\n[HP SEARCH] Train size: {len(X_tr)}, Val size: {len(X_val)}")
    steps_per_epoch = max(1, len(X_tr) // cfg.batch)

    best_hp = None
    best_loss = np.inf

    for i, hp in enumerate(cfg.hp_configs):
        print(f"\n[HP {i+1}/{len(cfg.hp_configs)}] {hp['name']}")
        print("  config:", {k: v for k, v in hp.items() if k != "name"})

        model = build_model_from_hp(
            hp,
            max_len_with_cls=max_len_with_cls,
            vocab_size=vocab_size,
            steps_per_epoch=steps_per_epoch,
            epochs=cfg.epochs_tune,
            cfg=cfg,
        )

        ds_tr = make_ds(X_tr, y_tr[:, None], batch=cfg.batch, shuffle=True)
        ds_val = make_ds(X_val, y_val[:, None], batch=cfg.batch)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=100,
                restore_best_weights=True,
                verbose=1,
            )
        ]

        hist = model.fit(
            ds_tr,
            validation_data=ds_val,
            epochs=cfg.epochs_tune,
            verbose=2,
            callbacks=callbacks,
        )

        val_loss = float(min(hist.history["val_loss"]))
        print(f"  best val_loss for {hp['name']}: {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_hp = hp

    print(f"\n[HP SEARCH] Best config: {best_hp['name']} (val_loss={best_loss:.6f})")
    return best_hp


def run_cross_validation(
    X: np.ndarray,
    y_scaled: np.ndarray,
    df: pd.DataFrame,
    y_mean: float,
    y_std: float,
    best_hp: Dict,
    max_len_with_cls: int,
    vocab_size: int,
    file_stem: str,
    cfg: ExperimentConfig,
) -> None:
    """
    5-fold cross-validation over the entire dataset using best_hp.
    """
    print("\n[CV] Running 5-fold cross validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    metrics_rows: List[Dict] = []
    preds_rows: List[Dict] = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        print(f"\n[CV] Fold {fold}/5")
        X_train_f = X[train_idx]
        X_val_f = X[val_idx]
        y_train_f = y_scaled[train_idx]
        y_val_f = y_scaled[val_idx]

        ds_train_f = make_ds(
            X_train_f, y_train_f[:, None], batch=cfg.batch, shuffle=True
        )
        ds_val_f = make_ds(X_val_f, y_val_f[:, None], batch=cfg.batch)

        steps_per_epoch = max(1, len(X_train_f) // cfg.batch)
        model = build_model_from_hp(
            best_hp,
            max_len_with_cls=max_len_with_cls,
            vocab_size=vocab_size,
            steps_per_epoch=steps_per_epoch,
            epochs=cfg.epochs,
            cfg=cfg,
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=100,
                restore_best_weights=True,
                verbose=1,
            )
        ]

        model.fit(
            ds_train_f,
            validation_data=ds_val_f,
            epochs=cfg.epochs,
            verbose=2,
            callbacks=callbacks,
        )

        ds_val_tokens = make_ds(X_val_f, batch=cfg.batch, shuffle=False)
        y_val_pred_scaled = model.predict(ds_val_tokens, verbose=0).reshape(-1)

        y_true = y_val_f * y_std + y_mean
        y_pred = y_val_pred_scaled * y_std + y_mean

        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        R = pearson_r(y_true, y_pred)
        delta_t95 = p95_width(y_true)
        residuals = y_pred - y_true
        delta_tr95 = p95_width(residuals)
        delta_tr95_pct = (
            float(100.0 * delta_tr95 / delta_t95) if delta_t95 > 0 else np.nan
        )
        ci_low, ci_high = residual_ci95(residuals)

        metrics_rows.append(
            dict(
                file=file_stem,
                fold=fold,
                n=len(val_idx),
                R=R,
                R2=r2,
                Dt95_min=delta_t95,
                Dtr95_pct=delta_tr95_pct,
                MAE_min=mae,
                MSE_min2=mse,
                RMSE_min=rmse,
                CI_low_min=ci_low,
                CI_high_min=ci_high,
            )
        )

        for local_i, global_idx in enumerate(val_idx):
            preds_rows.append(
                dict(
                    file=file_stem,
                    fold=fold,
                    index=int(global_idx),
                    sequence=df.iloc[global_idx]["sequence"],
                    rt_true_min=float(y_true[local_i]),
                    rt_pred_min=float(y_pred[local_i]),
                )
            )

    metrics_df = pd.DataFrame(metrics_rows).rename(
        columns={
            "R2": "R²",
            "Dt95_min": "Δt₉₅% (min)",
            "Dtr95_pct": "Δtr₉₅%",
            "MAE_min": "MAE (min)",
            "MSE_min2": "MSE (min²)",
            "RMSE_min": "RMSE (min)",
            "CI_low_min": "CI_low (min)",
            "CI_high_min": "CI_high (min)",
        }
    )

    out_dir = cfg.root
    out_dir.mkdir(parents=True, exist_ok=True)

    cv_metrics_path = out_dir / f"{file_stem}_cv_metrics.csv"
    metrics_df.to_csv(cv_metrics_path, index=False)
    print("[CV] Saved metrics →", cv_metrics_path)

    preds_df = pd.DataFrame(preds_rows).sort_values(["fold", "index"])
    cv_preds_path = out_dir / f"{file_stem}_test_predictions_cv.csv"
    preds_df.to_csv(cv_preds_path, index=False)
    print("[CV] Saved predictions →", cv_preds_path)


def train_one_file(
    path: Path,
    cfg: ExperimentConfig,
    load_df: DataFrameLoader,
) -> Dict:
    file_name = path.name
    file_stem = path.stem

    print(f"\n=== {file_name} ===")
    df = load_df(path)
    seqs = df["sequence"].tolist()

    y_minutes, detected = normalize_rt_units(
        df["rt"].values, unit_mode=cfg.unit_mode, sec_to_min_threshold=cfg.sec_to_min_threshold
    )
    print(f"[Unit] Detected/used units for metrics: {detected}")

    y_mean = float(y_minutes.mean())
    y_std_raw = y_minutes.std()
    y_std = float(y_std_raw if y_std_raw > 1e-6 else 1.0)
    y_scaled = ((y_minutes - y_mean) / y_std).astype(np.float32)

    alphabet = infer_alphabet(seqs)
    tok = build_tokenizer(alphabet)
    max_len = max(len(s) for s in seqs)
    max_len_with_cls = max_len + 1
    vocab_size = max(tok.values()) + 1

    X = np.stack([encode_sequence(s, tok, max_len) for s in seqs]).astype(np.int32)
    indices = np.arange(len(df))

    idx_tr, idx_te, X_tr, X_te, y_tr, y_te = train_test_split(
        indices,
        X,
        y_scaled,
        test_size=0.20,
        random_state=42,
    )

    if cfg.hp_search:
        X_tr_sub, X_val, y_tr_sub, y_val = train_test_split(
            X_tr, y_tr, test_size=0.20, random_state=123
        )
        best_hp = tune_hyperparams(
            X_tr_sub,
            y_tr_sub,
            X_val,
            y_val,
            max_len_with_cls=max_len_with_cls,
            vocab_size=vocab_size,
            cfg=cfg,
        )
    else:
        best_hp = {
            "name": "manual_default",
            "D_MODEL": cfg.d_model,
            "N_LAYERS": cfg.n_layers,
            "N_HEADS": cfg.n_heads,
            "D_FF": cfg.d_ff,
            "DROPOUT": cfg.dropout,
            "BASE_LR": cfg.base_lr,
        }
        print("\n[HP SEARCH] Disabled, using manual defaults:", best_hp)

    run_cross_validation(
        X,
        y_scaled,
        df,
        y_mean,
        y_std,
        best_hp,
        max_len_with_cls=max_len_with_cls,
        vocab_size=vocab_size,
        file_stem=file_stem,
        cfg=cfg,
    )

    steps_per_epoch = max(1, len(X_tr) // cfg.batch)
    model = build_model_from_hp(
        best_hp,
        max_len_with_cls=max_len_with_cls,
        vocab_size=vocab_size,
        steps_per_epoch=steps_per_epoch,
        epochs=cfg.epochs,
        cfg=cfg,
    )

    ds_tr_full = make_ds(X_tr, y_tr[:, None], batch=cfg.batch, shuffle=True)
    ds_va = make_ds(X_te, y_te[:, None], batch=cfg.batch)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=100,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    print("\n[TRAIN] Final model training with best hyperparameters...")
    history = model.fit(
        ds_tr_full,
        validation_data=ds_va,
        epochs=cfg.epochs,
        verbose=2,
        callbacks=callbacks,
    )

    out_dir = cfg.root
    out_dir.mkdir(parents=True, exist_ok=True)

    hist_df = pd.DataFrame(
        {
            "epoch": np.arange(1, len(history.history["loss"]) + 1),
            "loss": history.history["loss"],
            "val_loss": history.history["val_loss"],
        }
    )
    hist_path = out_dir / f"{file_stem}_train_history.csv"
    hist_df.to_csv(hist_path, index=False)
    print("[TRAIN] Saved train history →", hist_path)

    ds_tr_tokens_noshuf = make_ds(X_tr, batch=cfg.batch, shuffle=False)
    y_tr_pred_scaled = model.predict(ds_tr_tokens_noshuf, verbose=0).reshape(-1)
    y_tr_true = y_tr * y_std + y_mean
    y_tr_pred = y_tr_pred_scaled * y_std + y_mean

    ds_te_tokens = make_ds(X_te, batch=cfg.batch, shuffle=False)
    y_pred_scaled = model.predict(ds_te_tokens, verbose=0).reshape(-1)
    y_true = y_te * y_std + y_mean
    y_pred = y_pred_scaled * y_std + y_mean

    train_pred_df = pd.DataFrame(
        {
            "index": idx_tr,
            "sequence": df.iloc[idx_tr]["sequence"].values,
            "rt_true_min": y_tr_true.astype(float),
            "rt_pred_min": y_tr_pred.astype(float),
        }
    )
    train_pred_path = out_dir / f"{file_stem}_train_predictions.csv"
    train_pred_df.to_csv(train_pred_path, index=False)
    print("[PRED] Saved train predictions →", train_pred_path)

    val_pred_df = pd.DataFrame(
        {
            "index": idx_te,
            "sequence": df.iloc[idx_te]["sequence"].values,
            "rt_true_min": y_true.astype(float),
            "rt_pred_min": y_pred.astype(float),
        }
    )
    val_pred_path = out_dir / f"{file_stem}_validation_predictions.csv"
    val_pred_df.to_csv(val_pred_path, index=False)
    print("[PRED] Saved validation/test predictions →", val_pred_path)

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    R = pearson_r(y_true, y_pred)
    delta_t95 = p95_width(y_true)
    residuals = y_pred - y_true
    delta_tr95 = p95_width(residuals)
    delta_tr95_pct = (
        float(100.0 * delta_tr95 / delta_t95) if delta_t95 > 0 else np.nan
    )
    ci_low, ci_high = residual_ci95(residuals)

    print(f"Samples (n): {len(df)} | MaxLen: {max_len}")
    print(f"Best HP    : {best_hp['name']}")
    print(f"R        : {R:.6f}")
    print(f"R²       : {r2:.6f}")
    print(f"MAE      : {mae:.6f} min")
    print(f"MSE      : {mse:.6f} min^2")
    print(f"RMSE     : {rmse:.6f} min")
    print(f"Δt95%    : {delta_t95:.6f} min")
    print(
        f"Δtr95%   : {delta_tr95_pct:.3f} % "
        f"(absolute: {delta_tr95:.6f} min)"
    )
    print(
        f"95% CI (residuals, min): "
        f"[{ci_low:.6f}, {ci_high:.6f}]"
    )

    return dict(
        file=file_name,
        best_hp=best_hp["name"],
        n=len(df),
        max_len=max_len,
        R=R,
        R2=r2,
        Dt95_min=delta_t95,
        Dtr95_pct=delta_tr95_pct,
        MAE_min=mae,
        MSE_min2=mse,
        RMSE_min=rmse,
        CI_low_min=ci_low,
        CI_high_min=ci_high,
    )


def run_experiment(cfg: ExperimentConfig, load_df: DataFrameLoader) -> pd.DataFrame:
    """
    High-level helper:
      - loops over cfg.files within cfg.root
      - trains & evaluates
      - returns a summary DataFrame and writes CSV
    """
    set_seed(42)

    results: List[Dict] = []
    root = cfg.root
    root.mkdir(parents=True, exist_ok=True)

    for fname in cfg.files:
        path = root / fname
        if path.exists():
            results.append(train_one_file(path, cfg=cfg, load_df=load_df))
        else:
            print(f"Missing: {path}")

    res_df = pd.DataFrame(results).rename(
        columns={
            "best_hp": "best_hp",
            "R2": "R²",
            "Dt95_min": "Δt₉₅% (min)",
            "Dtr95_pct": "Δtr₉₅%",
            "MAE_min": "MAE (min)",
            "MSE_min2": "MSE (min²)",
            "RMSE_min": "RMSE (min)",
            "CI_low_min": "CI_low (min)",
            "CI_high_min": "CI_high (min)",
        }
    )

    res_df = res_df[
        [
            "file",
            "best_hp",
            "n",
            "max_len",
            "R",
            "R²",
            "Δt₉₅% (min)",
            "Δtr₉₅%",
            "MAE (min)",
            "MSE (min²)",
            "RMSE (min)",
            "CI_low (min)",
            "CI_high (min)",
        ]
    ]

    out_csv = root / "rt_transformer_metrics.csv"
    res_df.to_csv(out_csv, index=False)

    print("\nSaved metrics →", out_csv)
    print("─" * 80)
    print(res_df.to_string(index=False))
    print("─" * 80)

    return res_df


__all__ = [
    "make_ds",
    "tune_hyperparams",
    "run_cross_validation",
    "train_one_file",
    "run_experiment",
]

