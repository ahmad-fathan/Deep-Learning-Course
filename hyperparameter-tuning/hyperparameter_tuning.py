"""Contoh hyperparameter tuning paling sederhana (Bab 8) - TensorFlow/Keras.

Random Search atas 3 hyperparameter MLP pada dataset digits.
Data dibagi train/val/test: val untuk memilih konfigurasi, test dipakai sekali di akhir.
Early stopping dipasang supaya jumlah epoch tidak perlu ikut di-tuning.

    pip install tensorflow scikit-learn
    python hyperparameter_tuning.py
"""

import os
import random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # redam log info TF

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 42
MAX_EPOCHS, PATIENCE = 40, 5
N_TRIALS = 12


def build(units, dropout):
    return tf.keras.Sequential([
        tf.keras.layers.Input((64,)),
        tf.keras.layers.Dense(units, activation="relu"),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(units // 2, activation="relu"),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])


def train(cfg, x_tr, y_tr, x_va, y_va):
    """Latih satu model, kembalikan (val_accuracy terbaik, model, jumlah epoch)."""
    tf.random.set_seed(SEED)
    model = build(cfg["units"], cfg["dropout"])
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg["lr"]),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    hist = model.fit(
        x_tr, y_tr,
        validation_data=(x_va, y_va),
        epochs=MAX_EPOCHS, batch_size=cfg["batch_size"], verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=PATIENCE,
            restore_best_weights=True)],
    )
    return max(hist.history["val_accuracy"]), model, len(hist.history["loss"])


def main():
    random.seed(SEED)
    digits = load_digits()
    x, y = digits.data.astype("float32"), digits.target.astype("int32")
    x_tr, x_tmp, y_tr, y_tmp = train_test_split(x, y, test_size=0.4, random_state=SEED, stratify=y)
    x_va, x_te, y_va, y_te = train_test_split(x_tmp, y_tmp, test_size=0.5, random_state=SEED, stratify=y_tmp)

    # Search space. lr disampel pada skala logaritmik: 10**U(-4, -1.5) (Bagian 4.4)
    space = {
        "lr": lambda: 10 ** random.uniform(-4, -1.5),
        "units": lambda: random.choice([32, 64, 128, 256]),
        "dropout": lambda: random.uniform(0.0, 0.5),
        "batch_size": lambda: random.choice([16, 32, 64]),
    }

    best_cfg, best_score, results = None, -1.0, []
    for trial in range(1, N_TRIALS + 1):
        cfg = {k: f() for k, f in space.items()}
        acc, _, n_epoch = train(cfg, x_tr, y_tr, x_va, y_va)
        results.append((acc, cfg))
        print(f"trial {trial:2d}  val_acc={acc:.3f}  epochs={n_epoch:2d}  "
              f"lr={cfg['lr']:.4f} units={cfg['units']} "
              f"dropout={cfg['dropout']:.2f} bs={cfg['batch_size']}")
        if acc > best_score:
            best_cfg, best_score = cfg, acc

    pretty = {k: round(v, 5) if isinstance(v, float) else v for k, v in best_cfg.items()}
    print(f"\nbest params: {pretty}")
    print(f"best val_acc: {best_score:.3f}")

    # Latih ulang dengan seluruh data train+val, lalu uji SEKALI pada test set.
    x_all, y_all = np.concatenate([x_tr, x_va]), np.concatenate([y_tr, y_va])
    tf.random.set_seed(SEED)
    model = build(best_cfg["units"], best_cfg["dropout"])
    model.compile(optimizer=tf.keras.optimizers.Adam(best_cfg["lr"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_all, y_all, epochs=MAX_EPOCHS, batch_size=best_cfg["batch_size"], verbose=0)
    test_acc = model.evaluate(x_te, y_te, verbose=0)[1]
    print(f"test_acc: {test_acc:.3f}   (test set dipakai sekali, sesudah tuning selesai)")

    assert len(results) == N_TRIALS and max(r[0] for r in results) == best_score
    assert test_acc > 0.85, f"test_acc terlalu rendah: {test_acc:.3f}"


if __name__ == "__main__":
    main()
