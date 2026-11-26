# Hyperparameter Tuning

Dua contoh hyperparameter tuning dengan TensorFlow/Keras pada model MLP untuk dataset `digits`. Keduanya memakai model, data, dan pembagian train/validation/test yang sama, sehingga hasilnya bisa dibandingkan langsung.

## Notebook

| Notebook | Isi | Buka |
|---|---|---|
| `hyperparameter_tuning.ipynb` | Random Search manual, 12 trial, 3 hyperparameter | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahmad-fathan/Deep-Learning-Course/blob/main/hyperparameter-tuning/hyperparameter_tuning.ipynb) |
| `hyperparameter_tuning_optuna.ipynb` | Optuna: sampler TPE + pruner Hyperband | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahmad-fathan/Deep-Learning-Course/blob/main/hyperparameter-tuning/hyperparameter_tuning_optuna.ipynb) |

Hyperparameter yang dituning: `lr` (skala logaritmik), `units`, `dropout`. `batch_size` tetap 32 pada kedua notebook, dan jumlah epoch tidak dicari karena early stopping / pruning.

## Menjalankan

**Colab** — klik badge di atas, lalu *Runtime → Run all*. Notebook Optuna meng-install `optuna` sendiri di sel pertama.

**Lokal**

```bash
python3.11 -m venv .venv
.venv/bin/pip install jupyter tensorflow scikit-learn optuna
.venv/bin/jupyter lab hyperparameter_tuning.ipynb
```

## Keluaran

`hyperparameter_tuning.ipynb` (±17 detik):

```
trial  5  val_acc=0.983  lr=0.0011 units=128 dropout=0.08
trial  9  val_acc=0.100  lr=0.0290 units=64  dropout=0.43
best: {'lr': 0.00114, 'units': 128, 'dropout': 0.07774} val_acc=0.983
test_acc: 0.975
```

`hyperparameter_tuning_optuna.ipynb` (±38 detik):

```
jumlah trial selesai : 7
jumlah trial dipangkas: 5
best params: {'lr': 0.0013292918943162175, 'units': 256, 'dropout': 0.36599697090570255}
best val_acc: 0.983
test_acc: 0.981
```

## Catatan

- `tf.keras.utils.set_random_seed()` **jangan** dipanggil di dalam loop trial: ia me-reset modul `random` juga, sehingga semua trial mendapat konfigurasi identik. Pakai `tf.random.set_seed()`.
- Test set hanya dipakai sekali, setelah tuning selesai.
- Pada versi Optuna, `trial.report()` + `trial.should_prune()` per epoch membuat Hyperband dapat menghentikan trial yang tertinggal — 5 dari 12 trial dipangkas.
