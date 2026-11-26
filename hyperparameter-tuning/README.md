# Hyperparameter Tuning

Contoh paling sederhana hyperparameter tuning dengan TensorFlow/Keras: **Random Search** pada MLP untuk dataset `digits`.

[![Buka di Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahmad-fathan/Deep-Learning-Course/blob/main/hyperparameter-tuning/hyperparameter_tuning.ipynb)

## Isi

Notebook tunggal `hyperparameter_tuning.ipynb`:

1. Impor & seed
2. Split train/validation/test
3. Fungsi `train()` — satu kali training + early stopping
4. Random Search 12 trial atas `lr` (log-scale), `units`, `dropout`
5. Uji sekali pada test set
6. Latihan

## Menjalankan

- **Colab**: klik badge di atas, lalu *Runtime → Run all*. TensorFlow dan scikit-learn sudah tersedia.
- **Lokal**:

```bash
python3.11 -m venv .venv
.venv/bin/pip install jupyter tensorflow scikit-learn
.venv/bin/jupyter lab hyperparameter_tuning.ipynb
```

Keluaran (±17 detik, TF 2.21.0):

```
trial  5  val_acc=0.983  lr=0.0011 units=128 dropout=0.08
trial  9  val_acc=0.100  lr=0.0290 units=64  dropout=0.43   <- lr terlalu besar, divergen
trial 10  val_acc=0.975  lr=0.0008 units=128 dropout=0.18
...
best: {'lr': 0.00114, 'units': 128, 'dropout': 0.07774} val_acc=0.983
test_acc: 0.975
```

## Catatan

- `tf.keras.utils.set_random_seed()` **jangan** dipanggil di dalam loop trial: ia me-reset modul `random` juga, sehingga semua trial mendapat konfigurasi identik. Pakai `tf.random.set_seed()`.
- Test set hanya dipakai sekali, setelah tuning selesai.
- `batch_size` tetap 32 dan jumlah epoch tidak dituning karena early stopping.
