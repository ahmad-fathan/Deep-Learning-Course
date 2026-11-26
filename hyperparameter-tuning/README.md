# 08 — Hyperparameter Tuning

Contoh paling sederhana hyperparameter tuning dengan TensorFlow/Keras: **Random Search** pada model MLP untuk dataset `digits`.

## Isi

| Bagian modul | Diwujudkan sebagai |
|---|---|
| Peran validation set (3.3) | split 60/20/20 stratified: train / val / test |
| Random Search (4.3) | 12 trial, tiap trial ambil konfigurasi acak dari `space` |
| Sampling skala logaritmik (4.4) | `lr = 10 ** random.uniform(-4, -1.5)` |
| Early stopping (5.1) | `EarlyStopping(monitor="val_accuracy", patience=5)` — epoch tak perlu di-tuning |
| Siklus tuning (3.2) | skor validasi dipakai memilih; test set dievaluasi sekali di akhir |

Hyperparameter yang di-tuning: `lr`, `units`, `dropout`, `batch_size`.

## Menjalankan

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python hyperparameter_tuning.py
```

Keluaran (TF 2.21.0, Apple Silicon, ±16 detik):

```
trial  1  val_acc=0.955  epochs=29  lr=0.0040 units=32  dropout=0.37 bs=16
trial  3  val_acc=0.981  epochs=33  lr=0.0032 units=64  dropout=0.36 bs=64
trial  9  val_acc=0.986  epochs=26  lr=0.0008 units=128 dropout=0.18 bs=64
...
best params: {lr=0.00082, units=128, dropout=0.17764, batch_size=64}
best val_acc: 0.986
test_acc: 0.975
```

## Catatan

- `tf.keras.utils.set_random_seed()` **jangan** dipanggil di dalam loop trial: ia me-reset modul `random` juga, sehingga semua trial mendapat konfigurasi identik. Gunakan `tf.random.set_seed()`.
- Test set hanya dipakai sekali, setelah tuning selesai, agar skor yang dilaporkan tidak terlalu optimistis.
- Untuk mengganti ke Optuna (Bagian 6), cukup pindahkan isi `space` ke `trial.suggest_*` dan bungkus `train()` sebagai `objective`.
