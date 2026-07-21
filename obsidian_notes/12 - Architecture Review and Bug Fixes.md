# 12 - Architecture Review and Bug Fixes

← Back to [[00 - MOC (Map of Content)]]

## 🌟 Kekuatan Utama Arsitektur (Strengths)

1. **Anti-Leakage Pipeline yang Ketat:**
   - Penggunaan `shift(1)` di seluruh `features.py` memastikan model tidak bisa melihat *close price* hari ini untuk memprediksi hari ini.
   - **Triple Barrier Method (De Prado):** Sangat unggul daripada klasifikasi biner (arah besok) karena metode ini memperhitungkan *stop-loss*, *take-profit*, dan batas waktu vertikal (horizon), sehingga label yang dihasilkan sangat mencerminkan realita *trading*.
   - Sanity Check di `main.py` (Akurasi > 60% memicu *warning*) adalah *mindset* kuantitatif yang sangat baik karena akurasi >60% pada data harian sering kali merupakan tanda kebocoran.

2. **Desain Gymnasium Environment (`trading_env.py`):**
   - **Differential Sharpe Ratio (DSR):** Penggunaan DSR (Moody et al., 1998) sebagai *reward function* memungkinkan agen PPO mendapatkan sinyal *reward* yang stabil dan *additive* di setiap *step*, lebih baik daripada Sharpe Ratio biasa.
   - **Multi-Layer Penalties:** Agen dihukum untuk *transaction cost*, *drawdown* (kuadratik setelah 2%), dan *overtrading* (>15% action rate). Ini memaksa agen untuk mencari probabilitas tinggi (*sniper*) ketimbang sering melakukan transaksi.

3. **Penggunaan RecurrentPPO (LSTM):**
   - Data finansial merupakan *partially observable environment*. Menambahkan memori berupa LSTM melalui *sequence length* (`sb3-contrib`) sangat krusial dibandingkan hanya menggunakan MLP standar.

4. **Purged Walk-Forward Validation & Monte Carlo:**
   - Implementasi *sliding window* dengan **Embargo 60 bar** menghindari *lookahead bias* akibat korelasi serial antar *fold*.
   - Uji Monte Carlo yang mensimulasikan berbagai kondisi pasar memberikan pandangan yang realistis terkait *tail-risk*.

---

## 🔧 Area untuk Perbaikan (Bugs & Issues to Fix)

### 1. "Double-Shift" (Lag Berlebihan) di Trading Environment
- **Masalah:** Di `features.py`, fitur sudah digeser dengan `shift(1)`. Namun, di `trading_env.py` pada fungsi `_get_obs()`, observasi diambil menggunakan indeks:
  ```python
  obs_step = max(0, self.current_step - 1)
  ```
  Ini menyebabkan agen yang berada di `step = t` mengambil data fitur `t-1` yang sebenarnya adalah informasi dari `t-2`. Agen mengalami lag 2 hari.
- **Solusi Tindakan:** Karena fitur sudah aman berkat *shifting* awal, perbarui logika pengambilan observasi menjadi:
  ```python
  obs_step = self.current_step
  ```

### 2. Hilangnya Embargo Gap di Skrip Walk-Forward (`train_wf.py`)
- **Masalah:** Meskipun skrip `train.py` memiliki fitur embargo pada pemisahan data (`split_data`), *loop* Walk-Forward di `train_wf.py` (sekitar baris 50) memisahkan train/test murni berdasarkan tahun (`year`):
  ```python
  df_train = df[(df['year'] >= start_year) & (df['year'] < train_end)]
  df_test = df[(df['year'] >= train_end) & (df['year'] < test_end)]
  ```
  Tidak ada *gap* embargo di sini.
- **Solusi Tindakan:** Tambahkan *gap* (misalnya dengan menyisihkan X hari terakhir di akhir rentang data `train` atau menunda awal data `test`) untuk mencegah data target yang menyilang tahun (mis. prediksi akhir Desember dipengaruhi harga awal Januari).

### 3. Konvergensi RL & Hyperparameter
- **Masalah:** PPO/RecurrentPPO dengan `50,000` atau `100,000` *timesteps* terlalu sedikit untuk *environment* yang bising (seperti Gold Futures harian), sehingga agen tidak dapat keluar dari fase eksplorasi yang acak.
- **Solusi Tindakan:** Skalakan nilai `RL_TIMESTEPS` hingga kisaran **500,000 - 2,000,000**. Pada fase-fase awal, *reward* akan terlihat buruk, namun kebijakan (policy) LSTM akan mulai terbentuk di ratusan ribu langkah berikutnya. Penggunaan alat *tuning* seperti Optuna untuk menyesuaikan koefisien entropi dan parameter *learning rate* sangat direkomendasikan.
