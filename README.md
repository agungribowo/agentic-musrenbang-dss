# Agentic Musrenbang DSS

Sistem pendukung keputusan berbasis **Multi-Agent + RAG + LLM-as-a-Judge** untuk analisis usulan Musrenbang.
Proyek ini dirancang untuk:

- memetakan keluhan/usulan warga ke nomenklatur kamus usulan,
- menilai urgensi risiko, dampak sosial, dan kelayakan anggaran,
- menyusun kesimpulan eksekutif lintas agen,
- serta menjalankan evaluasi massal yang dapat dipantau di MLflow/DagsHub.

---

## Fitur Utama

- **Multi-Agent Pipeline**: `classifier`, `mitigation`, `sociology`, `economy`.
- **RAG Retrieval**: pencarian semantik dari kamus usulan menggunakan ChromaDB.
- **Cloud LLM + Local Judge**: reasoning di Groq, evaluasi kualitas di Ollama.
- **Mode Operasional Fleksibel**:
	- `--dry-run` (tanpa API eksternal),
	- `--no-mlflow` (tanpa log ke DagsHub/MLflow),
	- `--quick-case` (uji cepat orchestrator).
- **Cost Tracking Simulatif**: estimasi biaya token pada agen klasifikasi.

---

## Arsitektur Sistem

### 1) Retrieval Layer (RAG)

- **Vector Database**: ChromaDB (`PersistentClient`, folder `chroma_db/`)
- **Collection**: `musrenbang_kamus_id`
- **Embedding Model**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Fungsi**: mengubah dokumen kamus + query menjadi embedding untuk semantic search.

### 2) Reasoning Layer (LLM Utama)

- **Model**: `llama-3.3-70b-versatile` (Groq, `DEFAULT_LLM_MODEL`)
- **Fungsi**: analisis utama pada setiap agen dan sintesis rekomendasi.

### 3) Evaluation Layer (Judge)

- **Model**: `llama3.1` (Ollama, `OLLAMA_JUDGE_MODEL`)
- **Fungsi**: menilai kualitas output agen (akurasi, reasoning, feedback).

### Ringkasan Peran Model (Presentasi)

- `paraphrase-multilingual-MiniLM-L12-v2` → **Embedding model** (retrieval, non-generatif).
- `llama-3.3-70b-versatile` → **Model generatif utama** (reasoning/analisis).
- `llama3.1` → **Model evaluator/judge** (audit kualitas output).

---

## Alur Proses

```text
[Input Usulan Warga]
	|
	v
[ClassifierAgent]
RAG Retrieval (ChromaDB + Embedding)
+ Pemetaan Nomenklatur
	|
	v
[MitigationAgent]
Skor Bahaya / Urgensi
	|
	v
[SociologyAgent]
Skor Dampak Sosial
	|
	v
[EconomyAgent]
Skala Anggaran + Kelayakan Finansial
	|
	v
[Main Orchestrator]
Kesimpulan Eksekutif
	|
	+----------------------------+
				     v
			  [Auto Evaluation Pipeline]
			  run_auto_evaluation.py
			  Judge Model (Ollama llama3.1)
```

---

## Struktur Proyek

```text
Agentic-Musrenbang-DSS/
├── src/
│   ├── agents/
│   │   ├── classifier_agent.py     # Agen klasifikasi + estimasi biaya token
│   │   ├── mitigation_agent.py     # Agen penilaian risiko/bahaya
│   │   ├── sociology_agent.py      # Agen analisis dampak sosial
│   │   └── economy_agent.py        # Agen estimasi skala anggaran
│   ├── tools/
│   │   └── rag_engine.py           # RAG engine (ChromaDB + embedding)
│   ├── config/
│   │   └── settings.py             # Konfigurasi model, API, MLflow/DagsHub
│   └── utils/
│       └── logger.py               # Utilitas logging internal
├── data/
│   ├── external/                   # Dataset sumber (CSV musrenbang/renstramas)
│   ├── processed/                  # Data hasil proses (jika digunakan)
│   └── raw/                        # Data mentah
├── chroma_db/                      # Persisted vector store ChromaDB
├── notebooks/
│   └── 01_eda_data_warga.ipynb     # EDA awal data warga
├── main_orchestrator.py            # Entry point sistem multi-agent end-to-end
├── run_auto_evaluation.py          # Evaluasi massal otomatis (Groq + Ollama judge)
├── run_evaluation.py               # Evaluasi skenario/benchmark tambahan
├── setup.ps1                       # Bootstrap environment proyek di PowerShell
├── requirements.txt                # Daftar dependency Python
├── laporan_audit_agen.csv          # Output/rekap audit evaluasi
└── README.md                       # Dokumentasi utama proyek
```

---

## Setup Cepat

### 1) Siapkan environment

Untuk Windows PowerShell, jalankan:

```powershell
./setup.ps1
```

Opsi penting:

```powershell
./setup.ps1 -SkipInstall
./setup.ps1 -Recreate
./setup.ps1 -VenvDir .venv -RequirementsFile requirements.txt
```

### 2) Konfigurasi `.env`

Pastikan variabel berikut tersedia:

- `GROQ_API_KEY`
- `OLLAMA_API_URL`
- `OLLAMA_JUDGE_MODEL`
- `DAGSHUB_REPO_OWNER`
- `DAGSHUB_REPO_NAME`
- `DAGSHUB_USER_TOKEN`

> Catatan keamanan: jangan commit API key/token asli ke repository publik.

---

## Menjalankan Sistem

| Command | Tujuan |
|---|---|
| `python .\main_orchestrator.py` | Menjalankan pipeline multi-agent penuh (normal mode). |
| `python .\main_orchestrator.py --quick-case` | Uji cepat orchestrator dengan 1 kasus bawaan. |
| `python .\main_orchestrator.py --no-mlflow` | Menjalankan orchestrator tanpa logging ke MLflow/DagsHub. |
| `python .\main_orchestrator.py --quick-case --no-mlflow` | Demo operasional cepat, minim noise logging. |
| `python .\run_auto_evaluation.py` | Menjalankan evaluasi massal default (dengan API eksternal). |
| `python .\run_auto_evaluation.py --dry-run --sample-size 5` | Evaluasi cepat tanpa call API eksternal, sample terbatas. |
| `python .\run_auto_evaluation.py --dry-run --no-mlflow --sample-size 5` | Smoke test paling ringan: tanpa API eksternal dan tanpa logging MLflow. |
| `python .\run_auto_evaluation.py --dry-run --sampling-mode random --seed 42 --sample-size 10` | Evaluasi dengan sampel acak yang reproducible. |
| `python .\run_auto_evaluation.py --dry-run --sampling-mode stratified_rw --sample-size 10` | Evaluasi dengan distribusi sampel lintas RW (lebih representatif wilayah). |
| `python .\run_auto_evaluation.py --dry-run --sampling-mode stratified_kamus --sample-size 10` | Evaluasi dengan distribusi sampel lintas kategori usulan (`KAMUS USULAN`). |

### Opsi Sampling Evaluasi

- `--dry-run-mode`:
	- `static`: skor simulasi tetap (`accuracy=10`, `reasoning=9`).
	- `stochastic`: skor simulasi variatif namun reproducible mengikuti `--seed`.
- `--stochastic-profile` (shortcut preset rentang stochastic):
	- `conservative` → akurasi `4-8`, reasoning `3-7`
	- `balanced` → akurasi `5-10`, reasoning `4-10` (default)
	- `aggressive` → akurasi `8-10`, reasoning `7-10`
- Rentang skor saat `--dry-run-mode stochastic`:
	- `--stochastic-acc-min` / `--stochastic-acc-max`
	- `--stochastic-reasoning-min` / `--stochastic-reasoning-max`
	- Semua nilai harus di rentang `1..10`.
	- Jika Anda isi nilai manual, nilai tersebut akan override preset profile.
- `--sampling-mode`:
	- `head`: ambil baris teratas (baseline cepat, paling berisiko bias urutan data).
	- `random`: acak dari seluruh data.
	- `stratified_rw`: acak berstrata berdasarkan kolom `RW`.
	- `stratified_kamus`: acak berstrata berdasarkan kolom `KAMUS USULAN`.
- `--seed`: seed random untuk reproduksibilitas hasil sampling.

Contoh uji fairness realistis tanpa API eksternal:

```powershell
python .\run_auto_evaluation.py --dry-run --dry-run-mode stochastic --sampling-mode stratified_kamus --sample-size 20 --seed 42 --no-mlflow
```

Contoh skenario konservatif (cenderung skor rendah):

```powershell
python .\run_auto_evaluation.py --dry-run --dry-run-mode stochastic --stochastic-acc-min 4 --stochastic-acc-max 8 --stochastic-reasoning-min 3 --stochastic-reasoning-max 7 --sampling-mode stratified_kamus --sample-size 20 --seed 42 --no-mlflow
```

Contoh command ringkas berbasis preset profile:

```powershell
python .\run_auto_evaluation.py --dry-run --dry-run-mode stochastic --stochastic-profile aggressive --sampling-mode stratified_kamus --sample-size 20 --seed 42 --no-mlflow
```

Contoh skenario agresif (cenderung skor tinggi):

```powershell
python .\run_auto_evaluation.py --dry-run --dry-run-mode stochastic --stochastic-acc-min 8 --stochastic-acc-max 10 --stochastic-reasoning-min 7 --stochastic-reasoning-max 10 --sampling-mode stratified_kamus --sample-size 20 --seed 42 --no-mlflow
```

### Audit Fairness di MLflow

Saat evaluasi berjalan dengan MLflow aktif, sistem akan menulis ringkasan fairness pada run `KESIMPULAN_RATA_RATA_EVALUASI`:

- **Parameter**: `sampling_mode`, `sampling_seed`, `sampling_strata_col`.
- **Metric komposisi strata**:
	- `sample_strata_unique`
	- `sample_count_<strata_col>_<stratum>`
- **Metric fairness kualitas**:
	- `fairness_<strata_col>_<stratum>_pass_rate_pct`
	- `fairness_<strata_col>_<stratum>_passed`
	- `fairness_<strata_col>_<stratum>_failed`
	- `fairness_<strata_col>_<stratum>_total`
	- `fairness_gap_pass_rate_pct`
	- `fairness_alert` (0/1)
- **Threshold fairness alert**:
	- `fairness_alert_threshold_pct` (default: `20.0`)
	- Rule: `fairness_alert=true` jika `fairness_gap_pass_rate_pct > fairness_alert_threshold_pct`.
- **Artifact**:
	- `sampling/strata_composition_<strata_col>.json`
	- `sampling/fairness_summary_<strata_col>.json`
	- `sampling/fairness_dashboard_<strata_col>.md`

### Validation Threshold (Quality Gate)
Sistem menggunakan pendekatan *Soft Threshold* pada fase PoC ini untuk menentukan status "Lulus" atau "Gagal" dari setiap usulan warga berdasarkan skor dari `OLLAMA_JUDGE_MODEL`. Konstanta ini didefinisikan secara terpusat di `src/config/settings.py`:

- `VALIDATION_THRESHOLD_ACCURACY` (Default: 8.0)
- `VALIDATION_THRESHOLD_REASONING` (Default: 7.0)
- `VALIDATION_THRESHOLD_RELEVANCE` (Default: 7.0)

	Jika skor dari agen berada di bawah ambang batas ini, usulan tersebut akan dicatat sebagai `validation_pass = 0` (Gagal) di dalam MLflow, yang kemudian digunakan untuk menghitung persentase *Fairness Pass Rate*.


### Demo Checklist (Pre-flight Check)

Sebelum live demo, pastikan 5 poin ini aman:

1. **Interpreter benar**
	- Terminal aktif di root project dan menggunakan Python dari `.venv`.

2. **Environment siap**
	- Jalankan `./setup.ps1 -SkipInstall` minimal sekali sebelum sesi.

3. **Mode demo aman dipilih**
	- Untuk presentasi non-teknis, gunakan `--quick-case` + `--no-mlflow` agar cepat dan minim noise.

4. **Dependensi lokal evaluator aman**
	- Jika hanya demo alur, gunakan `--dry-run` (tidak butuh call API Groq/Ollama).

5. **Rencana fallback siap**
	- Jika jaringan/limit bermasalah, langsung pindah ke command dry-run agar demo tetap jalan.

### Demo Script 3 Menit (Live Presentasi)

Gunakan urutan berikut agar demo stabil, cepat, dan minim risiko error saat tampil:

1. **Aktifkan environment proyek**

```powershell
./setup.ps1 -SkipInstall
```

2. **Jalankan demo orchestrator super cepat (tanpa noise logging)**

```powershell
python .\main_orchestrator.py --quick-case --no-mlflow
```

3. **Jalankan demo evaluasi cepat 1 sampel (tanpa API eksternal)**

```powershell
python .\run_auto_evaluation.py --dry-run --no-mlflow --sample-size 1
```

4. **(Opsional, jika ingin tunjukkan mode produksi)**

```powershell
python .\main_orchestrator.py --quick-case
```

> Rekomendasi presenter: jalankan langkah 1 sebelum sesi dimulai, lalu tampilkan langkah 2 dan 3 saat live demo.

### Script Moderator (60–90 detik per command)

Gunakan naskah berikut saat live demo agar alur presentasi konsisten dan mudah diikuti audiens non-teknis.

#### Command 1 — Setup environment

```powershell
./setup.ps1 -SkipInstall
```

**Narasi (±60 detik):**
"Di langkah pertama, saya memastikan environment proyek aktif dan konsisten. Script ini menyiapkan virtual environment project, jadi semua dependency dan interpreter yang dipakai sesuai konfigurasi tim. Kenapa ini penting? Karena banyak error demo biasanya muncul hanya karena interpreter salah atau package terbaca dari environment lain. Dengan langkah ini, kita kunci fondasi eksekusi agar stabil sebelum masuk ke pipeline AI."

#### Command 2 — Orchestrator quick demo

```powershell
python .\main_orchestrator.py --quick-case --no-mlflow
```

**Narasi (±90 detik):**
"Sekarang kita jalankan orchestrator dalam mode quick-case supaya demonstrasi cepat dan fokus ke alur inti. Sistem akan memproses satu kasus usulan warga menggunakan rangkaian multi-agent: klasifikasi, mitigasi risiko, analisis sosial, dan estimasi ekonomi. Saya juga mengaktifkan no-mlflow agar output demo bersih, tanpa noise logging ke dashboard. Jadi yang audiens lihat benar-benar value proses pengambilan keputusan AI, dari input usulan sampai kesimpulan eksekutif."

#### Command 3 — Evaluasi cepat (tanpa API eksternal)

```powershell
python .\run_auto_evaluation.py --dry-run --no-mlflow --sample-size 1
```

**Narasi (±75 detik):**
"Di langkah terakhir, saya tunjukkan jalur evaluasi. Mode dry-run dipakai agar tidak tergantung call API eksternal, jadi aman untuk kondisi jaringan apa pun saat presentasi. Dengan sample-size 1, proses selesai cepat tetapi tetap merepresentasikan alur audit kualitas output model. No-mlflow tetap aktif agar sesi demo tidak mencemari eksperimen produksi. Ini menunjukkan bahwa sistem punya mode operasional yang fleksibel: bisa untuk demo cepat, debugging, maupun mode produksi penuh."

#### Fallback line (jika ada kendala live)

**Narasi singkat cadangan (±20 detik):**
"Kalau jaringan atau layanan eksternal sedang tidak stabil, kita tetap bisa validasi seluruh alur dengan mode dry-run, sehingga demo tetap berjalan dan objektif menampilkan pipeline sistem."

---

## Catatan Operasional

- Gunakan `--dry-run` untuk verifikasi pipeline tanpa biaya API eksternal.
- Gunakan `--no-mlflow` saat debugging agar dashboard DagsHub tidak noisy.
- Saat mode normal, evaluasi massal menggunakan jeda antar request untuk mengurangi risiko rate limit.

### Change Log Harian (Append Otomatis)

Gunakan script berikut agar entry changelog selalu **append** (bukan overwrite):

```powershell
./append_change_log.ps1 -Summary "Update evaluasi stochastic" -Scope "Evaluation pipeline" -Files "run_auto_evaluation.py,README.md" -Details "Tambah profile preset;Update docs" -VerifyCommand "python .\run_auto_evaluation.py --dry-run --sample-size 5" -VerifyResult "Sukses" -Notes "Session demo"
```

Catatan format input:
- `-Files` dipisahkan koma.
- `-Details` dipisahkan titik koma (`;`).

---

## Troubleshooting Singkat

- **`ModuleNotFoundError`**
	- Pastikan command dijalankan dari Python `.venv` project.
- **Judge tidak merespon**
	- Pastikan Ollama aktif dan model `OLLAMA_JUDGE_MODEL` sudah tersedia lokal.
- **Log ke DagsHub gagal**
	- Cek kredensial `DAGSHUB_*` di `.env`, atau jalankan dengan `--no-mlflow`.

---

## Status

Proyek aktif dikembangkan untuk kebutuhan eksperimen dan operasional Musrenbang berbasis AI agentic.
