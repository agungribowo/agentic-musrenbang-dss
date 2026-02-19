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

---

## Catatan Operasional

- Gunakan `--dry-run` untuk verifikasi pipeline tanpa biaya API eksternal.
- Gunakan `--no-mlflow` saat debugging agar dashboard DagsHub tidak noisy.
- Saat mode normal, evaluasi massal menggunakan jeda antar request untuk mengurangi risiko rate limit.

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
