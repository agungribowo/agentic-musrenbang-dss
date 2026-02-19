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

1. Input usulan warga masuk ke sistem.
2. `ClassifierAgent` melakukan retrieval RAG + pemetaan nomenklatur.
3. `MitigationAgent` memberi skor bahaya/urgensi.
4. `SociologyAgent` menilai dampak sosial berbasis profil lokal.
5. `EconomyAgent` mengestimasi skala anggaran dan kelayakan.
6. Orchestrator menyusun kesimpulan eksekutif.
7. Evaluator (`run_auto_evaluation.py`) mengaudit output dengan LLM judge.

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

### A) Jalankan orchestrator multi-agent

```powershell
python .\main_orchestrator.py
```

Mode operasi:

```powershell
python .\main_orchestrator.py --no-mlflow
python .\main_orchestrator.py --quick-case
python .\main_orchestrator.py --quick-case --no-mlflow
```

### B) Jalankan evaluasi massal

```powershell
python .\run_auto_evaluation.py
```

Mode evaluasi cepat:

```powershell
python .\run_auto_evaluation.py --dry-run --sample-size 5
python .\run_auto_evaluation.py --dry-run --no-mlflow --sample-size 5
```

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
