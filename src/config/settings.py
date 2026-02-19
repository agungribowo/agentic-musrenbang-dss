import os
from dotenv import load_dotenv
import dagshub
import mlflow
from groq import Groq

# Muat variabel dari file .env saat modul pertama kali diimpor
load_dotenv()

# ==========================================
# CREDENTIALS & ENVIRONMENT VARIABLES
# ==========================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Konfigurasi Ollama (Evaluator)
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_JUDGE_MODEL = os.getenv("OLLAMA_JUDGE_MODEL", "llama3.1")

# Konfigurasi DagsHub & MLflow
DAGSHUB_OWNER = os.getenv("DAGSHUB_REPO_OWNER")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO_NAME")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_USER_TOKEN")

# Konfigurasi LLM Utama (Cloud - Groq)
# Kita menggunakan Llama 3.3 70B, model open-source terpintar dan tercepat dari Groq
DEFAULT_LLM_MODEL = "llama-3.3-70b-versatile"
_TRACKING_SETUP_DONE = False

# ==========================================
# SIMULATED COST TRACKING (CLAWWORK LOGIC)
# ==========================================
# Estimasi biaya untuk model Llama-3.3-70b-versatile di Groq (dalam USD)
COST_PER_1M_INPUT_TOKENS = 0.59
COST_PER_1M_OUTPUT_TOKENS = 0.79

# ==========================================
# INISIALISASI KLIEN (SINGLETON)
# ==========================================
print("[System] Memuat konfigurasi utama (Groq Engine)...")

# 1. Inisialisasi Groq API
groq_client = None
if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("[-] Koneksi ke Groq API: BERHASIL")
    except Exception as e:
        print(f"[!] Gagal inisialisasi Groq API: {e}")
else:
    print("[!] ERROR: GROQ_API_KEY tidak ditemukan di .env!")

# 2. Fungsi Setup MLflow & DagsHub
def setup_mlflow_tracking():
    """
    Menginisialisasi koneksi MLflow ke DagsHub. 
    Dipanggil secara eksplisit oleh skrip yang butuh tracking (seperti orchestrator & evaluator).
    """
    global _TRACKING_SETUP_DONE
    if _TRACKING_SETUP_DONE:
        return

    if DAGSHUB_OWNER and DAGSHUB_REPO:
        try:
            # Pastikan kredensial masuk ke environment variable agar MLflow bisa autentikasi
            os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_OWNER
            os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
            
            dagshub.init(repo_owner=DAGSHUB_OWNER, repo_name=DAGSHUB_REPO, mlflow=True)
            tracking_uri = f"https://dagshub.com/{DAGSHUB_OWNER}/{DAGSHUB_REPO}.mlflow"
            mlflow.set_tracking_uri(tracking_uri)
            print(f"[-] Koneksi MLflow ke DagsHub ({DAGSHUB_REPO}): BERHASIL")
            _TRACKING_SETUP_DONE = True
        except Exception as e:
            print(f"[!] Gagal menghubungkan ke DagsHub: {e}")
    else:
        print("[!] Peringatan: Kredensial DagsHub tidak lengkap. Tracking berjalan lokal.")
        _TRACKING_SETUP_DONE = True

def setup_environment():
    """Backward-compatible setup untuk modul lama."""
    setup_mlflow_tracking()