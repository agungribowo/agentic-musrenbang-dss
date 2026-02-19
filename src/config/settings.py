import os
from dotenv import load_dotenv
from google import genai
import dagshub
import mlflow

# Muat variabel dari file .env saat modul pertama kali diimpor
load_dotenv()

# ==========================================
# CREDENTIALS & ENVIRONMENT VARIABLES
# ==========================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Konfigurasi Ollama (Evaluator)
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_JUDGE_MODEL = os.getenv("OLLAMA_JUDGE_MODEL", "llama3.1")

# Konfigurasi DagsHub & MLflow
DAGSHUB_OWNER = os.getenv("DAGSHUB_REPO_OWNER")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO_NAME")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_USER_TOKEN")

# Konfigurasi LLM Utama (Cloud - Gemini)
DEFAULT_LLM_MODEL = "gemini-2.0-flash"

# ==========================================
# INISIALISASI KLIEN (SINGLETON)
# ==========================================
print("[System] Memuat konfigurasi utama...")

# 1. Inisialisasi Gemini API (SDK Baru)
gemini_client = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    print("[-] Koneksi ke Gemini API: BERHASIL")
else:
    print("[!] ERROR: GEMINI_API_KEY tidak ditemukan!")

# 2. Fungsi Setup MLflow & DagsHub
def setup_mlflow_tracking():
    """
    Menginisialisasi koneksi MLflow ke DagsHub. 
    Dipanggil secara eksplisit oleh skrip yang butuh tracking (seperti orchestrator & evaluator).
    """
    if DAGSHUB_OWNER and DAGSHUB_REPO:
        try:
            # Pastikan kredensial masuk ke environment variable agar MLflow bisa autentikasi
            os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_OWNER
            os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
            
            dagshub.init(repo_owner=DAGSHUB_OWNER, repo_name=DAGSHUB_REPO, mlflow=True)
            tracking_uri = f"https://dagshub.com/{DAGSHUB_OWNER}/{DAGSHUB_REPO}.mlflow"
            mlflow.set_tracking_uri(tracking_uri)
            print(f"[-] Koneksi MLflow ke DagsHub ({DAGSHUB_REPO}): BERHASIL")
        except Exception as e:
            print(f"[!] Gagal menghubungkan ke DagsHub: {e}")
    else:
        print("[!] Peringatan: Kredensial DagsHub tidak lengkap. Tracking berjalan lokal.")