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
DAGSHUB_OWNER = os.getenv("DAGSHUB_REPO_OWNER")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO_NAME")

# ==========================================
# KONFIGURASI MODEL LLM (GLOBAL)
# ==========================================
# Otak Utama (Cloud - Gemini)
DEFAULT_LLM_MODEL = "gemini-2.0-flash"

# Otak Evaluator/Judge (Lokal - Ollama)
# Mengambil dari .env jika ada, jika tidak gunakan default localhost
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_JUDGE_MODEL = os.getenv("OLLAMA_JUDGE_MODEL", "llama3.1")

# ==========================================
# INISIALISASI KLIEN (SINGLETON)
# ==========================================
print("[System] Memuat konfigurasi utama...")

# 1. Inisialisasi Gemini API (SDK Baru)
if not GEMINI_API_KEY:
    raise ValueError("ERROR: GEMINI_API_KEY tidak ditemukan di file .env!")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
print("[-] Koneksi ke Gemini API: BERHASIL")

# 2. Fungsi Setup MLflow & DagsHub (Dipanggil manual oleh script yang membutuhkan tracking)
def setup_mlflow_tracking():
    if DAGSHUB_OWNER and DAGSHUB_REPO:
        try:
            dagshub.init(repo_owner=DAGSHUB_OWNER, repo_name=DAGSHUB_REPO, mlflow=True)
            tracking_uri = f"https://dagshub.com/{DAGSHUB_OWNER}/{DAGSHUB_REPO}.mlflow"
            mlflow.set_tracking_uri(tracking_uri)
            print(f"[-] Koneksi MLflow ke DagsHub ({DAGSHUB_REPO}): BERHASIL")
        except Exception as e:
            print(f"[!] Gagal menghubungkan ke DagsHub. Pesan error: {e}")
    else:
        print("[!] Peringatan: Kredensial DagsHub tidak lengkap. Eksperimen berjalan secara lokal.")