import os
from dotenv import load_dotenv
from google import genai
import dagshub
import mlflow

# Muat variabel dari file .env
load_dotenv()

# Ambil variabel lingkungan
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DAGSHUB_OWNER = os.getenv("DAGSHUB_REPO_OWNER")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO_NAME")

# Definisi nama model global agar konsisten di seluruh agen
DEFAULT_LLM_MODEL = "gemini-2.5-flash"

def setup_environment():
    global gemini_client
    print("Memulai inisialisasi sistem...")

    # 1. Validasi dan Konfigurasi Gemini API (SDK Baru)
    if not GEMINI_API_KEY:
        raise ValueError("ERORR: GEMINI_API_KEY tidak ditemukan di file .env!")
    
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    print("[-] Koneksi ke Gemini API (SDK Baru): BERHASIL")

    # 2. Konfigurasi MLflow & DagsHub
    if DAGSHUB_OWNER and DAGSHUB_REPO:
        try:
            dagshub.init(repo_owner=DAGSHUB_OWNER, repo_name=DAGSHUB_REPO, mlflow=True)
            tracking_uri = f"https://dagshub.com/{DAGSHUB_OWNER}/{DAGSHUB_REPO}.mlflow"
            mlflow.set_tracking_uri(tracking_uri)
            print(f"[-] Koneksi MLflow ke DagsHub ({DAGSHUB_REPO}): BERHASIL")
        except Exception as e:
            print(f"[!] Gagal menghubungkan ke DagsHub. Pesan error: {e}")
    else:
        print("[!] Peringatan: Kredensial DagsHub tidak lengkap. Eksperimen tidak akan tercatat di cloud.")

if __name__ == "__main__":
    setup_environment()