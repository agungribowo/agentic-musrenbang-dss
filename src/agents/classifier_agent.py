import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(PROJECT_ROOT)

import mlflow
import src.config.settings as settings
from src.tools.rag_engine import KamusRAG

class ClassifierAgent:
    """
    Agen AI pertama: Bertugas membaca keluhan warga dan memetakannya ke Kamus Usulan.
    Menggunakan SDK 'google-genai' terbaru.
    """
    def __init__(self):
        print("[Agen Klasifikasi] Membangunkan agen...")
        
        # 1. Panggil pengaturan (Koneksi DagsHub & Set Client Gemini)
        settings.setup_mlflow_tracking()
        
        # 2. Ambil "Otak" (Client LLM) dari settings
        self.client = settings.gemini_client
        self.model_name = settings.DEFAULT_LLM_MODEL
        
        # 3. Inisialisasi Ingatan (RAG)
        self.rag_tool = KamusRAG()

    def analyze(self, keluhan_warga, run_name="Uji_Coba_Klasifikasi"):
        print(f"\n[Agen Klasifikasi] Menerima kasus keluhan: '{keluhan_warga}'")

        mlflow.set_experiment("Eksperimen_01_Agen_Classifier")
        
        with mlflow.start_run(run_name=run_name):
            
            # TAHAP 1: Cari Referensi di Kamus (Top 3 teratas)
            top_k = 3
            referensi_kamus = self.rag_tool.search_kamus(keluhan_warga, top_k=top_k)

            # TAHAP 2: Menyusun Prompt
            PROMPT_VERSION = "v1.0_SystemPrompt"
            prompt = f"""
            Anda adalah 'Agen Klasifikasi Musrenbangkel' yang ahli dalam tata kelola birokrasi pemerintahan Indonesia.
            Tugas Anda adalah memetakan keluhan warga yang menggunakan bahasa sehari-hari ke dalam SATU nomenklatur resmi yang paling tepat.

            Keluhan Warga: "{keluhan_warga}"

            {referensi_kamus}

            Instruksi:
            1. Pilih SATU 'Usulan' dari kandidat di atas yang secara semantik paling relevan dengan keluhan warga.
            2. Berikan alasan analitis yang logis (maksimal 3 kalimat) mengapa Anda memilih nomenklatur tersebut.
            3. Jika keluhan warga mengandung unsur bahaya/darurat, tolong sebutkan juga di alasan Anda.

            Format Jawaban:
            NOMENKLATUR TERPILIH: [Nama Usulan]
            DINAS TERKAIT: [Nama Dinas]
            ALASAN PENALARAN: [Alasan logis Anda]
            """

            # TAHAP 3: Berpikir dan Mengambil Keputusan (Memakai SDK Baru)
            print("[Agen Klasifikasi] Sedang berpikir dan menyusun argumen birokrasi...")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            hasil_akhir = response.text

            print("\n==========================================")
            print("KEPUTUSAN AGEN KLASIFIKASI")
            print("==========================================")
            print(hasil_akhir)
            print("==========================================")

            # TAHAP 4: Pencatatan Eksperimen ke DagsHub
            print("\n[Agen Klasifikasi] Menyimpan jejak penalaran ke DagsHub...")
            mlflow.log_param("model_llm", self.model_name)
            mlflow.log_param("rag_top_k", top_k)
            mlflow.log_param("prompt_version", PROMPT_VERSION)
            
            mlflow.log_text(keluhan_warga, "1_input_warga.txt")
            mlflow.log_text(prompt, "2_prompt_lengkap.txt")
            mlflow.log_text(hasil_akhir, "3_output_keputusan.txt")
            
            print("[Agen Klasifikasi] Selesai! Log tersimpan aman di Cloud.")
            
            return hasil_akhir

if __name__ == "__main__":
    agen = ClassifierAgent()
    
    # Keluhan uji coba
    kasus = "Pak, jalan paving di gang RT 04 itu sudah hancur parah dan berlubang, kalau malam gelap gulita rawan ibu-ibu jatuh dari motor."
    
    agen.analyze(kasus, run_name="Tes_Jalan_Paving_Rusak")