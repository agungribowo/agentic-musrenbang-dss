import os
import sys

# ==========================================
# PENGATURAN PATH OTOMATIS
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(PROJECT_ROOT)

import mlflow
import src.config.settings as settings

class MitigationAgent:
    """
    Agen AI Kedua: Ahli Mitigasi Bencana dan Penilaian Risiko.
    Bertugas memberikan Skor Bahaya (1-10) untuk menentukan urgensi usulan.
    """
    def __init__(self):
        print("[Agen Mitigasi] Membangunkan agen analis risiko...")
        
        # 1. Panggil pengaturan koneksi
        settings.setup_environment()
        
        # 2. Ambil "Otak" (Client LLM) dari settings
        self.client = settings.gemini_client
        self.model_name = settings.DEFAULT_LLM_MODEL

    def analyze_risk(self, keluhan_warga, hasil_klasifikasi, run_name="Uji_Coba_Mitigasi"):
        print(f"\n[Agen Mitigasi] Mengevaluasi tingkat bahaya dari keluhan: '{keluhan_warga}'")

        # Membuat eksperimen baru khusus untuk Agen Mitigasi di MLflow
        mlflow.set_experiment("Eksperimen_02_Agen_Mitigasi")
        
        with mlflow.start_run(run_name=run_name):
            
            # TAHAP 1: Menyusun Prompt Evaluasi Risiko
            PROMPT_VERSION = "v1.0_RiskScoring"
            prompt = f"""
            Anda adalah 'Analis Mitigasi dan Risiko' untuk tim Musyawarah Perencanaan Pembangunan (Musrenbangkel).
            Tugas Anda adalah menilai seberapa DARURAT dan BERBAHAYA kondisi yang dilaporkan warga, sehingga layak diprioritaskan untuk didanai.

            DATA MASUKAN:
            Keluhan Warga: "{keluhan_warga}"
            Konteks Nomenklatur (Dari Agen Klasifikasi): 
            "{hasil_klasifikasi}"

            KRITERIA SKORING BAHAYA (1-10):
            - Skor 1-3 (Rendah): Tidak ada ancaman fisik/nyawa. Bersifat estetika atau pelengkap (contoh: minta pot bunga, gapura).
            - Skor 4-6 (Sedang): Mengganggu kenyamanan atau potensi kerugian ekonomi ringan (contoh: jalan sedikit berlubang, saluran air mampet tapi tidak banjir parah).
            - Skor 7-8 (Tinggi): Ada ancaman nyata terhadap kesehatan, keselamatan fisik, atau kerusakan harta benda (contoh: jalan gelap rawan kecelakaan, banjir masuk rumah).
            - Skor 9-10 (Kritis): Kondisi darurat mengancam nyawa, berpotensi memicu wabah, atau fasilitas vital lumpuh total (contoh: jembatan utama putus, gizi buruk akut/stunting masif).

            INSTRUKSI:
            Berikan evaluasi objektif dan keluarkan skor bahaya dalam format yang kaku di bawah ini.
            
            Format Jawaban:
            SKOR BAHAYA: [Angka 1-10]
            KATEGORI RISIKO: [Rendah / Sedang / Tinggi / Kritis]
            ANALISIS RISIKO: [Berikan alasan logis maksimal 3 kalimat mengapa skor tersebut diberikan berdasarkan kriteria di atas]
            """

            # TAHAP 2: Berpikir dan Mengambil Keputusan
            print("[Agen Mitigasi] Sedang mengkalkulasi matriks bahaya dan urgensi...")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            hasil_akhir = response.text

            print("\n==========================================")
            print("LAPORAN PENILAIAN RISIKO (AGEN MITIGASI)")
            print("==========================================")
            print(hasil_akhir)
            print("==========================================")

            # TAHAP 3: Pencatatan Eksperimen ke DagsHub
            print("\n[Agen Mitigasi] Menyimpan log skoring ke DagsHub...")
            mlflow.log_param("model_llm", self.model_name)
            mlflow.log_param("prompt_version", PROMPT_VERSION)
            
            # Kita catat masukannya
            mlflow.log_text(keluhan_warga, "1_input_warga.txt")
            mlflow.log_text(hasil_klasifikasi, "2_input_klasifikasi.txt")
            # Kita catat keputusannya
            mlflow.log_text(hasil_akhir, "3_output_skoring.txt")
            
            print("[Agen Mitigasi] Selesai! Log tersimpan aman di Cloud.")
            
            return hasil_akhir

# ==========================================
# PENGUJIAN AGEN MITIGASI
# ==========================================
if __name__ == "__main__":
    agen = MitigationAgent()
    
    # Kita gunakan kasus yang sama agar terlihat kesinambungannya
    kasus = "Pak, jalan paving di gang RT 04 itu sudah hancur parah dan berlubang, kalau malam gelap gulita rawan ibu-ibu jatuh dari motor."
    
    # Simulasi output dari Agen Klasifikasi yang sebelumnya
    klasifikasi_sebelumnya = """
    NOMENKLATUR TERPILIH: Peningkatan infrastruktur jalan dan drainase permukiman (lingkungan)
    DINAS TERKAIT: Perangkat Daerah pengampu Urusan Pemerintahan Bidang Perumahan Dan Kawasan Permukiman
    ALASAN PENALARAN: Keluhan secara spesifik merujuk pada kerusakan jalan paving di gang permukiman. Kondisi gelap rawan jatuh menunjukkan unsur bahaya keselamatan.
    """
    
    agen.analyze_risk(
        keluhan_warga=kasus, 
        hasil_klasifikasi=klasifikasi_sebelumnya, 
        run_name="Tes_Risiko_Jalan_Rusak"
    )