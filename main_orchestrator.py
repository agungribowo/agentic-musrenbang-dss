import os
import sys

# Memastikan Python bisa membaca seluruh modul di dalam folder src/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

import time
import mlflow
import src.config.settings as settings

# Mengimpor Keempat Staf Ahli (Agen AI)
from src.agents.classifier_agent import ClassifierAgent
from src.agents.mitigation_agent import MitigationAgent
from src.agents.sociology_agent import SociologyAgent
from src.agents.economy_agent import EconomyAgent

class MusrenbangOrchestrator:
    """
    Sistem Inti: Mengatur alur kerja Multi-Agent System.
    Menerima input warga, mengoperkannya ke 4 agen secara berurutan, 
    dan mencetak Laporan Eksekutif.
    """
    def __init__(self):
        print("\n=======================================================")
        print("MEMULAI SISTEM PENDUKUNG KEPUTUSAN (DSS) MUSRENBANGKEL")
        print("=======================================================")
        
        # Inisialisasi semua agen (Mereka akan otomatis memuat model dan database RAG)
        self.agen_klasifikasi = ClassifierAgent()
        self.agen_mitigasi = MitigationAgent()
        self.agen_sosiologi = SociologyAgent()
        self.agen_ekonomi = EconomyAgent()
        
        # Otak utama untuk merangkum laporan akhir
        self.client = settings.gemini_client
        self.model_name = settings.DEFAULT_LLM_MODEL

    def process_usulan(self, keluhan_warga, id_usulan="USULAN_001"):
        print(f"\n\n[ORKESTRATOR] Memproses Dokumen {id_usulan}")
        print(f"[INPUT WARGA] : '{keluhan_warga}'\n")
        
        # Menyiapkan variabel untuk menampung laporan staf
        waktu_mulai = time.time()

        # ---------------------------------------------------------
        # TAHAP 1: KLASIFIKASI KAMUS USULAN
        # ---------------------------------------------------------
        print(">>> Meneruskan ke Agen Klasifikasi...")
        hasil_klasifikasi = self.agen_klasifikasi.analyze(
            keluhan_warga, run_name=f"{id_usulan}_Klasifikasi"
        )

        # ---------------------------------------------------------
        # TAHAP 2: PENILAIAN RISIKO & BAHAYA
        # ---------------------------------------------------------
        print(">>> Meneruskan ke Agen Mitigasi (Risiko)...")
        hasil_mitigasi = self.agen_mitigasi.analyze_risk(
            keluhan_warga, hasil_klasifikasi, run_name=f"{id_usulan}_Mitigasi"
        )

        # ---------------------------------------------------------
        # TAHAP 3: PENILAIAN DAMPAK SOSIAL LOKAL (NUSUKAN)
        # ---------------------------------------------------------
        print(">>> Meneruskan ke Agen Sosiologi...")
        hasil_sosiologi = self.agen_sosiologi.analyze_social_impact(
            keluhan_warga, hasil_klasifikasi, hasil_mitigasi, run_name=f"{id_usulan}_Sosiologi"
        )

        # ---------------------------------------------------------
        # TAHAP 4: ESTIMASI SKALA ANGGARAN
        # ---------------------------------------------------------
        print(">>> Meneruskan ke Agen Ekonomi...")
        hasil_ekonomi = self.agen_ekonomi.analyze_budget(
            keluhan_warga, hasil_klasifikasi, hasil_sosiologi, run_name=f"{id_usulan}_Ekonomi"
        )

        # ---------------------------------------------------------
        # TAHAP 5: KESIMPULAN EKSEKUTIF (HAKIM FINAL)
        # ---------------------------------------------------------
        print("\n[ORKESTRATOR] Semua staf telah melapor. Menyusun Laporan Eksekutif...")
        
        prompt_kesimpulan = f"""
        Anda adalah Ketua Bappeda yang tegas dan analitis.
        Tugas Anda adalah merangkum laporan dari 4 staf ahli Anda menjadi satu paragraf KESIMPULAN EKSEKUTIF yang akan dibaca oleh Wali Kota.

        Data Keluhan Awal: "{keluhan_warga}"
        
        Ringkasan Laporan Staf:
        1. Birokrasi: {hasil_klasifikasi}
        2. Risiko: {hasil_mitigasi}
        3. Sosial: {hasil_sosiologi}
        4. Anggaran: {hasil_ekonomi}

        INSTRUKSI:
        Buatlah 1 paragraf (maksimal 4 kalimat) yang menyimpulkan apakah usulan ini SANGAT PRIORITAS, PRIORITAS MENENGAH, atau DITOLAK. 
        Gunakan gaya bahasa birokrasi pemerintahan yang profesional. Sebutkan secara singkat skor risikonya dan kelayakan anggarannya.
        """

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt_kesimpulan
        )
        kesimpulan_final = response.text
        
        waktu_selesai = time.time()
        durasi = round(waktu_selesai - waktu_mulai, 2)

        # ==========================================
        # CETAK LAPORAN AKHIR
        # ==========================================
        print("\n" + "="*60)
        print(f"📄 DOKUMEN HASIL ANALISIS MULTI-AGENT DSS (Waktu Proses: {durasi} detik)")
        print("="*60)
        print(f"KODE USULAN   : {id_usulan}")
        print(f"KELUHAN WARGA : {keluhan_warga}")
        print("-"*60)
        print(hasil_klasifikasi.strip())
        print("-"*60)
        print(hasil_mitigasi.strip())
        print("-"*60)
        print(hasil_sosiologi.strip())
        print("-"*60)
        print(hasil_ekonomi.strip())
        print("="*60)
        print("📌 KESIMPULAN EKSEKUTIF PIMPINAN:")
        print(kesimpulan_final.strip())
        print("="*60)

        # Menyimpan hasil akhir sistem ke DagsHub
        mlflow.set_experiment("Eksperimen_00_Sistem_Utama_Orchestrator")
        with mlflow.start_run(run_name=f"{id_usulan}_Final_Report"):
            mlflow.log_text(kesimpulan_final, "Laporan_Eksekutif_Final.txt")

if __name__ == "__main__":
    # Inisialisasi Sistem
    sistem_dss = MusrenbangOrchestrator()
    
    # KASUS UJI COBA TERAKHIR KITA
    # Mari kita uji dengan kasus yang "abu-abu" (Penting tapi mungkin butuh biaya mahal)
    kasus_warga = "Pak, tanggul sungai di batas RW 11 itu sudah retak panjang, kalau hujan deras airnya mulai rembes ke jalanan kampung. Warga takut kalau sampai jebol bisa merendam satu RW seperti tahun lalu."
    
    sistem_dss.process_usulan(kasus_warga, id_usulan="USULAN_RW11_001")