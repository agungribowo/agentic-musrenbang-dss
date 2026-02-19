import os
import sys
import argparse

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

DEFAULT_CASE_ID = "USULAN_RW11_001"
DEFAULT_CASE_TEXT = "Pak, tanggul sungai di batas RW 11 itu sudah retak panjang, kalau hujan deras airnya mulai rembes ke jalanan kampung. Warga takut kalau sampai jebol bisa merendam satu RW seperti tahun lalu."
QUICK_CASE_ID = "QUICK_CASE_001"
QUICK_CASE_TEXT = "Lampu penerangan jalan di gang RW 09 mati total selama seminggu dan warga khawatir rawan kecelakaan malam hari."

class MusrenbangOrchestrator:
    """
    Sistem Inti: Mengatur alur kerja Multi-Agent System.
    Menerima input warga, mengoperkannya ke 4 agen secara berurutan, 
    dan mencetak Laporan Eksekutif.
    """
    def __init__(self, enable_mlflow=True):
        self.enable_mlflow = enable_mlflow
        mode_labels = ["NORMAL"]
        if not self.enable_mlflow:
            mode_labels.append("NO-MLFLOW")

        print("\n=======================================================")
        print("MEMULAI SISTEM PENDUKUNG KEPUTUSAN (DSS) MUSRENBANGKEL")
        print("=======================================================")
        print(f"⚙️ [MODE] {' | '.join(mode_labels)}")
        
        # Inisialisasi semua agen (Mereka akan otomatis memuat model dan database RAG)
        self.agen_klasifikasi = ClassifierAgent(enable_mlflow=self.enable_mlflow)
        self.agen_mitigasi = MitigationAgent(enable_mlflow=self.enable_mlflow)
        self.agen_sosiologi = SociologyAgent(enable_mlflow=self.enable_mlflow)
        self.agen_ekonomi = EconomyAgent(enable_mlflow=self.enable_mlflow)
        
        # Otak utama untuk merangkum laporan akhir
        self.client = settings.groq_client
        self.model_name = settings.DEFAULT_LLM_MODEL

    @staticmethod
    def _normalize_classifier_output(classifier_result):
        """
        Kompatibilitas hasil `ClassifierAgent.analyze()`.
        Mendukung format lama (str) dan format baru (tuple: text, cost_usd).
        """
        if isinstance(classifier_result, tuple):
            hasil_text = str(classifier_result[0]) if len(classifier_result) > 0 else ""
            estimated_cost = classifier_result[1] if len(classifier_result) > 1 else None
            return hasil_text, estimated_cost

        return str(classifier_result), None

    def process_usulan(self, keluhan_warga, id_usulan="USULAN_001"):
        print(f"\n\n[ORKESTRATOR] Memproses Dokumen {id_usulan}")
        print(f"[INPUT WARGA] : '{keluhan_warga}'\n")
        
        # Menyiapkan variabel untuk menampung laporan staf
        waktu_mulai = time.time()

        # ---------------------------------------------------------
        # TAHAP 1: KLASIFIKASI KAMUS USULAN
        # ---------------------------------------------------------
        print(">>> Meneruskan ke Agen Klasifikasi...")
        classifier_result = self.agen_klasifikasi.analyze(
            keluhan_warga, run_name=f"{id_usulan}_Klasifikasi"
        )
        hasil_klasifikasi, biaya_klasifikasi = self._normalize_classifier_output(classifier_result)
        if biaya_klasifikasi is not None:
            print(f"[ORKESTRATOR] Estimasi biaya klasifikasi: ${biaya_klasifikasi:.4f}")

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

        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt_kesimpulan,
                }
            ],
            model=self.model_name,
            temperature=0.1,
        )
        kesimpulan_final = response.choices[0].message.content
        
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
        if self.enable_mlflow:
            mlflow.set_experiment("Eksperimen_00_Sistem_Utama_Orchestrator")
            with mlflow.start_run(run_name=f"{id_usulan}_Final_Report"):
                mlflow.log_text(kesimpulan_final, "Laporan_Eksekutif_Final.txt")
        else:
            print("[ORKESTRATOR] Mode no-mlflow aktif: log final tidak dikirim ke DagsHub.")

def parse_args():
    parser = argparse.ArgumentParser(description="Orchestrator Multi-Agent Musrenbang DSS")
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Matikan semua tracking MLflow/DagsHub untuk eksekusi lokal."
    )
    parser.add_argument(
        "--quick-case",
        action="store_true",
        help="Gunakan 1 kasus singkat bawaan untuk uji operasional harian yang lebih cepat."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Inisialisasi Sistem
    sistem_dss = MusrenbangOrchestrator(enable_mlflow=not args.no_mlflow)

    if args.quick_case:
        print("[ORKESTRATOR] QUICK-CASE aktif: menggunakan kasus singkat bawaan.")
        kasus_warga = QUICK_CASE_TEXT
        id_usulan = QUICK_CASE_ID
    else:
        kasus_warga = DEFAULT_CASE_TEXT
        id_usulan = DEFAULT_CASE_ID

    sistem_dss.process_usulan(kasus_warga, id_usulan=id_usulan)