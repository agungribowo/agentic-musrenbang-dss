import os
import sys
from contextlib import nullcontext

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(PROJECT_ROOT)

import mlflow
import src.config.settings as settings

class EconomyAgent:
    """
    Agen AI Keempat: Estimator Skala Anggaran.
    Bertugas mengklasifikasikan usulan ke dalam skala Mikro, Menengah, atau Makro,
    serta memberikan Skor Kelayakan Finansial (1-10).
    """
    def __init__(self, enable_mlflow=True):
        print("[Agen Ekonomi] Membangunkan agen perencana anggaran...")
        self.enable_mlflow = enable_mlflow
        if self.enable_mlflow:
            settings.setup_mlflow_tracking()
        self.client = settings.groq_client
        self.model_name = settings.DEFAULT_LLM_MODEL

    def analyze_budget(self, keluhan_warga, hasil_klasifikasi, analisis_sosiologi, run_name="Uji_Coba_Ekonomi"):
        print(f"\n[Agen Ekonomi] Menghitung proksi anggaran untuk keluhan: '{keluhan_warga}'")

        if self.enable_mlflow:
            mlflow.set_experiment("Eksperimen_04_Agen_Ekonomi")

        run_context = mlflow.start_run(run_name=run_name) if self.enable_mlflow else nullcontext()
        with run_context:
            
            PROMPT_VERSION = "v1.0_BudgetScaling"
            prompt = f"""
            Anda adalah 'Estimator Anggaran Publik' untuk Musrenbangkel tingkat Kelurahan.
            Tugas Anda adalah memprediksi kompleksitas pembiayaan dari sebuah usulan warga. Anda TIDAK perlu menghitung nominal pasti, cukup berikan estimasi skalanya.

            DATA MASUKAN:
            Keluhan Warga: "{keluhan_warga}"
            Konteks Nomenklatur: "{hasil_klasifikasi}"
            Dampak Sosial (Dari Agen Sebelumnya): "{analisis_sosiologi}"

            KRITERIA SKALA ANGGARAN:
            1. MIKRO: Biaya sangat murah, pengerjaan cepat, tidak butuh alat berat/tender. (Contoh: Beli pot bunga, alat tulis posyandu, honor narasumber pelatihan, patroli keamanan warga).
            2. MENENGAH: Butuh material bahan bangunan, tenaga tukang, atau pengadaan barang elektronik. (Contoh: Pavingisasi gang kecil, perbaikan selokan mampet, pengadaan komputer balai RT).
            3. MAKRO: Proyek infrastruktur raksasa yang butuh alat berat, pembebasan lahan, atau di luar wewenang kelurahan. (Contoh: Pengaspalan jalan raya, pembangunan jembatan beton antar wilayah, normalisasi sungai besar).

            KRITERIA SKOR KELAYAKAN FINANSIAL (1-10) UNTUK DANA KELURAHAN:
            - Skor 8-10: Sangat layak didanai kelurahan (Anggaran Mikro/Menengah tapi dampak sosialnya tinggi). Ini adalah "Quick Win" (Kemenangan Cepat).
            - Skor 4-7: Layak dipertimbangkan (Anggaran sepadan dengan dampaknya).
            - Skor 1-3: Tidak layak pakai dana kelurahan (Skala Makro yang terlalu mahal, atau anggaran Mikro tapi dampaknya sangat sempit/pribadi).

            INSTRUKSI:
            Evaluasi skala usulan tersebut dan berikan metrik anggaran.
            
            Format Jawaban:
            SKALA ANGGARAN: [Mikro / Menengah / Makro]
            SKOR KELAYAKAN FINANSIAL: [Angka 1-10]
            ANALISIS ANGGARAN: [Berikan alasan maksimal 3 kalimat. Sebutkan mengapa usulan ini masuk kategori Mikro/Menengah/Makro dan nilai perbandingan antara perkiraan biaya dengan dampak sosialnya]
            """

            print("[Agen Ekonomi] Sedang menghitung rasio Cost-Benefit (Biaya vs Manfaat)...")
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model_name,
                temperature=0.1,
            )
            hasil_akhir = response.choices[0].message.content

            print("\n==========================================")
            print("LAPORAN ESTIMASI ANGGARAN (AGEN EKONOMI)")
            print("==========================================")
            print(hasil_akhir)
            print("==========================================")

            if self.enable_mlflow:
                print("\n[Agen Ekonomi] Menyimpan metrik finansial ke DagsHub...")
                mlflow.log_param("model_llm", self.model_name)
                mlflow.log_param("prompt_version", PROMPT_VERSION)
                
                mlflow.log_text(keluhan_warga, "1_input_warga.txt")
                mlflow.log_text(analisis_sosiologi, "2_input_sosiologi.txt")
                mlflow.log_text(hasil_akhir, "3_output_anggaran.txt")
                
                print("[Agen Ekonomi] Selesai! Log tersimpan aman di Cloud.")
            else:
                print("[Agen Ekonomi] Mode no-mlflow aktif: log eksperimen dilewati.")
            
            return hasil_akhir

if __name__ == "__main__":
    agen = EconomyAgent()
    
    # Kita lanjutkan kasus kos-kosan/narkoba yang mendapat skor Sosiologi 9 tadi
    kasus = "Banyak warga luar yang ngekos di RW 09 sering bawa teman ke kamar sampai larut malam dan dicurigai transaksi narkoba, warga resah minta diadakan patroli dan pembinaan keamanan."
    
    klasifikasi = "NOMENKLATUR TERPILIH: Fasilitasi penanganan konflik sosial."
    
    # Simulasi output dari Agen Sosiologi (Berdampak luas)
    sosiologi = "KATEGORI: Sistemik Isu Strategis. Mengatasi masalah keamanan kos dan narkoba sesuai Renstramas."
    
    agen.analyze_budget(
        keluhan_warga=kasus, 
        hasil_klasifikasi=klasifikasi, 
        analisis_sosiologi=sosiologi,
        run_name="Tes_Anggaran_Patroli_Kos"
    )