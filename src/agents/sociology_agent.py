import os
import sys
from contextlib import nullcontext

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(PROJECT_ROOT)

import mlflow
import src.config.settings as settings

class SociologyAgent:
    """
    Agen AI Ketiga: Ahli Sosiologi dan Perencanaan Wilayah.
    Bertugas memberikan Skor Dampak Sosial (1-10) berdasarkan Profil Demografi Kelurahan.
    """
    def __init__(self, enable_mlflow=True):
        print("[Agen Sosiologi] Membangunkan agen analis dampak sosial...")
        self.enable_mlflow = enable_mlflow
        if self.enable_mlflow:
            settings.setup_mlflow_tracking()
        self.client = settings.groq_client
        self.model_name = settings.DEFAULT_LLM_MODEL

        # ==========================================
        # CONTEXT GROUNDING: PROFIL KELURAHAN
        # Di versi produksi nanti, teks ini bisa dibaca dari file .txt atau database.
        # Untuk eksperimen ini, kita letakkan di sini agar mudah dibaca AI.
        # ==========================================
        self.profil_kelurahan = """
        PROFIL KELURAHAN NUSUKAN (2026-2030):
        1. Geografi: Kepadatan sangat tinggi (59m2/orang). Wilayah Rawan Bencana: RW 7, RW 8, RW 9, RW 11, RW 13, RW 18.
        2. Demografi Rentan: Terdapat populasi Lansia (>60 tahun) sekitar 4.224 jiwa, dan Balita (0-4 thn) 1.867 jiwa.
        3. Isu Strategis Keamanan: Banyak rumah kos/kontrakan disalahgunakan untuk asusila, narkoba, sarang teroris. Siskamling tidak maksimal. Rawan penipuan/gendam.
        4. Isu Strategis Ekonomi: 6.526 jiwa belum bekerja, 3.427 mengurus rumah tangga. SDM kurang wawasan, lemah modal, daya saing UKM rendah, dampak pandemi.
        """

    def analyze_social_impact(self, keluhan_warga, hasil_klasifikasi, skor_mitigasi, run_name="Uji_Coba_Sosiologi"):
        print(f"\n[Agen Sosiologi] Mengevaluasi luasan dampak dari keluhan: '{keluhan_warga}'")

        if self.enable_mlflow:
            mlflow.set_experiment("Eksperimen_03_Agen_Sosiologi")

        run_context = mlflow.start_run(run_name=run_name) if self.enable_mlflow else nullcontext()
        with run_context:
            
            PROMPT_VERSION = "v1.0_SocialImpactScoring"
            prompt = f"""
            Anda adalah 'Analis Dampak Sosial dan Kependudukan' untuk Musrenbangkel.
            Tugas Anda adalah menilai seberapa LUAS manfaat dari usulan ini dan apakah usulan ini memecahkan ISU STRATEGIS kelurahan.

            DATA PROFIL LOKAL KELURAHAN:
            {self.profil_kelurahan}

            DATA USULAN MASUK:
            Keluhan Warga: "{keluhan_warga}"
            Konteks Nomenklatur: "{hasil_klasifikasi}"
            Laporan Risiko Sebelumnya: "{skor_mitigasi}"

            KRITERIA SKORING DAMPAK SOSIAL (1-10):
            - Skor 1-3 (Dampak Sempit): Hanya menguntungkan 1 individu atau keluarga tertentu (Contoh: bedah rumah pribadi, modal usaha perorangan).
            - Skor 4-6 (Dampak Lingkungan RT): Menguntungkan satu gang atau RT, perbaikan infrastruktur minor yang tidak tercantum dalam isu strategis mendesak.
            - Skor 7-8 (Dampak Skala RW / Kelompok Rentan): Menguntungkan banyak warga di tingkat RW, ATAU melindungi kelompok rentan (Lansia/Balita/Ibu), ATAU berada di area Rawan Bencana yang disebutkan di profil.
            - Skor 9-10 (Dampak Sistemik / Isu Strategis): Menyelesaikan masalah utama kelurahan (Keamanan Kos/Kontrakan, Pengangguran Masif, Pemberdayaan UKM, Kepadatan Penduduk Ekstrem).

            INSTRUKSI:
            Analisis keluhan warga dengan merujuk LANGSUNG pada 'DATA PROFIL LOKAL KELURAHAN' di atas.
            
            Format Jawaban:
            SKOR SOSIOLOGI: [Angka 1-10]
            KATEGORI DAMPAK: [Individu / Skala RT / Skala RW & Rentan / Sistemik Isu Strategis]
            ANALISIS SOSIAL: [Berikan alasan logis maksimal 3 kalimat. Wajib menyebutkan korelasi antara usulan dengan data Profil Kelurahan Nusukan]
            """

            print("[Agen Sosiologi] Sedang membedah data demografi dan sebaran manfaat...")
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
            print("LAPORAN DAMPAK SOSIAL (AGEN SOSIOLOGI)")
            print("==========================================")
            print(hasil_akhir)
            print("==========================================")

            if self.enable_mlflow:
                print("\n[Agen Sosiologi] Menyimpan jejak analisis ke DagsHub...")
                mlflow.log_param("model_llm", self.model_name)
                mlflow.log_param("prompt_version", PROMPT_VERSION)
                
                mlflow.log_text(keluhan_warga, "1_input_warga.txt")
                mlflow.log_text(self.profil_kelurahan, "2_input_profil_lokal.txt")
                mlflow.log_text(hasil_akhir, "3_output_skor_sosial.txt")
                
                print("[Agen Sosiologi] Selesai! Log tersimpan aman di Cloud.")
            else:
                print("[Agen Sosiologi] Mode no-mlflow aktif: log eksperimen dilewati.")
            
            return hasil_akhir

if __name__ == "__main__":
    agen = SociologyAgent()
    
    # KASUS UJI COBA (Kita gunakan contoh yang memicu Isu Keamanan Nusukan)
    kasus = "Banyak warga luar yang ngekos di RW 09 sering bawa teman ke kamar sampai larut malam dan dicurigai transaksi narkoba, warga resah minta diadakan patroli dan pembinaan keamanan."
    
    # Simulasi output agen sebelumnya
    klasifikasi = "NOMENKLATUR TERPILIH: Fasilitasi penanganan konflik sosial. DINAS TERKAIT: Kesatuan Bangsa Dan Politik."
    mitigasi = "SKOR BAHAYA: 7. KATEGORI: Tinggi. Potensi kerawanan sosial dan tindak kriminal."
    
    agen.analyze_social_impact(
        keluhan_warga=kasus, 
        hasil_klasifikasi=klasifikasi, 
        skor_mitigasi=mitigasi,
        run_name="Tes_Isu_Keamanan_Kos"
    )