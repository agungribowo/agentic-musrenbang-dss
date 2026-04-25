import os
import sys
from contextlib import nullcontext

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(PROJECT_ROOT)

import mlflow
import src.config.settings as settings
from src.tools.rag_engine import KamusRAG

class ClassifierAgent:
    """
    Agen AI pertama: Bertugas membaca keluhan warga dan memetakannya ke Kamus Usulan.
    Menggunakan Groq API untuk kecepatan inferensi tinggi.
    """
    def __init__(self, enable_mlflow=True):
        print("[Agen Klasifikasi] Membangunkan agen...")
        self.enable_mlflow = enable_mlflow
        
        # 1. Panggil pengaturan (Koneksi DagsHub & Set Client Groq)
        if self.enable_mlflow:
            settings.setup_mlflow_tracking()
        
        # 2. Ambil "Otak" (Client LLM) dari settings
        # Pastikan Anda menggunakan groq_client
        self.client = settings.groq_client
        self.model_name = settings.DEFAULT_LLM_MODEL
        
        # 3. Inisialisasi Ingatan (RAG)
        self.rag_tool = KamusRAG()

    def analyze(self, keluhan_warga, run_name="Uji_Coba_Klasifikasi"):
        print(f"\n[Agen Klasifikasi] Menerima kasus keluhan: '{keluhan_warga}'")

        # Kita gunakan blok try-except untuk menangkap error jika tidak ada eksperimen yang aktif
        if self.enable_mlflow:
            try:
                mlflow.set_experiment("Eksperimen_01_Agen_Classifier")
            except Exception:
                pass
        
        # Tambahkan nested=True agar aman dijalankan massal oleh skrip evaluator
        run_context = mlflow.start_run(run_name=run_name, nested=True) if self.enable_mlflow else nullcontext()
        with run_context:
            
            # TAHAP 1: Cari Referensi di Kamus (Top 3 teratas)
            top_k = 3
            referensi_kamus = self.rag_tool.search_kamus(keluhan_warga, top_k=top_k)

            # TAHAP 2: Menyusun Prompt
            PROMPT_VERSION = "v1.1_Groq_SystemPrompt"
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

            # TAHAP 3: Berpikir dan Mengambil Keputusan (Memakai format Groq/OpenAI)
            print("[Agen Klasifikasi] Sedang berpikir dan menyusun argumen birokrasi...")
            try:
                response = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model=self.model_name,
                    temperature=0.1, # Suhu rendah agar stabil dan tidak halusinasi
                )
                hasil_akhir = response.choices[0].message.content
                
                # --- TAMBAHAN CLAWWORK: Menangkap Token Langsung dari Groq ---
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                
                # Menghitung Biaya Simulasi
                cost_input = (prompt_tokens / 1_000_000) * settings.COST_PER_1M_INPUT_TOKENS
                cost_output = (completion_tokens / 1_000_000) * settings.COST_PER_1M_OUTPUT_TOKENS
                total_cost_usd = cost_input + cost_output
                
                # Bangun token info dictionary untuk di-return
                token_info = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_cost_usd": total_cost_usd
                }

                print("\n==========================================")
                print("KEPUTUSAN AGEN KLASIFIKASI")
                print("==========================================")
                print(hasil_akhir)
                print("==========================================")

                # TAHAP 4: Pencatatan Eksperimen ke DagsHub (Sesuai kode asli Anda)
                if self.enable_mlflow:
                    print("\n[Agen Klasifikasi] Menyimpan jejak penalaran ke DagsHub...")
                    mlflow.log_param("model_llm", self.model_name)
                    mlflow.log_param("rag_top_k", top_k)
                    mlflow.log_param("prompt_version", PROMPT_VERSION)
                    mlflow.log_param("api_vendor", "Groq")
                    
                    # Log Ekonomi
                    mlflow.log_metric("prompt_tokens", prompt_tokens)
                    mlflow.log_metric("completion_tokens", completion_tokens)
                    mlflow.log_metric("simulated_cost_usd", total_cost_usd)
                    
                    mlflow.log_text(keluhan_warga, "1_input_warga.txt")
                    mlflow.log_text(prompt, "2_prompt_lengkap.txt")
                    mlflow.log_text(hasil_akhir, "3_output_keputusan.txt")
                    
                    print("[Agen Klasifikasi] Selesai! Log tersimpan aman di Cloud.")
                else:
                    print("[Agen Klasifikasi] Mode no-mlflow aktif: log eksperimen dilewati.")
                
                # Mengembalikan tuple berisi teks, biaya, dan token info
                return hasil_akhir, total_cost_usd, token_info
                
            except Exception as e:
                error_msg = f"Gagal memproses klasifikasi via Groq API: {e}"
                print(f"[!] ERROR: {error_msg}")
                if self.enable_mlflow:
                    mlflow.log_param("error", str(e))
                return error_msg, 0.0, None

# Blok pengujian lokal (Tetap dibiarkan, tidak akan mengganggu)
if __name__ == "__main__":
    agen = ClassifierAgent()
    
    kasus = "Pak, jalan paving di gang RT 04 itu sudah hancur parah dan berlubang, kalau malam gelap gulita rawan ibu-ibu jatuh dari motor."

    hasil_text, biaya, token_info = agen.analyze(kasus, run_name="Tes_Jalan_Paving_Rusak")
    print(f"\n[LOCAL TEST] Estimasi biaya simulasi: ${biaya:.4f}")
    print(f"[LOCAL TEST] Panjang output: {len(str(hasil_text))} karakter")
    if token_info:
        print(f"[LOCAL TEST] Prompt tokens: {token_info.get('prompt_tokens', 0)}")
        print(f"[LOCAL TEST] Completion tokens: {token_info.get('completion_tokens', 0)}")