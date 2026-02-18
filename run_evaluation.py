import pandas as pd
import mlflow
import time # Kunci untuk mengatasi Error 429
import src.config.settings as settings
from src.agents.mitigation_agent import MitigationAgent

# Inisialisasi
settings.setup_environment()
client = settings.gemini_client
model_judge = settings.DEFAULT_LLM_MODEL 

agen_terdakwa = MitigationAgent()

def llm_judge_audit(keluhan, hasil_agen, kriteria_rubrik):
    prompt_hakim = f"""
    Anda adalah AUDITOR KUALITAS AI yang objektif.
    TUGAS: Nilai kinerja Agen Mitigasi berdasarkan keluhan warga dan output yang dia hasilkan.
    
    INPUT:
    1. Keluhan Warga: "{keluhan}"
    2. Output Agen Mitigasi: "{hasil_agen}"
    3. Rubrik Acuan: "{kriteria_rubrik}"
    
    PERTANYAAN AUDIT:
    1. Apakah Agen memberikan skor yang masuk akal sesuai rubrik?
    2. Apakah alasan Agen logis dan langsung menjawab keluhan?
    
    FORMAT JAWABAN JSON:
    {{
        "score_accuracy": 5,
        "reasoning_quality": 5,
        "audit_note": "Tuliskan komentar singkat maksimal 1 kalimat."
    }}
    """
    
    try:
        response = client.models.generate_content(
            model=model_judge,
            contents=prompt_hakim,
            # Memaksa output JSON
            config={'response_mime_type': 'application/json'} 
        )
        return response.parsed # Otomatis menjadi dictionary Python
    except Exception as e:
        print(f"\n[!] Gagal memanggil Hakim: {e}")
        # Jika gagal, kembalikan nilai default agar program tidak crash
        return {"score_accuracy": 0, "reasoning_quality": 0, "audit_note": "Gagal karena limitasi API."}

test_cases = [
    {
        "kasus": "Pak, ada pot bunga pecah di depan balai RW, tolong diganti biar cantik.",
        "klasifikasi_konteks": "Pemeliharaan taman"
    },
    {
        "kasus": "Jalan berlubang sedikit di gang buntu, tidak membahayakan tapi kurang nyaman.",
        "klasifikasi_konteks": "Pemeliharaan jalan lingkungan"
    },
    {
        "kasus": "Tanggul sungai jebol, air mulai masuk rumah setinggi lutut, arus deras, banyak lansia terjebak.",
        "klasifikasi_konteks": "Penanggulangan bencana alam"
    }
]

print("\n[AUDITOR] Memulai Audit Kinerja Agen Mitigasi...")
results = []

mlflow.set_experiment("Eksperimen_99_Evaluasi_Judge")

with mlflow.start_run(run_name="Batch_Audit_02_Dengan_Jeda"):
    for i, data in enumerate(test_cases):
        print(f"\n==============================")
        print(f"--- Audit Kasus #{i+1} ---")
        print(f"==============================")
        
        # 1. Jalankan Agen Terdakwa
        output_agen = agen_terdakwa.analyze_risk(
            data["kasus"], 
            data["klasifikasi_konteks"], 
            run_name=f"SubRun_{i}"
        )
        
        # --- REM TANGAN PERTAMA ---
        print("\n[SISTEM] Menunggu 15 detik agar server Google/Gemini tidak kepanasan (Anti Rate-Limit)...")
        time.sleep(15) 
        
        # 2. Panggil Sang Hakim
        print("[HAKIM] Sedang menilai kinerja agen...")
        penilaian = llm_judge_audit(
            keluhan=data["kasus"],
            hasil_agen=output_agen,
            kriteria_rubrik="Skor 1-3 (Estetika), 4-6 (Kenyamanan), 7-8 (Bahaya Fisik), 9-10 (Ancaman Nyawa)"
        )
        
        # Menampilkan penilaian jika bukan error limit (tipe dictionary)
        if isinstance(penilaian, dict):
            acc = penilaian.get('score_accuracy', 0)
            reason = penilaian.get('reasoning_quality', 0)
            note = penilaian.get('audit_note', 'Tidak ada catatan')
            
            print(f"   -> Nilai Akurasi: {acc}/5")
            print(f"   -> Nilai Penalaran: {reason}/5")
            print(f"   -> Catatan Hakim: {note}")
            
            results.append({
                "kasus": data["kasus"],
                "skor_akurasi": acc,
                "skor_penalaran": reason,
                "catatan_hakim": note
            })
        
        # --- REM TANGAN KEDUA ---
        if i < len(test_cases) - 1:
            print("\n[SISTEM] Menunggu 15 detik sebelum memproses kasus berikutnya...")
            time.sleep(15)

    # 3. Buat Laporan Tabel (DataFrame)
    if results:
        df_report = pd.DataFrame(results)
        avg_acc = df_report["skor_akurasi"].mean()
        
        print("\n" + "="*50)
        print(f"HASIL AKHIR AUDIT: Rata-rata Akurasi Agen = {avg_acc}/5.0")
        print("="*50)
        
        # Simpan ke MLflow
        df_report.to_csv("laporan_audit_agen.csv", index=False)
        mlflow.log_artifact("laporan_audit_agen.csv")
        mlflow.log_metric("average_accuracy", avg_acc)
    else:
        print("\n[!] Audit gagal diselesaikan karena masalah koneksi/limit.")