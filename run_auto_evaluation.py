import os
import sys
import time
import json
import requests
import pandas as pd
import mlflow

# Konfigurasi Path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

import src.config.settings as settings
from src.agents.classifier_agent import ClassifierAgent

CSV_PATH = os.path.join(CURRENT_DIR, "data", "external", "Form Musrenbangkel 2026.csv")

def evaluate_classifier_with_llama(user_query, agent_response, ground_truth_kamus):
    prompt = f"""
    Anda adalah Auditor Ahli Perencanaan Pembangunan (LLM-as-a-Judge) tingkat Kecamatan.
    Tugas Anda mengevaluasi akurasi Agen Klasifikasi dalam memetakan usulan warga ke Nomenklatur Musrenbang.
    
    Usulan Warga: "{user_query}"
    Nomenklatur Pilihan Agen (Gemini): "{agent_response}"
    Nomenklatur Seharusnya (Ground Truth dari CSV): "{ground_truth_kamus}"
    
    Berikan penilaian metrik (skala 1-10) untuk:
    1. accuracy_score: Seberapa cocok Nomenklatur pilihan Agen dengan Ground Truth (Beri 10 jika sama persis atau maknanya sangat identik, beri nilai rendah jika melenceng ke dinas yang salah).
    2. reasoning_score: Kualitas alasan logis yang diberikan Agen.
    
    OUTPUT HARUS berformat JSON dengan key: "accuracy_score", "reasoning_score", dan "feedback".
    """

    payload = {
        "model": settings.OLLAMA_JUDGE_MODEL,
        "prompt": prompt,
        "format": "json",
        "stream": False
    }

    try:
        response = requests.post(settings.OLLAMA_API_URL, json=payload, timeout=120)
        response.raise_for_status()
        result_text = response.json().get("response", "{}")
        return json.loads(result_text)
    except Exception as e:
        print(f"[ERROR] Evaluasi Ollama gagal: {e}")
        return None

def main():
    print("=====================================================")
    print("🚀 [INFO] MEMULAI PIPELINE EVALUASI OTOMATIS (MASSAL)")
    print("=====================================================")
    
    settings.setup_mlflow_tracking()
    agen_klasifikasi = ClassifierAgent()
    
    print(f"\n[INFO] Membaca dataset dari: {CSV_PATH}")
    try:
        try:
            df = pd.read_csv(CSV_PATH, sep=';')
            if 'KAMUS USULAN' not in df.columns:
                df = pd.read_csv(CSV_PATH, sep=',')
        except Exception as e:
            print(f"[ERROR] Gagal membaca CSV: {e}")
            return
            
        kolom_masalah = [col for col in df.columns if 'PRIORITAS PERMASALAHAN' in col][0]
        df = df.dropna(subset=[kolom_masalah, 'KAMUS USULAN'])
        
        # KITA AMBIL 15 BARIS PERTAMA
        df_sample = df.head(15)
        print(f"[INFO] Dataset siap! Mengaudit {len(df_sample)} kasus usulan warga...")
        
    except Exception as e:
        print(f"[ERROR] Dataset bermasalah: {e}")
        return

    mlflow.set_experiment("Evaluasi-Massal-Klasifikasi")
    
    total_accuracy = 0
    total_reasoning = 0
    berhasil = 0

    # 3. Looping Ujian Otomatis (TANPA mlflow.start_run global agar tidak bentrok)
    for index, row in df_sample.iterrows():
        kasus = str(row[kolom_masalah])
        lokasi = f"RW {row['RW']}" if 'RW' in df.columns else ""
        kasus_lengkap = f"{kasus} (Lokasi: {lokasi})"
        ground_truth = str(row['KAMUS USULAN'])
        
        print(f"\n[{berhasil+1}/{len(df_sample)}] Mengaudit Kasus: {kasus[:60]}...")
        
        # --- STEP A: Agen Utama Bekerja (Gemini) ---
        try:
            # Ini akan otomatis membuka dan menutup run MLflow-nya sendiri sesuai Blueprint
            hasil_agen = agen_klasifikasi.analyze(kasus_lengkap, run_name=f"Eval_Row_{index}")
        except Exception as e:
            print(f"[!] Gemini API Error (Mungkin Limit): {e}")
            print("[!] Menunggu 30 detik untuk pendinginan...")
            time.sleep(30)
            continue 

        # --- STEP B: Agen Hakim Menilai (Ollama Lokal) ---
        print("   -> [HAKIM] Memanggil Ollama Local untuk membandingkan hasil...")
        metrics = evaluate_classifier_with_llama(kasus_lengkap, hasil_agen, ground_truth)
        
        if metrics:
            acc_score = metrics.get("accuracy_score", 0)
            reason_score = metrics.get("reasoning_score", 0)
            
            print(f"   -> Akurasi: {acc_score}/10 | Penalaran: {reason_score}/10")
            print(f"   -> Catatan Hakim: {metrics.get('feedback', '')}")
            
            # Kita buat rekaman baru KHUSUS untuk hasil Hakim agar terpisah dan rapi
            with mlflow.start_run(run_name=f"Hakim_Row_{index}"):
                mlflow.log_metric("classifier_accuracy", acc_score)
                mlflow.log_metric("classifier_reasoning", reason_score)
            
            total_accuracy += acc_score
            total_reasoning += reason_score
            berhasil += 1
        
        # --- REM TANGAN API ---
        print("   -> [SISTEM] Jeda 12 detik mencegah Gemini Rate Limit...")
        time.sleep(12) 

    # 4. Kalkulasi Rapor Akhir
    if berhasil > 0:
        avg_acc = round(total_accuracy / berhasil, 2)
        avg_res = round(total_reasoning / berhasil, 2)
        
        # Rekaman final untuk nilai rata-rata keseluruhan
        with mlflow.start_run(run_name="KESIMPULAN_RATA_RATA_EVALUASI"):
            mlflow.log_param("judge_model", settings.OLLAMA_JUDGE_MODEL)
            mlflow.log_param("dataset_size", len(df_sample))
            mlflow.log_metric("average_classifier_accuracy", avg_acc)
            mlflow.log_metric("average_classifier_reasoning", avg_res)
        
        print("\n" + "="*53)
        print(f"🎉 [SUCCESS] EVALUASI MASSAL SELESAI ({berhasil} Kasus)")
        print("="*53)
        print(f"-> Rata-rata Akurasi Peta Kamus : {avg_acc}/10")
        print(f"-> Rata-rata Kualitas Penalaran : {avg_res}/10")
        print("[INFO] Semua data berhasil diunggah ke DagsHub!")
        print("="*53)

if __name__ == "__main__":
    main()