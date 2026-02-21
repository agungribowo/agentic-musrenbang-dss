import os
import sys
import time
import json
import argparse
import requests
import pandas as pd
import mlflow

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
    Nomenklatur Pilihan Agen (Groq): "{agent_response}"
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

def build_dry_run_response(ground_truth_kamus):
    return (
        f"NOMENKLATUR TERPILIH: {ground_truth_kamus}\n"
        "DINAS TERKAIT: (SIMULASI DRY-RUN)\n"
        "ALASAN PENALARAN: Output simulasi untuk pengujian pipeline tanpa API eksternal."
    ), 0.0

def build_dry_run_metrics():
    return {"accuracy_score": 10, "reasoning_score": 9, "feedback": "Simulasi dry-run statis."}

def normalize_classifier_result(classifier_result):
    if isinstance(classifier_result, tuple):
        hasil_text = str(classifier_result[0]) if len(classifier_result) > 0 else ""
        cost_usd = float(classifier_result[1]) if len(classifier_result) > 1 else 0.0
        return hasil_text, cost_usd

    return str(classifier_result), 0.0

def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline evaluasi otomatis klasifikasi Musrenbang")
    parser.add_argument("--dry-run", action="store_true", help="Jalankan pipeline tanpa memanggil API.")
    parser.add_argument("--sample-size", type=int, default=15, help="Jumlah baris data yang dievaluasi.")
    parser.add_argument("--no-mlflow", action="store_true", help="Matikan tracking MLflow.")
    return parser.parse_args()

def main():
    args = parse_args()
    dry_run = args.dry_run
    enable_mlflow = not args.no_mlflow
    
    print("=====================================================")
    print("🚀 [INFO] MEMULAI PIPELINE EVALUASI OTOMATIS (MASSAL)")
    print("=====================================================")
    
    if enable_mlflow:
        settings.setup_mlflow_tracking()
    agen_klasifikasi = None if dry_run else ClassifierAgent(enable_mlflow=enable_mlflow)
    
    try:
        try:
            df = pd.read_csv(CSV_PATH, sep=';')
            if 'KAMUS USULAN' not in df.columns:
                df = pd.read_csv(CSV_PATH, sep=',')
        except Exception:
            print("[ERROR] Gagal membaca CSV.")
            return
            
        kolom_masalah = [col for col in df.columns if 'PRIORITAS PERMASALAHAN' in col][0]
        df = df.dropna(subset=[kolom_masalah, 'KAMUS USULAN'])
        
        sample_size = max(args.sample_size, 1)
        df_sample = df.head(sample_size)
        
    except Exception as e:
        print(f"[ERROR] Dataset bermasalah: {e}")
        return

    if enable_mlflow:
        mlflow.set_experiment("Evaluasi-Massal-Klasifikasi")
    
    total_accuracy = 0
    total_reasoning = 0
    total_batch_cost = 0.0  # Akumulator Biaya
    berhasil = 0

    for index, row in df_sample.iterrows():
        kasus = str(row[kolom_masalah])
        lokasi = f"RW {row['RW']}" if 'RW' in df.columns else ""
        kasus_lengkap = f"{kasus} (Lokasi: {lokasi})"
        ground_truth = str(row['KAMUS USULAN'])
        
        print(f"\n[{berhasil+1}/{len(df_sample)}] Mengaudit Kasus: {kasus[:60]}...")
        
        # --- STEP A: Agen Utama Bekerja (Groq) ---
        try:
            if dry_run:
                hasil_agen, cost_usd = build_dry_run_response(ground_truth)
            else:
                classifier_result = agen_klasifikasi.analyze(kasus_lengkap, run_name=f"Eval_Row_{index}")
                hasil_agen, cost_usd = normalize_classifier_result(classifier_result)
            
            total_batch_cost += cost_usd
                
        except Exception as e:
            print(f"[!] Groq API Error: {e}")
            time.sleep(30)
            continue 

        # --- STEP B: Agen Hakim Menilai (Ollama Lokal) ---
        if dry_run:
            metrics = build_dry_run_metrics()
        else:
            print("   -> [HAKIM] Memanggil Ollama Local untuk mengevaluasi...")
            metrics = evaluate_classifier_with_llama(kasus_lengkap, hasil_agen, ground_truth)
        
        if metrics:
            acc_score = metrics.get("accuracy_score", 0)
            reason_score = metrics.get("reasoning_score", 0)
            
            # --- TAMBAHAN LOGIKA SOFT THRESHOLD PoC v1 ---
            # Mengevaluasi apakah skor memenuhi standar minimal dari settings.py
            is_passed = (
                acc_score >= settings.VALIDATION_THRESHOLD_ACCURACY and
                reason_score >= settings.VALIDATION_THRESHOLD_REASONING
            )
            status_teks = "LULUS" if is_passed else "GAGAL VALIDASI"
            # ---------------------------------------------
            
            print(f"   -> Akurasi: {acc_score}/10 | Penalaran: {reason_score}/10")
            
            if enable_mlflow:
                with mlflow.start_run(run_name=f"Hakim_Row_{index}"):
                    # Log skor mentah (Metrics asli Anda)
                    mlflow.log_metric("classifier_accuracy", acc_score)
                    mlflow.log_metric("classifier_reasoning", reason_score)
                    
                    # Log metrik Soft Threshold
                    mlflow.log_metric("validation_pass", 1 if is_passed else 0)
                    
                    # Log parameter kebijakan yang digunakan (Audit Trail)
                    mlflow.log_param("val_threshold_accuracy", settings.VALIDATION_THRESHOLD_ACCURACY)
                    mlflow.log_param("val_threshold_reasoning", settings.VALIDATION_THRESHOLD_REASONING)
                    mlflow.set_tag("validation_policy_version", "v1.0-soft")
            
            total_accuracy += acc_score
            total_reasoning += reason_score
            berhasil += 1
        
        if not dry_run:
            time.sleep(12)

    # 4. Kalkulasi Rapor Akhir
    if berhasil > 0:
        avg_acc = round(total_accuracy / berhasil, 2)
        avg_res = round(total_reasoning / berhasil, 2)
        
        if enable_mlflow:
            with mlflow.start_run(run_name="KESIMPULAN_RATA_RATA_EVALUASI"):
                mlflow.log_param("judge_model", settings.OLLAMA_JUDGE_MODEL)
                mlflow.log_param("dataset_size", len(df_sample))
                mlflow.log_metric("average_classifier_accuracy", avg_acc)
                mlflow.log_metric("average_classifier_reasoning", avg_res)
                mlflow.log_metric("total_batch_simulated_cost_usd", total_batch_cost)
        
        print("\n" + "="*53)
        print(f"🎉 [SUCCESS] EVALUASI SELESAI ({berhasil} Kasus)")
        print(f"-> Rata-rata Akurasi : {avg_acc}/10")
        print(f"-> Rata-rata Penalaran : {avg_res}/10")
        print(f"-> Total Biaya Simulasi: ${total_batch_cost:.4f}")
        print("="*53)

if __name__ == "__main__":
    main()