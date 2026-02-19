import requests
import json
import mlflow

# Import konfigurasi SSoT
import src.config.settings as settings

def evaluate_with_llama(agent_name, user_query, agent_response, ground_truth):
    """
    Memanggil Ollama REST API menggunakan konfigurasi dari settings.py.
    """
    prompt = f"""
    Anda adalah Auditor Ahli Perencanaan Pembangunan (LLM-as-a-Judge) di Kelurahan Nusukan, Surakarta.
    Tugas Anda mengevaluasi respon agen ({agent_name}) terhadap usulan Musrenbang warga.
    
    Usulan Warga: "{user_query}"
    Respon Agen: "{agent_response}"
    Konteks/Ground Truth: "{ground_truth}"
    
    Berikan penilaian metrik (skala 1-10) untuk:
    1. relevance_score: Relevansi respon agen dengan nomenklatur/konteks.
    2. accuracy_score: Keakuratan skor agen dibandingkan ground truth.
    3. reasoning_score: Kualitas alasan agen.
    
    OUTPUT HARUS berformat JSON dengan key: "relevance_score", "accuracy_score", "reasoning_score", dan "feedback".
    """

    payload = {
        "model": settings.OLLAMA_JUDGE_MODEL,
        "prompt": prompt,
        "format": "json",
        "stream": False
    }

    try:
        # Request ke Localhost Ollama (Zero RPM Limit)
        response = requests.post(settings.OLLAMA_API_URL, json=payload, timeout=120)
        response.raise_for_status()
        
        result_text = response.json().get("response", "{}")
        eval_metrics = json.loads(result_text)
        return eval_metrics
        
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Koneksi ke Ollama gagal: {e}")
        return None
    except json.JSONDecodeError:
        print(f"[ERROR] Ollama tidak mengembalikan JSON yang valid. Output: {result_text}")
        return None

def main():
    print("=====================================================")
    print("[INFO] Memulai Pipeline Evaluasi (LLM-as-a-Judge)")
    print("=====================================================")
    
    # 1. Inisialisasi MLflow Tracking
    settings.setup_mlflow_tracking()
    mlflow.set_experiment("Evaluasi-Agen-Musrenbang")
    
    # 2. Mock Data Log (Simulasi hasil dari main_orchestrator.py)
    mock_logs = [
        {
            "agent": "sociology_agent",
            "query": "Pembangunan poskamling di RW 03 Nusukan karena rawan curanmor",
            "response": "Skor Dampak: 8. Terkonfirmasi dengan data isu strategis keamanan Nusukan.",
            "ground_truth": "RW 03 Nusukan memiliki tingkat curanmor tinggi. Pembangunan poskamling sangat valid."
        },
        {
            "agent": "mitigation_agent",
            "query": "Perbaikan talud sungai anyar yang longsor",
            "response": "Skor Bahaya: 9. Mengancam keselamatan warga di bantaran.",
            "ground_truth": "Talud sungai anyar kritis dan berisiko longsor susulan. Prioritas mitigasi utama."
        }
    ]

    # 3. Eksekusi Evaluasi
    with mlflow.start_run(run_name="Evaluasi_Llama3_Local"):
        # Log parameter arsitektur
        mlflow.log_param("judge_model", settings.OLLAMA_JUDGE_MODEL)
        mlflow.log_param("main_agent_model", settings.DEFAULT_LLM_MODEL)
        mlflow.log_param("evaluation_backend", "Ollama_WSL_Local")

        total_accuracy = 0
        total_relevance = 0
        
        for idx, log in enumerate(mock_logs):
            print(f"\n-> Mengaudit Agen: {log['agent']}")
            metrics = evaluate_with_llama(log['agent'], log['query'], log['response'], log['ground_truth'])
            
            if metrics:
                print(f"   Metrik Diterima: {metrics}")
                
                # Ekstraksi skor (fallback ke 0 jika gagal parsing)
                acc_score = metrics.get("accuracy_score", 0)
                rel_score = metrics.get("relevance_score", 0)
                
                # Log ke DagsHub
                mlflow.log_metric(f"{log['agent']}_accuracy", acc_score, step=idx)
                mlflow.log_metric(f"{log['agent']}_relevance", rel_score, step=idx)
                
                total_accuracy += acc_score
                total_relevance += rel_score

        # 4. Kalkulasi & Log Rata-rata Keseluruhan
        if mock_logs:
            avg_acc = total_accuracy / len(mock_logs)
            avg_rel = total_relevance / len(mock_logs)
            mlflow.log_metric("average_system_accuracy", avg_acc)
            mlflow.log_metric("average_system_relevance", avg_rel)
            
            print("\n=====================================================")
            print(f"[SUCCESS] Evaluasi Selesai.")
            print(f"-> Rata-rata Akurasi  : {avg_acc}/10")
            print(f"-> Rata-rata Relevansi: {avg_rel}/10")
            print("[INFO] Semua metrik telah dikirim ke DagsHub.")
            print("=====================================================")

if __name__ == "__main__":
    main()