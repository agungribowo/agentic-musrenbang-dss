import os
import sys
import time
import json
import re
import argparse
import random
import requests
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

import src.config.settings as settings

from src.agents.classifier_agent import ClassifierAgent

CSV_PATH = os.path.join(CURRENT_DIR, "data", "external", "Form Musrenbangkel 2026.csv")
FAIRNESS_ALERT_THRESHOLD_PCT = 20.0
STOCHASTIC_PROFILE_RANGES = {
    "conservative": {
        "acc_min": 4,
        "acc_max": 8,
        "reasoning_min": 3,
        "reasoning_max": 7,
    },
    "balanced": {
        "acc_min": 5,
        "acc_max": 10,
        "reasoning_min": 4,
        "reasoning_max": 10,
    },
    "aggressive": {
        "acc_min": 8,
        "acc_max": 10,
        "reasoning_min": 7,
        "reasoning_max": 10,
    },
}


def get_validation_thresholds():
    return settings.VALIDATION_THRESHOLD_ACCURACY, settings.VALIDATION_THRESHOLD_REASONING

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
        response = requests.post(
            settings.OLLAMA_API_URL,
            json=payload,
            timeout=120,
        )
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

def build_dry_run_metrics_static():
    return {"accuracy_score": 10, "reasoning_score": 9, "feedback": "Simulasi dry-run statis."}


def build_dry_run_metrics_stochastic(
    rng,
    acc_min,
    acc_max,
    reasoning_min,
    reasoning_max,
):
    accuracy_score = rng.randint(acc_min, acc_max)
    reasoning_score = rng.randint(reasoning_min, reasoning_max)
    return {
        "accuracy_score": accuracy_score,
        "reasoning_score": reasoning_score,
        "feedback": "Simulasi dry-run stokastik (reproducible berdasarkan seed).",
    }

def normalize_classifier_result(classifier_result):
    if isinstance(classifier_result, tuple):
        hasil_text = str(classifier_result[0]) if len(classifier_result) > 0 else ""
        cost_usd = float(classifier_result[1]) if len(classifier_result) > 1 else 0.0
        return hasil_text, cost_usd

    return str(classifier_result), 0.0


def _stratified_sample(df, sample_size, strata_col, seed):
    working_df = df.dropna(subset=[strata_col]).copy()
    if working_df.empty:
        return df.sample(n=sample_size, random_state=seed)

    strata_counts = working_df[strata_col].value_counts().sort_index()
    strata_weights = strata_counts / strata_counts.sum()
    allocations = (strata_weights * sample_size).round().astype(int)
    allocations[allocations < 1] = 1

    while allocations.sum() > sample_size:
        reducible = allocations[allocations > 1]
        if reducible.empty:
            break
        key_to_reduce = reducible.sort_values(ascending=False).index[0]
        allocations.loc[key_to_reduce] -= 1

    while allocations.sum() < sample_size:
        remaining_capacity = strata_counts - allocations
        expandable = remaining_capacity[remaining_capacity > 0]
        if expandable.empty:
            break
        key_to_increase = expandable.sort_values(ascending=False).index[0]
        allocations.loc[key_to_increase] += 1

    sampled_parts = []
    for idx, (stratum, take_n) in enumerate(allocations.items()):
        if take_n <= 0:
            continue
        stratum_df = working_df[working_df[strata_col] == stratum]
        take_n = min(take_n, len(stratum_df))
        sampled_parts.append(stratum_df.sample(n=take_n, random_state=seed + idx))

    if not sampled_parts:
        return df.sample(n=sample_size, random_state=seed)

    sampled_df = pd.concat(sampled_parts).drop_duplicates()

    if len(sampled_df) < sample_size:
        remaining_df = df.drop(index=sampled_df.index, errors="ignore")
        remaining_need = min(sample_size - len(sampled_df), len(remaining_df))
        if remaining_need > 0:
            filler_df = remaining_df.sample(n=remaining_need, random_state=seed)
            sampled_df = pd.concat([sampled_df, filler_df])

    if len(sampled_df) > sample_size:
        sampled_df = sampled_df.sample(n=sample_size, random_state=seed)

    return sampled_df


def _to_metric_safe_key(raw_text, max_length=48):
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", str(raw_text).strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "unknown"
    return cleaned[:max_length]


def _build_strata_composition(df_sample, strata_col):
    if not strata_col or strata_col not in df_sample.columns:
        return None

    value_counts = df_sample[strata_col].fillna("UNKNOWN").astype(str).value_counts().sort_index()
    return {str(key): int(val) for key, val in value_counts.items()}


def _build_fairness_summary(strata_eval_stats, strata_col):
    if not strata_col or not strata_eval_stats:
        return None, None

    fairness_rows = []
    for stratum_name, stats in sorted(strata_eval_stats.items(), key=lambda item: item[0]):
        total = int(stats.get("total", 0))
        passed = int(stats.get("passed", 0))
        failed = max(total - passed, 0)
        pass_rate = round((passed / total) * 100, 2) if total > 0 else 0.0
        fairness_rows.append(
            {
                "stratum": str(stratum_name),
                "total": total,
                "passed": passed,
                "failed": failed,
                "pass_rate_pct": pass_rate,
            }
        )

    if not fairness_rows:
        return None, None, None, None

    pass_rates = [row["pass_rate_pct"] for row in fairness_rows]
    fairness_gap_pct = round(max(pass_rates) - min(pass_rates), 2) if pass_rates else None
    fairness_alert = (
        fairness_gap_pct is not None and fairness_gap_pct > FAIRNESS_ALERT_THRESHOLD_PCT
    )

    lines = [f"[FAIRNESS] Dashboard per strata ({strata_col}):"]
    for row in fairness_rows:
        lines.append(
            f"- {row['stratum']}: total={row['total']} | lulus={row['passed']} | "
            f"gagal={row['failed']} | pass_rate={row['pass_rate_pct']}%"
        )

    if fairness_gap_pct is not None:
        lines.append(f"[FAIRNESS] gap_pass_rate_pct={fairness_gap_pct}%")
        lines.append(
            f"[FAIRNESS] alert_threshold_pct={FAIRNESS_ALERT_THRESHOLD_PCT}% | "
            f"fairness_alert={str(fairness_alert).lower()}"
        )

    return fairness_rows, "\n".join(lines), fairness_gap_pct, fairness_alert

def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline evaluasi otomatis klasifikasi Musrenbang")
    parser.add_argument("--dry-run", action="store_true", help="Jalankan pipeline tanpa memanggil API.")
    parser.add_argument(
        "--dry-run-mode",
        choices=["static", "stochastic"],
        default="static",
        help="Mode simulasi skor pada dry-run: static (tetap) atau stochastic (variatif).",
    )
    parser.add_argument(
        "--stochastic-profile",
        choices=["conservative", "balanced", "aggressive"],
        default="balanced",
        help="Preset rentang skor stochastic untuk demo cepat.",
    )
    parser.add_argument(
        "--stochastic-acc-min",
        type=int,
        default=None,
        help="Batas bawah skor accuracy untuk mode dry-run stochastic (1-10).",
    )
    parser.add_argument(
        "--stochastic-acc-max",
        type=int,
        default=None,
        help="Batas atas skor accuracy untuk mode dry-run stochastic (1-10).",
    )
    parser.add_argument(
        "--stochastic-reasoning-min",
        type=int,
        default=None,
        help="Batas bawah skor reasoning untuk mode dry-run stochastic (1-10).",
    )
    parser.add_argument(
        "--stochastic-reasoning-max",
        type=int,
        default=None,
        help="Batas atas skor reasoning untuk mode dry-run stochastic (1-10).",
    )
    parser.add_argument("--sample-size", type=int, default=15, help="Jumlah baris data yang dievaluasi.")
    parser.add_argument(
        "--sampling-mode",
        choices=["random", "head", "stratified_rw", "stratified_kamus"],
        default="random",
        help="Mode sampling: random, head, stratified_rw, atau stratified_kamus.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed untuk sampling random agar hasil reproducible.",
    )
    parser.add_argument("--no-mlflow", action="store_true", help="Matikan tracking MLflow.")
    return parser.parse_args()

def main():
    args = parse_args()
    dry_run = args.dry_run
    enable_mlflow = not args.no_mlflow
    mlflow = None
    val_threshold_accuracy, val_threshold_reasoning = get_validation_thresholds()
    sampling_strata_col = None
    sample_composition = None
    strata_eval_stats = {}
    dry_run_rng = random.Random(args.seed)

    profile_ranges = STOCHASTIC_PROFILE_RANGES[args.stochastic_profile]
    effective_stochastic_acc_min = (
        args.stochastic_acc_min
        if args.stochastic_acc_min is not None
        else profile_ranges["acc_min"]
    )
    effective_stochastic_acc_max = (
        args.stochastic_acc_max
        if args.stochastic_acc_max is not None
        else profile_ranges["acc_max"]
    )
    effective_stochastic_reasoning_min = (
        args.stochastic_reasoning_min
        if args.stochastic_reasoning_min is not None
        else profile_ranges["reasoning_min"]
    )
    effective_stochastic_reasoning_max = (
        args.stochastic_reasoning_max
        if args.stochastic_reasoning_max is not None
        else profile_ranges["reasoning_max"]
    )

    range_checks = [
        ("stochastic-acc-min", effective_stochastic_acc_min),
        ("stochastic-acc-max", effective_stochastic_acc_max),
        ("stochastic-reasoning-min", effective_stochastic_reasoning_min),
        ("stochastic-reasoning-max", effective_stochastic_reasoning_max),
    ]
    for name, value in range_checks:
        if value < 1 or value > 10:
            print(f"[ERROR] Nilai --{name} harus di rentang 1..10. Saat ini: {value}")
            return

    if effective_stochastic_acc_min > effective_stochastic_acc_max:
        print("[ERROR] --stochastic-acc-min tidak boleh lebih besar dari --stochastic-acc-max.")
        return

    if effective_stochastic_reasoning_min > effective_stochastic_reasoning_max:
        print("[ERROR] --stochastic-reasoning-min tidak boleh lebih besar dari --stochastic-reasoning-max.")
        return
    
    print("=====================================================")

    if dry_run and args.dry_run_mode == "stochastic":
        print(
            "[INFO] Dry-run stochastic profile: "
            f"{args.stochastic_profile} | "
            f"acc={effective_stochastic_acc_min}-{effective_stochastic_acc_max} | "
            f"reasoning={effective_stochastic_reasoning_min}-{effective_stochastic_reasoning_max}"
        )
    print("🚀 [INFO] MEMULAI PIPELINE EVALUASI OTOMATIS (MASSAL)")
    print("=====================================================")
    
    if enable_mlflow:
        settings.setup_mlflow_tracking()
        mlflow = settings.mlflow
        if mlflow is None:
            print(f"[WARN] MLflow tidak bisa diimport, mode no-mlflow diaktifkan otomatis: {settings.MLFLOW_IMPORT_ERROR}")
            enable_mlflow = False

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
        sample_size = min(sample_size, len(df))

        if args.sampling_mode == "random":
            df_sample = df.sample(n=sample_size, random_state=args.seed)
            print(f"[INFO] Sampling mode: random | sample_size={sample_size} | seed={args.seed}")
        elif args.sampling_mode == "stratified_rw":
            if "RW" not in df.columns:
                print("[WARN] Kolom RW tidak ditemukan. Fallback ke random sampling.")
                df_sample = df.sample(n=sample_size, random_state=args.seed)
                print(f"[INFO] Sampling mode: random(fallback) | sample_size={sample_size} | seed={args.seed}")
            else:
                df_sample = _stratified_sample(df, sample_size, "RW", args.seed)
                sampling_strata_col = "RW"
                print(
                    f"[INFO] Sampling mode: stratified_rw | strata={df_sample['RW'].nunique()} RW | "
                    f"sample_size={len(df_sample)} | seed={args.seed}"
                )
        elif args.sampling_mode == "stratified_kamus":
            if "KAMUS USULAN" not in df.columns:
                print("[WARN] Kolom KAMUS USULAN tidak ditemukan. Fallback ke random sampling.")
                df_sample = df.sample(n=sample_size, random_state=args.seed)
                print(f"[INFO] Sampling mode: random(fallback) | sample_size={sample_size} | seed={args.seed}")
            else:
                df_sample = _stratified_sample(df, sample_size, "KAMUS USULAN", args.seed)
                sampling_strata_col = "KAMUS USULAN"
                print(
                    f"[INFO] Sampling mode: stratified_kamus | strata={df_sample['KAMUS USULAN'].nunique()} kategori | "
                    f"sample_size={len(df_sample)} | seed={args.seed}"
                )
        else:
            df_sample = df.head(sample_size)
            print(f"[INFO] Sampling mode: head | sample_size={sample_size}")

        sample_composition = _build_strata_composition(df_sample, sampling_strata_col)
        if sample_composition:
            print(f"[INFO] Komposisi strata ({sampling_strata_col}): {sample_composition}")
        
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
            if args.dry_run_mode == "stochastic":
                metrics = build_dry_run_metrics_stochastic(
                    dry_run_rng,
                    effective_stochastic_acc_min,
                    effective_stochastic_acc_max,
                    effective_stochastic_reasoning_min,
                    effective_stochastic_reasoning_max,
                )
            else:
                metrics = build_dry_run_metrics_static()
        else:
            print("   -> [HAKIM] Memanggil Ollama Local untuk mengevaluasi...")
            metrics = evaluate_classifier_with_llama(kasus_lengkap, hasil_agen, ground_truth)
        
        if metrics:
            acc_score = metrics.get("accuracy_score", 0)
            reason_score = metrics.get("reasoning_score", 0)
            
            # --- TAMBAHAN LOGIKA SOFT THRESHOLD PoC v1 ---
            # Mengevaluasi apakah skor memenuhi standar minimal dari settings.py
            is_passed = (
                acc_score >= val_threshold_accuracy and
                reason_score >= val_threshold_reasoning
            )
            status_teks = "LULUS" if is_passed else "GAGAL VALIDASI"
            # ---------------------------------------------
            
            print(f"   -> Akurasi: {acc_score}/10 | Penalaran: {reason_score}/10 | Status: {status_teks}")

            if sampling_strata_col and sampling_strata_col in row:
                stratum_value = str(row[sampling_strata_col])
                if stratum_value not in strata_eval_stats:
                    strata_eval_stats[stratum_value] = {"total": 0, "passed": 0}
                strata_eval_stats[stratum_value]["total"] += 1
                strata_eval_stats[stratum_value]["passed"] += 1 if is_passed else 0
            
            if enable_mlflow:
                with mlflow.start_run(run_name=f"Hakim_Row_{index}"):
                    # Log skor mentah (Metrics asli Anda)
                    mlflow.log_metric("classifier_accuracy", acc_score)
                    mlflow.log_metric("classifier_reasoning", reason_score)
                    
                    # Log metrik Soft Threshold
                    mlflow.log_metric("validation_pass", 1 if is_passed else 0)
                    
                    # Log parameter kebijakan yang digunakan (Audit Trail)
                    mlflow.log_param("val_threshold_accuracy", val_threshold_accuracy)
                    mlflow.log_param("val_threshold_reasoning", val_threshold_reasoning)
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
        fairness_rows, fairness_text, fairness_gap_pct, fairness_alert = _build_fairness_summary(
            strata_eval_stats,
            sampling_strata_col,
        )

        if fairness_text:
            print("\n" + fairness_text)
        
        if enable_mlflow:
            with mlflow.start_run(run_name="KESIMPULAN_RATA_RATA_EVALUASI"):
                mlflow.log_param("judge_model", settings.OLLAMA_JUDGE_MODEL)
                mlflow.log_param("dataset_size", len(df_sample))
                mlflow.log_param("sampling_mode", args.sampling_mode)
                mlflow.log_param("sampling_seed", args.seed)
                mlflow.log_param("sampling_strata_col", sampling_strata_col or "NONE")
                mlflow.log_param("dry_run_mode", args.dry_run_mode if dry_run else "none")
                mlflow.log_param("stochastic_profile", args.stochastic_profile)
                mlflow.log_param("stochastic_acc_min", effective_stochastic_acc_min)
                mlflow.log_param("stochastic_acc_max", effective_stochastic_acc_max)
                mlflow.log_param("stochastic_reasoning_min", effective_stochastic_reasoning_min)
                mlflow.log_param("stochastic_reasoning_max", effective_stochastic_reasoning_max)
                mlflow.log_metric("average_classifier_accuracy", avg_acc)
                mlflow.log_metric("average_classifier_reasoning", avg_res)
                mlflow.log_metric("total_batch_simulated_cost_usd", total_batch_cost)
                mlflow.log_metric("sample_size_selected", len(df_sample))
                mlflow.log_param("fairness_alert_threshold_pct", FAIRNESS_ALERT_THRESHOLD_PCT)

                if sample_composition:
                    mlflow.log_metric("sample_strata_unique", len(sample_composition))
                    used_metric_keys = set()
                    col_key = _to_metric_safe_key(sampling_strata_col)
                    for idx, (stratum_name, stratum_count) in enumerate(sample_composition.items()):
                        stratum_key = _to_metric_safe_key(stratum_name)
                        metric_key = f"sample_count_{col_key}_{stratum_key}"
                        if metric_key in used_metric_keys:
                            metric_key = f"{metric_key}_{idx}"
                        used_metric_keys.add(metric_key)
                        mlflow.log_metric(metric_key, stratum_count)

                    composition_payload = {
                        "sampling_mode": args.sampling_mode,
                        "sampling_seed": args.seed,
                        "sampling_strata_col": sampling_strata_col,
                        "sample_size": len(df_sample),
                        "strata_composition": sample_composition,
                    }
                    artifact_name = f"sampling/strata_composition_{_to_metric_safe_key(sampling_strata_col)}.json"
                    mlflow.log_text(
                        json.dumps(composition_payload, ensure_ascii=False, indent=2),
                        artifact_name,
                    )

                if fairness_rows:
                    col_key = _to_metric_safe_key(sampling_strata_col)
                    for idx, row_stats in enumerate(fairness_rows):
                        stratum_key = _to_metric_safe_key(row_stats["stratum"])
                        base_key = f"fairness_{col_key}_{stratum_key}"
                        if idx > 0:
                            base_key = f"{base_key}_{idx}"

                        mlflow.log_metric(f"{base_key}_pass_rate_pct", row_stats["pass_rate_pct"])
                        mlflow.log_metric(f"{base_key}_passed", row_stats["passed"])
                        mlflow.log_metric(f"{base_key}_failed", row_stats["failed"])
                        mlflow.log_metric(f"{base_key}_total", row_stats["total"])

                    if fairness_gap_pct is not None:
                        mlflow.log_metric("fairness_gap_pass_rate_pct", fairness_gap_pct)
                        mlflow.log_metric("fairness_alert", 1 if fairness_alert else 0)
                        mlflow.set_tag("fairness_alert", str(fairness_alert).lower())

                    fairness_payload = {
                        "sampling_mode": args.sampling_mode,
                        "sampling_seed": args.seed,
                        "sampling_strata_col": sampling_strata_col,
                        "thresholds": {
                            "accuracy": val_threshold_accuracy,
                            "reasoning": val_threshold_reasoning,
                        },
                        "fairness_rows": fairness_rows,
                        "fairness_gap_pass_rate_pct": fairness_gap_pct,
                        "fairness_alert_threshold_pct": FAIRNESS_ALERT_THRESHOLD_PCT,
                        "fairness_alert": fairness_alert,
                    }
                    mlflow.log_text(
                        json.dumps(fairness_payload, ensure_ascii=False, indent=2),
                        f"sampling/fairness_summary_{_to_metric_safe_key(sampling_strata_col)}.json",
                    )
                    mlflow.log_text(
                        fairness_text,
                        f"sampling/fairness_dashboard_{_to_metric_safe_key(sampling_strata_col)}.md",
                    )
        
        print("\n" + "="*53)
        print(f"🎉 [SUCCESS] EVALUASI SELESAI ({berhasil} Kasus)")
        print(f"-> Rata-rata Akurasi : {avg_acc}/10")
        print(f"-> Rata-rata Penalaran : {avg_res}/10")
        print(f"-> Total Biaya Simulasi: ${total_batch_cost:.4f}")
        print("="*53)

if __name__ == "__main__":
    main()