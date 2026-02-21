# Change Log (Append-Only)

Dokumen ini **tidak di-overwrite**. Setiap sesi kerja menambahkan entry baru di bagian paling bawah.

## Format Entry

Gunakan format berikut untuk setiap sesi baru:

```markdown
## YYYY-MM-DD

### Session YYYY-MM-DDTHH:mm:sszzz
- Ringkasan: ...
- Scope: ...
- File Changed:
  - file_a.py
  - file_b.md
- Detail:
  - Perubahan penting 1
  - Perubahan penting 2
- Verification:
  - Command: ...
  - Result: ...
- Notes: ...
```

---

## 2026-02-21

### Session 2026-02-21T09:16:12.8510656+07:00
- Ringkasan: Iterasi besar evaluasi otomatis untuk validasi threshold, sampling representatif, fairness audit, dan dry-run simulator.
- Scope: Evaluasi classifier, logging MLflow, fairness observability, dokumentasi operasional.
- File Changed:
  - run_auto_evaluation.py
  - src/config/settings.py
  - README.md
  - change.md
- Detail:
  - Menambahkan soft-threshold dan status lulus/gagal per kasus.
  - Menambahkan mode sampling `random`, `head`, `stratified_rw`, `stratified_kamus` + `seed` reproducible.
  - Menambahkan logging komposisi strata ke MLflow + artifact JSON.
  - Menambahkan dashboard fairness per strata (total/lulus/gagal/pass-rate).
  - Menambahkan `fairness_gap_pass_rate_pct` dan `fairness_alert` (rule gap > 20%).
  - Menambahkan mode `--dry-run-mode stochastic` untuk skor simulasi non-statis.
  - Menambahkan rentang stochastic via CLI (min/max accuracy dan reasoning).
  - Menambahkan preset ringkas `--stochastic-profile conservative|balanced|aggressive` (manual min/max override preset).
  - Merapikan README untuk seluruh opsi evaluasi dan fairness observability.
- Verification:
  - Command: `python .\run_auto_evaluation.py --dry-run --dry-run-mode stochastic --sampling-mode stratified_kamus --sample-size 20 --seed 42 --no-mlflow`
  - Result: Sukses; skor variatif, fairness dashboard aktif, fairness alert terpicu pada skenario konservatif.
- Notes: Entry ini adalah baseline untuk changelog append harian otomatis.

### Session 2026-02-21T09:19:28.9248207+07:00
- Ringkasan: Template append test
- Scope: Changelog tooling
- File Changed:
  - change.md
  - append_change_log.ps1
  - README.md
- Detail:
  - Create append script
  - Switch to append-only format
- Verification:
  - Command: Get-Date -Format o
  - Result: Sukses
- Notes: Smoke test entry
