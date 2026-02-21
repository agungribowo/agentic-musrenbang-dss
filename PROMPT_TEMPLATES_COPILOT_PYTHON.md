# Paket Template Prompt Copilot (Python Workflow)

Template ini dirancang khusus untuk repo **Agentic-Musrenbang-DSS** agar penggunaan Copilot Free lebih hemat prompt, tapi tetap produktif.

Cara pakai cepat:
- Ganti placeholder dalam tanda `<...>`.
- Kirim 1 prompt lengkap (hindari prompt bertahap yang berulang).
- Minta output ringkas dan terstruktur.

---

## 1) DEBUG

### A. Debug error runtime (cepat & fokus)

```text
Konteks:
- Project: Agentic-Musrenbang-DSS
- File: <path_file>
- Tujuan: memperbaiki error tanpa mengubah API publik

Masalah:
- Error message: <paste_error>
- Langkah reproduksi: <command/langkah>
- Perilaku saat ini: <actual_behavior>
- Perilaku diharapkan: <expected_behavior>

Batasan:
- Ubah seminimal mungkin.
- Jangan refactor besar.
- Jangan ubah nama fungsi publik.

Tolong beri:
1) akar masalah,
2) patch minimal,
3) 3 langkah verifikasi.
```

### B. Debug pipeline multi-agent (orchestrator)

```text
Analisis alur di main_orchestrator.py untuk kasus berikut:
- Input usulan: <isi_input>
- Agen yang dicurigai: <classifier/mitigation/sociology/economy>
- Gejala: <misclassification/score aneh/exception>

Tolong:
- Telusuri titik gagal paling mungkin.
- Usulkan patch minimal pada file terkait di src/agents/.
- Pertahankan kontrak output antar-agen.
- Beri checklist verifikasi dengan command:
  - python .\main_orchestrator.py --quick-case --no-mlflow
```

### C. Debug performa lambat

```text
Saya mengalami eksekusi lambat pada <file/fungsi>. Tolong lakukan analisis bottleneck tanpa mengubah arsitektur besar.

Batasan:
- Fokus pada perbaikan low-risk (caching ringan, I/O, loop boros, query retrieval).
- Jangan ganti provider model.

Output yang saya mau:
1) 3 bottleneck paling mungkin (urut dampak),
2) patch kecil prioritas #1,
3) metrik sebelum/sesudah yang perlu diukur.
```

---

## 2) REFACTOR

### A. Refactor aman (readability + maintainability)

```text
Refactor file <path_file> agar lebih rapi tanpa mengubah perilaku.

Target:
- Kurangi duplikasi.
- Pecah fungsi yang terlalu panjang.
- Perjelas nama variabel/fungsi internal.

Batasan:
- Tidak mengubah public interface.
- Tidak menambah dependency baru.
- Perubahan seminimal mungkin.

Keluaran:
1) daftar perubahan singkat,
2) patch,
3) risiko regresi yang perlu dicek.
```

### B. Refactor lintas agen dengan kontrak tetap

```text
Saya ingin merapikan pola yang berulang di:
- src/agents/classifier_agent.py
- src/agents/mitigation_agent.py
- src/agents/sociology_agent.py
- src/agents/economy_agent.py

Tujuan:
- Konsistenkan pola logging, error handling, dan format output.

Batasan keras:
- Skema output setiap agen tetap kompatibel dengan main_orchestrator.py.
- Jangan ubah alur bisnis inti.

Berikan:
1) rencana refactor kecil bertahap (maks 3 tahap),
2) patch tahap 1 dulu,
3) cara validasi cepat setelah tahap 1.
```

---

## 3) UNIT TEST

### A. Buat unit test untuk fungsi spesifik

```text
Buat unit test untuk fungsi <nama_fungsi> di <path_file>.

Kebutuhan:
- Framework: pytest.
- Minimal 3 skenario: normal case, edge case, invalid input.
- Gunakan mocking jika ada panggilan API/LLM.

Batasan:
- Test harus deterministic.
- Jangan memanggil layanan eksternal sungguhan.

Output:
1) file test yang dibuat,
2) command menjalankan test,
3) penjelasan singkat cakupan test.
```

### B. Tambah regression test dari bug nyata

```text
Berikut bug yang sudah terjadi:
- Lokasi: <path_file>
- Gejala: <bug_desc>
- Input pemicu: <sample_input>

Tolong:
1) buat regression test yang gagal di kode lama,
2) buat patch fix minimal,
3) pastikan test lulus setelah fix.

Gunakan pytest dan hindari dependensi eksternal.
```

### C. Test untuk RAG engine (tanpa internet)

```text
Buat test untuk src/tools/rag_engine.py dengan fokus:
- inisialisasi komponen,
- query retrieval,
- handling saat koleksi kosong/error.

Batasan:
- Jangan akses layanan jaringan.
- Mock komponen yang berat.

Keluaran:
- Struktur test + fixture yang disarankan,
- patch test minimal,
- perintah run test terfokus.
```

---

## 4) DOKUMENTASI

### A. Update README setelah perubahan kode

```text
Tolong update README.md berdasarkan perubahan terbaru di <file/fitur>.

Yang perlu diperbarui:
- Ringkasan fitur,
- cara menjalankan,
- contoh command,
- batasan/known issue (jika ada).

Gaya:
- Bahasa Indonesia,
- ringkas, langsung pakai,
- konsisten dengan struktur README yang ada.

Output:
1) daftar section yang diubah,
2) patch README.
```

### B. Tulis docstring untuk modul/fungsi

```text
Tambahkan/rapikan docstring di <path_file> untuk fungsi:
- <fungsi_1>
- <fungsi_2>

Aturan:
- Jelaskan tujuan, parameter, return, dan error penting.
- Tidak mengubah logika kode.
- Format konsisten di seluruh file.

Keluaran:
- patch docstring saja (tanpa refactor lain).
```

### C. Buat changelog ringkas per sesi kerja

```text
Buat ringkasan changelog dari perubahan hari ini.

Format:
- Added
- Changed
- Fixed

Sumber:
- file yang diubah: <daftar_file>
- tujuan perubahan: <tujuan>

Output maksimal 12 bullet, siap tempel ke PR description.
```

---

## 5) TEMPLATE KOMBO (HEMAT KUOTA)

Gunakan ini jika ingin **sekali prompt langsung end-to-end**:

```text
Project: Agentic-Musrenbang-DSS
Task: <debug/refactor/test/docs>
Lokasi utama: <file_1>, <file_2>
Masalah/tujuan: <deskripsi_singkat>
Batasan: patch minimal, jangan ubah API publik, tanpa dependency baru

Saya butuh output berurutan:
1) diagnosis singkat,
2) patch yang diperlukan,
3) test/verifikasi paling minimum,
4) update dokumentasi yang relevan.

Jika ada asumsi, tulis eksplisit sebelum patch.
```

---

## 6) Checklist Anti-Boros Copilot Free

- Tempel hanya error dan cuplikan kode yang relevan (jangan seluruh log panjang).
- Nyatakan batasan sejak awal: `patch minimal`, `tanpa ubah API`, `tanpa dependency baru`.
- Minta format keluaran tetap: diagnosis → patch → verifikasi.
- Hindari regenerasi file penuh jika cukup ubah beberapa baris.
- Simpan prompt yang berhasil sebagai snippet untuk dipakai ulang.
