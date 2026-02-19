import matplotlib.pyplot as plt
import numpy as np

# Data dari MLflow Anda
metrics = ['Akurasi', 'Penalaran']
scores = [5.80, 7.80]
cost_usd = 0.0064

# Setup Gambar
fig, ax1 = plt.subplots(figsize=(8, 5))

# ==========================================
# 1. WARNA BERBEDA UNTUK AKURASI & PENALARAN
# ==========================================
# Hex color: Biru (Akurasi), Oranye (Penalaran)
bar_colors = ['#1f77b4', '#ff7f0e'] 

# Pass list 'bar_colors' ke dalam argumen color
bars = ax1.bar(metrics, scores, color=bar_colors, width=0.4)

# Konfigurasi Sumbu Y Kiri
ax1.set_ylabel('Skor (Skala 0-10)', fontweight='bold')
ax1.set_ylim(0, 10)

# Tambahkan label angka di atas batang dengan warna yang senada
for bar, color in zip(bars, bar_colors):
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f"{yval:.2f}", 
             ha='center', va='bottom', color=color, fontweight='bold')

# ==========================================
# 2. SUMBU Y KEDUA (BIAYA SIMULASI)
# ==========================================
ax2 = ax1.twinx()  
color_cost = '#d62728' # Merah untuk Biaya

# Posisi sumbu X tiruan untuk Biaya (indeks ke-2)
ax2.bar(['Biaya Simulasi'], [cost_usd], color=color_cost, width=0.2)

# Konfigurasi Sumbu Y Kanan
ax2.set_ylabel('Biaya Eksekusi (USD)', color=color_cost, fontweight='bold')
ax2.set_ylim(0, 0.01) # Batas atas grafik agar proporsional
ax2.tick_params(axis='y', labelcolor=color_cost)

# Tambahkan label angka untuk biaya
ax2.text(2, cost_usd + 0.0002, f"${cost_usd}", 
         ha='center', va='bottom', color=color_cost, fontweight='bold')

# ==========================================
# 3. FORMATTING FINAL & EXPORT
# ==========================================
plt.title('Grafik 4.1. Perbandingan Kinerja Kognitif dan Biaya Komputasi', pad=20, fontweight='bold')

# Menghilangkan garis tepi atas agar lebih bersih (opsional ala grafik modern)
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

fig.tight_layout()

# Simpan sebagai PNG kualitas tinggi (300 dpi) untuk dokumen Tesis Word
plt.savefig('Grafik_Evaluasi_Bab4_Warna_Warni.png', dpi=300, bbox_inches='tight')
print("Grafik berhasil disimpan sebagai 'Grafik_Evaluasi_Bab4_Warna_Warni.png'")
plt.show()