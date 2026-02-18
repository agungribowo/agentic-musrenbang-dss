import os
import pandas as pd
import chromadb

# ==========================================
# PENGATURAN PATH OTOMATIS (Sangat Penting untuk Reproducibility)
# ==========================================
# Ini memastikan script selalu tahu di mana folder root proyek berada,
# tidak peduli dari folder mana Anda menjalankan script-nya di terminal.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

# Path ke file CSV Kamus Usulan dan Folder Database Vektor
CSV_FILENAME = "Kamus Usulan Renstramas.csv"
CSV_CANDIDATE_DIRS = ["external", "eksternal", "exsternal"]
DB_PATH = os.path.join(PROJECT_ROOT, "chroma_db")


def resolve_csv_path():
    for folder_name in CSV_CANDIDATE_DIRS:
        candidate = os.path.join(PROJECT_ROOT, "data", folder_name, CSV_FILENAME)
        if os.path.exists(candidate):
            return candidate

    return os.path.join(PROJECT_ROOT, "data", CSV_CANDIDATE_DIRS[0], CSV_FILENAME)


def load_kamus_csv(csv_path):
    read_attempts = [
        {"sep": ",", "encoding": "utf-8"},
        {"sep": ";", "encoding": "utf-8"},
        {"sep": ",", "encoding": "utf-8-sig"},
        {"sep": ";", "encoding": "utf-8-sig"},
    ]

    last_error = None
    for params in read_attempts:
        try:
            return pd.read_csv(csv_path, **params)
        except Exception as error:
            last_error = error

    raise last_error

class KamusRAG:
    """
    Mesin Pencari Semantik (RAG Engine) untuk Kamus Usulan Musrenbang.
    Tool ini akan dipanggil oleh Agen untuk mencari referensi nomenklatur.
    """
    def __init__(self):
        print("[RAG Engine] Menghubungkan ke Memori Vektor...")
        # Menggunakan PersistentClient agar data vektor tersimpan di hardisk
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.collection_name = "musrenbang_kamus"
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

        # Cek apakah database masih kosong. Jika ya, otomatis isi dari CSV.
        if self.collection.count() == 0:
            self._populate_db()
        else:
            print(f"[RAG Engine] Memori siap. Terdapat {self.collection.count()} nomenklatur aktif.")

    def _populate_db(self):
        """Membaca file CSV dan memasukkannya ke dalam ChromaDB."""
        csv_path = resolve_csv_path()
        print(f"[RAG Engine] Database kosong. Membaca data dari: {csv_path}")
        try:
            # Membaca CSV dengan fallback delimiter agar tahan variasi format export.
            df = load_kamus_csv(csv_path)
            
            # Memastikan kolom tidak ada yang kosong/NaN agar tidak error
            df = df.fillna("Tidak Terdefinisi")

            print("[RAG Engine] Sedang menerjemahkan teks ke dalam Vektor (Embedding)...")
            
            # Looping per baris sesuai dengan struktur file Excel/CSV Anda
            for index, row in df.iterrows():
                self.collection.add(
                    documents=[str(row['Kamus Usulan'])],
                    metadatas=[{
                        "dinas": str(row['Bidang Urusan Nomenklatur']), 
                        "bidang": str(row['Bidang Pembahasan Musrenbang'])
                    }],
                    ids=[str(row['No'])]
                )
            print(f"[RAG Engine] BERHASIL! {self.collection.count()} nomenklatur telah dihafal oleh sistem.")
            
        except FileNotFoundError:
            print("[ERROR] File CSV tidak ditemukan.")
            print(f"[ERROR] Nama file yang dicari: {CSV_FILENAME}")
            print(f"[ERROR] Folder yang dicek: data/{', data/'.join(CSV_CANDIDATE_DIRS)}")
        except Exception as e:
            print(f"[ERROR] Terjadi kegagalan sistem saat membaca CSV: {e}")

    def search_kamus(self, query_text, top_k=3):
        """
        Fungsi utama yang akan dipanggil oleh Agen.
        Mencari nomenklatur yang secara semantik paling dekat dengan keluhan warga.
        """
        print(f"[RAG Engine] Melakukan pencarian semantik untuk: '{query_text}'")
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k
        )
        
        # Merapikan format output agar mudah dibaca dan dipahami oleh LLM (Gemini)
        formatted_results = "KANDIDAT KAMUS USULAN:\n"
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                dinas = results['metadatas'][0][i]['dinas']
                bidang = results['metadatas'][0][i]['bidang']
                formatted_results += f"{i+1}. Usulan: {doc} | Bidang: {bidang} | Dinas: {dinas}\n"
        
        return formatted_results

# ==========================================
# BLOK PENGUJIAN LOKAL (Hanya berjalan jika file ini dieksekusi langsung)
# ==========================================
if __name__ == "__main__":
    # 1. Inisialisasi Mesin RAG
    rag = KamusRAG()
    
    # 2. Simulasi pencarian
    tes_keluhan = "Banyak ibu hamil di RW 23 yang kekurangan gizi dan butuh penanganan posyandu"
    hasil = rag.search_kamus(tes_keluhan, top_k=2)
    
    print("\n--- HASIL PENCARIAN SYSTEM ---")
    print(hasil)