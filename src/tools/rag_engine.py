import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# ==========================================
# PENGATURAN PATH OTOMATIS (Sangat Penting untuk Reproducibility)
# ==========================================
# Ini memastikan script selalu tahu di mana folder root proyek berada,
# tidak peduli dari folder mana Anda menjalankan script-nya di terminal.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

# Path ke file CSV Kamus Usulan dan Folder Database Vektor
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "external", "Kamus Usulan Renstramas.csv")
DB_PATH = os.path.join(PROJECT_ROOT, "chroma_db")
REQUIRED_COLUMNS = [
    "No",
    "Bidang Pembahasan Musrenbang",
    "Kamus Usulan",
    "Bidang Urusan Nomenklatur",
]

def load_kamus_csv(csv_path):
    read_attempts = [
        {"sep": ";", "encoding": "utf-8"},
        {"sep": ";", "encoding": "utf-8-sig"},
        {"sep": ",", "encoding": "utf-8"},
        {"sep": ",", "encoding": "utf-8-sig"},
        {"sep": None, "encoding": "utf-8", "engine": "python"},
        {"sep": None, "encoding": "utf-8-sig", "engine": "python"},
    ]

    last_error = None
    for params in read_attempts:
        try:
            df = pd.read_csv(csv_path, **params)
            if set(REQUIRED_COLUMNS).issubset(df.columns):
                return df
        except Exception as error:
            last_error = error

    if last_error:
        raise ValueError(
            f"CSV terbaca tapi kolom wajib tidak sesuai. Kolom wajib: {REQUIRED_COLUMNS}"
        ) from last_error
    raise ValueError("CSV tidak dapat dibaca. Periksa delimiter/encoding file.")

class KamusRAG:
    """
    Mesin Pencari Semantik (RAG Engine) untuk Kamus Usulan Musrenbang.
    """
    def __init__(self):
        print("[RAG Engine] Menghubungkan ke Memori Vektor...")
        self.client = chromadb.PersistentClient(path=DB_PATH)
        
        # ---------------------------------------------------------
        # UPGRADE: Menggunakan Model Multilingual (Paham Bahasa Indonesia)
        # ---------------------------------------------------------
        indonesian_model = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        self.collection_name = "musrenbang_kamus_id"
        
        # Memasukkan model bahasa Indonesia ke dalam database
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=indonesian_model
        )

        if self.collection.count() == 0:
            self._populate_db()
        else:
            print(f"[RAG Engine] Memori siap. Terdapat {self.collection.count()} nomenklatur aktif.")

    def _populate_db(self):
        print(f"[RAG Engine] Database kosong. Membaca data dari: {CSV_PATH}")
        try:
            df = load_kamus_csv(CSV_PATH)
            df = df.fillna("Tidak Terdefinisi")

            print("[RAG Engine] Sedang menerjemahkan teks ke Vektor Multilingual (Proses ini butuh unduh model sekitar 400MB)...")
            
            for index, row in df.iterrows():
                self.collection.add(
                    documents=[str(row['Kamus Usulan'])],
                    metadatas=[{
                        "dinas": str(row['Bidang Urusan Nomenklatur']), 
                        "bidang": str(row['Bidang Pembahasan Musrenbang'])
                    }],
                    ids=[str(row['No'])]
                )
            print(f"[RAG Engine] BERHASIL! {self.collection.count()} nomenklatur telah dihafal dengan model Bahasa Indonesia.")
            
        except FileNotFoundError:
            print(f"[ERROR] File CSV tidak ditemukan di: {CSV_PATH}")
        except Exception as e:
            print(f"[ERROR] Terjadi kegagalan sistem saat membaca CSV: {e}")

    def search_kamus(self, query_text, top_k=3):
        print(f"\n[RAG Engine] Melakukan pencarian semantik untuk: '{query_text}'")
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k
        )
        
        formatted_results = "KANDIDAT KAMUS USULAN:\n"
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                dinas = results['metadatas'][0][i]['dinas']
                bidang = results['metadatas'][0][i]['bidang']
                formatted_results += f"{i+1}. Usulan: {doc} | Bidang: {bidang} | Dinas: {dinas}\n"
        
        return formatted_results

if __name__ == "__main__":
    rag = KamusRAG()
    
    # Tes Keluhan
    tes_keluhan = "Banyak ibu hamil di RW 23 yang kekurangan gizi dan butuh penanganan posyandu"
    hasil = rag.search_kamus(tes_keluhan, top_k=3)
    
    print("\n--- HASIL PENCARIAN SYSTEM MULTILINGUAL ---")
    print(hasil)