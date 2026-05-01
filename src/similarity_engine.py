import pandas as pd
import numpy as np
import pickle
import os
import sys
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# nlp_pipeline'dan preprocess fonksiyonunu içeri aktaralım
# (scriptin src içinden veya ana dizinden çalıştırılma durumunu idare etmek için path ekliyoruz)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from nlp_pipeline import preprocess_ingredient_list
except ImportError as e:
    print(f"nlp_pipeline import hatası: {e}. Lütfen nlp_pipeline.py dosyasının aynı dizinde olduğundan emin olun.")

class SimilarityEngine:
    def __init__(self, embeddings_path, recipes_path):
        """
        Adım 1: Dosyaları Yükleme
        1. kişinin paylaştığı vektör dosyasını ve orijinal veri setini lokal projeye yükleyen fonksiyon.
        """
        print("Tavsiye Motoru başlatılıyor...")
        
        # 1. Vektörleri (embeddings) ve indeksleri yükle
        print(f"Vektörler yükleniyor: {embeddings_path}")
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
            self.embeddings = data['embeddings']
            self.indices = data['index']
            
        # 2. Orijinal veriyi yükle (JSON formatındaki ham tarifler)
        print(f"Orijinal tarif verisi yükleniyor: {recipes_path}")
        self.df = pd.read_json(recipes_path)
        
        # Kullanıcı girdisini vektörleştirmek için SentenceTransformer modeli
        print("Embedding modeli (all-MiniLM-L6-v2) yükleniyor...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Tavsiye Motoru hazır!\n")

    def get_user_vector(self, user_input):
        """
        Kullanıcı girdisini alır, NLP pipeline'dan geçirir ve vektöre (embedding) dönüştürür.
        """
        # 1. Girdiyi NLP pipeline'ından (temizlik, stopword, lemma vb.) geçir
        processed_input = preprocess_ingredient_list(user_input)
        
        # Eğer preprocess liste döndürdüyse boşlukla birleştirip string yap
        if isinstance(processed_input, list):
            processed_input = " ".join(processed_input)
            
        # 2. String'i vektöre çevir
        vector = self.model.encode([processed_input])
        return vector

    def find_similar_recipes_by_vector(self, user_vector, top_n=5):
        """
        Adım 2: Kosinüs Benzerliği
        Adım 3: Sıralama Algoritması
        Adım 4: Formatlama
        (Bu metot 3. kişi olan Inference Pipeline tarafından doğrudan vektör ile çağrılmak üzere ayrılmıştır.)
        """
        # Adım 2: Kosinüs Benzerliği (Cosine Similarity)
        # user_vector şekli (1, 384), self.embeddings şekli (N, 384)
        similarities = cosine_similarity(user_vector, self.embeddings)[0]
        
        # Adım 3: Sıralama Algoritması
        # np.argsort küçükten büyüğe sıralar. Biz en yüksek skoru aradığımız için [::-1] ile ters çeviriyoruz.
        # Ardından ilk 'top_n' (örn: 5) indeksi seçiyoruz.
        top_indices_positions = np.argsort(similarities)[::-1][:top_n]
        
        # Adım 4: Formatlama
        # JSON arayüzüne gönderilecek formatı hazırlama
        results = []
        for pos in top_indices_positions:
            # Pkl dosyasındaki index sırası ile orijinal DataFrame'deki index numarasını eşliyoruz
            original_index = self.indices[pos]
            score = similarities[pos]
            
            # Tarif detaylarını DataFrame'den çekiyoruz
            recipe_data = self.df.loc[original_index]
            
            formatted_result = {
                "title": recipe_data.get('title', 'Bilinmeyen Tarif'),
                "ingredients": recipe_data.get('ingredients', []),
                "instructions": recipe_data.get('instructions', ''),
                "match_percentage": round(score * 100, 2) # Yüzdelik formata çevirme (örn: 85.45)
            }
            results.append(formatted_result)
            
        return results

    def find_similar_recipes(self, user_input, top_n=5):
        """
        Gelen metni önce vektöre çevirir, sonra benzerlerini bulur.
        """
        # Kullanıcı vektörünü oluştur
        user_vector = self.get_user_vector(user_input)
        
        # Doğrudan vektör üzerinden çalışan metoda gönder
        return self.find_similar_recipes_by_vector(user_vector, top_n)


# ------------------------------------------------------------------
# Test / Demo
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Dosya yollarını ayarla (src klasörünün bir üstündeki data klasörü)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    embeddings_file = os.path.join(base_dir, "data", "recipe_embeddings.pkl")
    recipes_file = os.path.join(base_dir, "data", "full_format_recipes.json")
    
    # Motoru başlat (Sadece bir kere başlatılmalı, sonrasında fonksiyonlar defalarca çağrılabilir)
    try:
        engine = SimilarityEngine(embeddings_file, recipes_file)
        
        # Test girdisi (Kullanıcının buzdolabındaki malzemeler)
        test_input = "chicken breast, fresh tomatoes, garlic, olive oil, basil"
        print("=" * 60)
        print(f"Test Girdisi: {test_input}")
        print("=" * 60)
        
        # Benzer tarifleri bul
        recommendations = engine.find_similar_recipes(test_input, top_n=5)
        
        # Sonuçları ekrana yazdır
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. Tarif: {rec['title']}")
            print(f"Eşleşme Yüzdesi: %{rec['match_percentage']}")
            print(f"Ham Malzemeler : {', '.join(rec['ingredients'][:4])} ...") # Sadece ilk 4'ünü göster
            print(f"Talimat Özeti  : {str(rec['instructions'])[:100]}...") # Sadece ilk 100 karakteri göster
            print("-" * 60)
            
    except FileNotFoundError as e:
        print(f"\nHata: {e}")
        print("Lütfen veri dosyalarının doğru dizinde (data/) olduğundan emin olun.")
