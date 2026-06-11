import pandas as pd
import numpy as np
import pickle
import os
import sys
import ast
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# nlp_pipeline'dan preprocess fonksiyonunu içeri aktaralım
# (scriptin src içinden veya ana dizinden çalıştırılma durumunu idare etmek için path ekliyoruz)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from nlp_pipeline import preprocess_ingredient_list
except ImportError as e:
    print(f"nlp_pipeline import hatası: {e}. Lütfen nlp_pipeline.py dosyasının aynı dizinde olduğundan emin olun.")

class SimilarityEngine:
    def __init__(self, embeddings_path, recipes_path, model=None):
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
            
        # 2. Orijinal veriyi yükle (Temizlenmiş CSV varsa onu tercih ediyoruz, yoksa JSON yüklüyoruz)
        csv_path = recipes_path.replace('full_format_recipes.json', 'cleaned_recipes.csv') if isinstance(recipes_path, str) else ""
        if csv_path and os.path.exists(csv_path):
            print(f"Orijinal tarif verisi (CSV) yükleniyor: {csv_path}")
            self.df = pd.read_csv(csv_path)
        else:
            print(f"Orijinal tarif verisi (JSON) yükleniyor: {recipes_path}")
            self.df = pd.read_json(recipes_path)
            
        # Kullanıcı girdisini vektörleştirmek için SentenceTransformer modeli (Bellek tasarrufu için paylaşılabilir)
        if model is not None:
            print("Verilen model nesnesi kullanılıyor (Bellek optimizasyonu aktif).")
            self.model = model
        else:
            print("Yeni embedding modeli (all-MiniLM-L6-v2) yükleniyor...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
        # TF-IDF Vectorizer Kurulumu (Hibrit Arama için)
        print("TF-IDF Vectorizer eğitiliyor...")
        self.tfidf_vectorizer = TfidfVectorizer()
        
        # Preprocessed malzemeleri metin haline getirip corpus oluşturuyoruz
        processed_corpus = []
        # Eğer csv'den yüklediyse liste olanlar string temsilindedir, dönüştürüyoruz
        for val in self.df['ingredients_processed']:
            if not isinstance(val, str):
                processed_corpus.append("")
                continue
            if val.startswith('['):
                try:
                    lst = ast.literal_eval(val)
                    processed_corpus.append(" ".join(lst))
                except:
                    processed_corpus.append(val)
            else:
                processed_corpus.append(val)
                
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_corpus)
        print("TF-IDF Hazır!")
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

    def find_similar_recipes_by_vector(self, user_vector, processed_text=None, top_n=5, alpha=1.0):
        """
        Adım 2: Kosinüs Benzerliği (Semantic + Lexical Hybrid)
        Adım 3: Sıralama Algoritması
        Adım 4: Formatlama & Malzeme Eşleşme Analizi
        """
        # 1. Semantik Benzerlik (Cosine Similarity)
        semantic_similarities = cosine_similarity(user_vector, self.embeddings)[0]
        
        # 2. Kelime Bazlı Benzerlik (TF-IDF Cosine Similarity)
        if processed_text:
            query_tfidf = self.tfidf_vectorizer.transform([processed_text])
            tfidf_similarities = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
            
            # Hibrit Skor (Alpha ağırlıklı)
            similarities = alpha * semantic_similarities + (1 - alpha) * tfidf_similarities
        else:
            similarities = semantic_similarities
        
        # Adım 3: Sıralama Algoritması
        top_indices_positions = np.argsort(similarities)[::-1][:top_n]
        
        # Adım 4: Formatlama
        results = []
        user_words = []
        if processed_text:
            user_words = [w.strip() for w in processed_text.split() if w.strip()]
            
        for pos in top_indices_positions:
            original_index = self.indices[pos]
            score = similarities[pos]
            
            recipe_data = self.df.loc[original_index]
            
            # Orijinal malzemeleri listeye çevirme (CSV'den string geldiyse çözüyoruz)
            ingredients_val = recipe_data.get('ingredients', [])
            if isinstance(ingredients_val, str):
                try:
                    ingredients_val = ast.literal_eval(ingredients_val)
                except:
                    pass
            
            # İşlenmiş malzemeleri listeye çevirme
            proc_val = recipe_data.get('ingredients_processed', [])
            if isinstance(proc_val, str):
                try:
                    proc_val = ast.literal_eval(proc_val)
                except:
                    pass
            if not isinstance(proc_val, list):
                proc_val = str(proc_val).split()
            
            # Yapılış talimatlarını çekme (Orijinal sütun ismi 'directions')
            directions_val = recipe_data.get('directions', [])
            if isinstance(directions_val, str):
                try:
                    directions_val = ast.literal_eval(directions_val)
                except:
                    pass
            
            if isinstance(directions_val, list):
                instructions_str = "\n".join(directions_val)
            else:
                instructions_str = str(directions_val)
                
            # Malzeme Eşleşme Analizi (Hangi malzemeler var, hangileri eksik?)
            matched = []
            missing = []
            
            recipe_words = set()
            for ing in proc_val:
                recipe_words.update(ing.split())
                
            for word in user_words:
                if word in recipe_words:
                    matched.append(word)
                else:
                    missing.append(word)
            
            formatted_result = {
                "title": recipe_data.get('title', 'Bilinmeyen Tarif'),
                "ingredients": ingredients_val,
                "instructions": instructions_str,
                "match_percentage": round(score * 100, 2),
                "matched_ingredients": matched,
                "missing_ingredients": missing
            }
            results.append(formatted_result)
            
        return results

    def find_similar_recipes(self, user_input, top_n=5):
        """
        Gelen metni önce vektöre çevirir, sonra benzerlerini bulur.
        """
        user_vector = self.get_user_vector(user_input)
        
        processed_input = preprocess_ingredient_list(user_input)
        if isinstance(processed_input, list):
            processed_text = " ".join(processed_input)
        else:
            processed_text = processed_input
            
        return self.find_similar_recipes_by_vector(user_vector, processed_text=processed_text, top_n=top_n)


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
