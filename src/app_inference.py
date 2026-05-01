import os
import sys

# src dizinindeki diğer dosyaları import edebilmek için yolu ekliyoruz
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sentence_transformers import SentenceTransformer
from nlp_pipeline import preprocess_ingredient_list
from similarity_engine import SimilarityEngine

class RecipeInferencePipeline:
    def __init__(self, embeddings_path, recipes_path):
        """
        Adım 1: Lokal Model Kurulumu
        1. kişinin kullandığı modelin aynısı (all-MiniLM-L6-v2) lokal koda entegre edilir.
        """
        print("Çıkarım Boru Hattı (Inference Pipeline) başlatılıyor...")
        
        print("[Inference Pipeline] Embedding modeli (all-MiniLM-L6-v2) yükleniyor...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 2. kişinin yazdığı tavsiye motorunu başlat
        print("[Inference Pipeline] Tavsiye Motoru (Similarity Engine) başlatılıyor...")
        self.similarity_engine = SimilarityEngine(embeddings_path, recipes_path)
        
        print("Çıkarım Boru Hattı hazır!\n")

    def run_pipeline(self, user_input, top_n=5):
        """
        Kullanıcının yazdığı ilk metinden, tavsiye edilen son yemeğe kadar olan köprüyü kurar.
        """
        # Adım 4: Hata Yönetimi - Girdi Kontrolleri
        if not user_input or not isinstance(user_input, str):
            return {"error": "Lütfen geçerli bir metin girin.", "results": []}
            
        if user_input.strip().isnumeric():
            return {"error": "Sadece sayılardan oluşan bir arama yapılamaz. Lütfen malzeme isimleri girin.", "results": []}

        try:
            # Adım 2: Girdiyi Vektöre Çevirme
            # Kullanıcının girdiği ham metni alın ve NLP temizleme/lemmatization fonksiyonlarından geçirin
            processed_input = preprocess_ingredient_list(user_input)
            
            # Preprocess sonucu listeyse stringe çeviriyoruz
            if isinstance(processed_input, list):
                processed_text = " ".join(processed_input)
            else:
                processed_text = processed_input
                
            # NLP temizliğinden sonra geriye boş bir metin kalırsa hata yönetimi devreye girsin
            if not processed_text or processed_text.strip() == "":
                return {"error": "Girdiğiniz metin anlamlı bir malzeme içermiyor (sadece ölçü birimleri veya anlamsız kelimeler kalmış olabilir). Lütfen geçerli malzemeler girin.", "results": []}
                
            # Çıkan temiz metni modelinize vererek kullanıcının 384 boyutlu vektörünü oluşturun
            user_vector = self.model.encode([processed_text])
            
            # Adım 3: Sistemi Birleştirme
            # Elde edilen bu kullanıcı vektörünü alın ve 2. kişinin yazdığı tavsiye motoru fonksiyonuna (similarity arama kısmına) parametre olarak gönderin
            recommendations = self.similarity_engine.find_similar_recipes_by_vector(user_vector, top_n=top_n)
            
            return {
                "success": True,
                "clean_ingredients": processed_text,
                "results": recommendations
            }
            
        except Exception as e:
            # Sistem çökmemesi için genel exception handling
            return {"error": f"Boru hattı işleyişinde beklenmeyen bir hata oluştu: {str(e)}", "results": []}

if __name__ == "__main__":
    # Test Modülü
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    embeddings_file = os.path.join(base_dir, "data", "recipe_embeddings.pkl")
    recipes_file = os.path.join(base_dir, "data", "full_format_recipes.json")
    
    try:
        pipeline = RecipeInferencePipeline(embeddings_file, recipes_file)
        
        # Test 1: Başarılı Senaryo
        print("\n" + "="*50)
        print("TEST 1: Normal Girdi")
        res1 = pipeline.run_pipeline("chicken, fresh tomatoes, a little bit of salt")
        print("Durum:", "Başarılı" if "success" in res1 else "Hata")
        print("Temizlenen Malzemeler:", res1.get("clean_ingredients", ""))
        for i, rec in enumerate(res1.get("results", []), 1):
            print(f"{i}. {rec['title']} (%{rec['match_percentage']})")
            
        # Test 2: Hatalı Senaryo (Sadece Sayı)
        print("\n" + "="*50)
        print("TEST 2: Sadece Sayı")
        res2 = pipeline.run_pipeline("123456")
        print("Hata Mesajı:", res2.get("error"))
        
        # Test 3: Hatalı Senaryo (Sadece Stopwords / Anlamsız)
        print("\n" + "="*50)
        print("TEST 3: Anlamsız/Stopwords Girdi")
        res3 = pipeline.run_pipeline("and or with a the")
        print("Hata Mesajı:", res3.get("error"))
        
    except FileNotFoundError as e:
        print(f"Hata: Veri dosyaları bulunamadı. Lütfen data/ dizininde gerekli dosyaların olduğundan emin olun. Detay: {e}")
