import pandas as pd
import re

def clean_text(text):
    if not isinstance(text, str):
        return text
    
    # Küçük harfe çevirme
    text = text.lower()
    
    # Kesirli ifadeleri sil (örn: 1/2, 3/4)
    text = re.sub(r'\d+/\d+', '', text)
    
    # Ondalıklı (0.5) ve tam sayıları (1, 2 vb.) sil
    text = re.sub(r'\d+\.\d+|\d+', '', text)
    
    # Parantez içerisindeki spesifik gereksiz ifadeleri sil (örn: (optional))
    text = re.sub(r'\(optional\)', '', text)
    
    # Noktalama işaretlerini (parantezler, virgül, tire vb.) temizle
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Fazla boşlukları temizle
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def process_ingredients(ingredients):
    if isinstance(ingredients, list):
        # Liste ise her bir maddeyi temizle
        return [clean_text(ing) for ing in ingredients]
    elif isinstance(ingredients, str):
        return clean_text(ingredients)
    return ingredients

def basic_cleaning(json_path):
    """
    JSON dosyasını okuyup, temel temizlik adımlarını uygular.
    """
    print(f"'{json_path}' okunuyor...")
    # Adım 1: Veriyi Yükleme
    df = pd.read_json(json_path)
    
    # İçerisinde malzeme listesi (ingredients) veya tarif adı (title) boş olan satırları silin
    df = df.dropna(subset=['ingredients', 'title'])
    
    # Adım 2 & 3: Küçük Harfe Çevirme, Sayı ve Noktalama Temizliği
    print("Metin temizliği yapılıyor...")
    df['ingredients'] = df['ingredients'].apply(process_ingredients)
    
    print("Temizlik tamamlandı.")
    return df

if __name__ == "__main__":
    # Test çalıştırması
    test_text = "2 (10 ounce) packages of fresh tomatoes, chopped"
    print(f"Test Metni: {test_text}")
    print(f"Temizlenmiş Hali: {clean_text(test_text)}")
    
    # Dosyayı test et
    try:
        df_cleaned = basic_cleaning('data/full_format_recipes.json')
        print(df_cleaned[['title', 'ingredients']].head())
    except Exception as e:
        print("Hata oluştu:", e)
