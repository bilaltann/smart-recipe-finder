"""
nlp_pipeline.py
---------------
3. Kişi: İleri NLP ve Veri Boru Hattı (NLP Pipeline)

Görev:
    - Adım 1: İngilizce gereksiz bağlaçları (stopwords) NLTK ile temizle.
    - Adım 2: NLTK WordNetLemmatizer ile kelimeleri kök formlarına indirge.
    - Adım 3: basic_cleaning.py ve culinary_filtering.py fonksiyonlarını birleştirerek
              tam bir preprocess_ingredient_list pipeline'ı oluştur.

Bağımlılıklar:
    pip install nltk
    İlk çalıştırmada gerekli NLTK verileri otomatik indirilir.
"""

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# 1. ve 2. kişinin modülleri import ediliyor
from basic_cleaning import process_ingredients, basic_cleaning
from culinary_filtering import filter_culinary_terms

# ------------------------------------------------------------------
# NLTK kaynaklarını bir kez indir (zaten varsa geçer)
# ------------------------------------------------------------------
def _ensure_nltk_resources():
    """Gerekli NLTK kaynaklarını indirir; zaten mevcutsa sessizce geçer."""
    resources = [
        ("corpora/stopwords",       "stopwords"),
        ("corpora/wordnet",         "wordnet"),
        ("corpora/omw-1.4",         "omw-1.4"),
    ]
    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"[NLTK] '{pkg}' indiriliyor...")
            nltk.download(pkg, quiet=True)

_ensure_nltk_resources()

# ------------------------------------------------------------------
# Sabitler
# ------------------------------------------------------------------

# Proje kapsamında kaldırılacak İngilizce stopword'ler.
# Malzeme isimleri için kritik olan "no", "not", "without" gibi
# anlam taşıyan kelimeler listeden çıkarılmıştır.
_BASE_STOPWORDS = set(stopwords.words("english"))

# Malzeme bağlamında anlam taşıyan kelimeleri stopword listesinden koru
_KEEP_WORDS = {
    "no", "not", "without", "free", "low", "less",
    "more", "up", "down", "light", "dark", "extra",
    "sweet", "hot", "raw", "cold"
}
CULINARY_STOPWORDS = _BASE_STOPWORDS - _KEEP_WORDS

# Lemmatizer örneği (tek seferlik oluştur, her çağrıda yeniden yaratma)
_lemmatizer = WordNetLemmatizer()


# ------------------------------------------------------------------
# Adım 1: Stopwords Temizliği
# ------------------------------------------------------------------

def remove_stopwords(text: str) -> str:
    """
    Verilen metinden NLTK İngilizce stopword'lerini kaldırır.

    Kaldırılan başlıca kelimeler: "of", "and", "or", "with", "a", "an",
    "the", "in", "to", "for", "from", "at", vb.

    Parametreler:
        text (str): Temizlenecek ham metin.

    Döndürür:
        str: Stopword'lerden arındırılmış metin.

    Örnek:
        >>> remove_stopwords("cup of flour and a handful of sugar")
        'cup flour handful sugar'
    """
    if not isinstance(text, str):
        return text

    tokens = text.split()
    filtered = [w for w in tokens if w not in CULINARY_STOPWORDS]
    return " ".join(filtered)


# ------------------------------------------------------------------
# Adım 2: Lemmatization (Kök Bulma)
# ------------------------------------------------------------------

def lemmatize_text(text: str) -> str:
    """
    Metindeki her kelimeyi NLTK WordNetLemmatizer ile isim kök formuna
    indirger. Malzeme isimleri daima isim (NOUN) kategorisindedir;
    bu nedenle pos='n' sabit kullanımı hem daha hızlı hem de doğrudur.

    Çoğul ekleri kaldırır:
        "tomatoes" → "tomato"
        "carrots"  → "carrot"
        "onions"   → "onion"
        "berries"  → "berry"

    Parametreler:
        text (str): Lemmatize edilecek metin.

    Döndürür:
        str: Kök formlarından oluşan metin.

    Örnek:
        >>> lemmatize_text("tomatoes carrots onions")
        'tomato carrot onion'
    """
    if not isinstance(text, str):
        return text

    tokens = text.split()
    # pos='n' → isim (noun) olarak lemmatize et;
    # malzeme sözcükleri için en doğru ve en hızlı yaklaşım.
    lemmatized = [
        _lemmatizer.lemmatize(word, pos=wordnet.NOUN)
        for word in tokens
    ]
    return " ".join(lemmatized)


def nlp_process(text: str) -> str:
    """
    3. kişinin NLP adımlarını sırasıyla uygular:
        1. Stopwords temizliği
        2. Lemmatization

    Parametreler:
        text (str): İşlenecek ham metin.

    Döndürür:
        str: NLP işlemlerinden geçirilmiş temiz metin.
    """
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text


# ------------------------------------------------------------------
# Adım 3: Tam Pipeline (Boru Hattı)
# ------------------------------------------------------------------

def preprocess_ingredient_list(ingredients):
    """
    Bir tarife ait malzeme listesini (veya tek bir malzeme metnini) alır;
    üç aşamalı tam pipeline'ı uygulayarak derin öğrenme modeline
    hazır, temiz bir çıktı döndürür.

    Pipeline Akışı:
        ┌──────────────────────────────────────────────────┐
        │ 1. Temel Temizlik  (1. Kişi — basic_cleaning)    │
        │    • Küçük harfe çevirme                         │
        │    • Sayı silme (1/2, 0.5, 3 vb.)               │
        │    • Noktalama temizliği                         │
        ├──────────────────────────────────────────────────┤
        │ 2. Mutfak Filtresi (2. Kişi — culinary_filtering)│
        │    • Ölçü birimlerini sil (cup, oz, tbsp …)      │
        │    • Hazırlık terimlerini sil (chopped, diced …) │
        ├──────────────────────────────────────────────────┤
        │ 3. NLP & Lemmatization (3. Kişi — nlp_pipeline)  │
        │    • İngilizce stopword'leri sil                 │
        │    • Kelimeleri kök formuna indirge              │
        └──────────────────────────────────────────────────┘

    Parametreler:
        ingredients (list[str] | str):
            • Liste gelirse → her eleman işlenir, liste döner.
            • Metin gelirse → tek metin işlenir, string döner.

    Döndürür:
        list[str] | str: Pipeline'dan geçirilmiş malzeme(ler).

    Örnek:
        >>> preprocess_ingredient_list(["2 cups of fresh tomatoes, chopped",
        ...                             "1/2 teaspoon salt"])
        ['tomato', 'salt']
    """
    if isinstance(ingredients, list):
        result = []
        for ingredient in ingredients:
            # --- Aşama 1: Temel Temizlik ---
            cleaned = process_ingredients(ingredient)

            # --- Aşama 2: Mutfak Filtresi ---
            filtered = filter_culinary_terms(cleaned)

            # --- Aşama 3: NLP & Lemmatization ---
            nlp_result = nlp_process(filtered)

            # Boş kalan malzemeleri listeye ekleme
            if nlp_result.strip():
                result.append(nlp_result.strip())

        return result

    elif isinstance(ingredients, str):
        cleaned  = process_ingredients(ingredients)
        filtered = filter_culinary_terms(cleaned)
        return nlp_process(filtered)

    # Desteklenmeyen tipleri olduğu gibi geri döndür
    return ingredients


def preprocess_dataset(df):
    """
    Tüm DataFrame'e pipeline'ı uygular.

    basic_cleaning() ile yüklenmiş bir DataFrame bekler.
    'ingredients' sütununu preprocess_ingredient_list ile işler ve
    'ingredients_processed' adlı yeni sütuna yazar.

    Parametreler:
        df (pd.DataFrame): 'ingredients' sütunu içeren DataFrame.

    Döndürür:
        pd.DataFrame: 'ingredients_processed' sütunu eklenmiş DataFrame.
    """
    print("NLP Pipeline uygulanıyor...")
    df = df.copy()
    df["ingredients_processed"] = df["ingredients"].apply(
        preprocess_ingredient_list
    )
    print("Pipeline tamamlandı.")
    return df


# ------------------------------------------------------------------
# Test / Demo
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("NLP Pipeline — Adım Adım Demo")
    print("=" * 60)

    test_cases = [
        "2 cups of fresh tomatoes, chopped",
        "1/2 teaspoon of salt and pepper",
        "3 tablespoons unsalted butter, melted",
        "4 large onions, sliced and diced",
        "1 (10 ounce) package of frozen spinach, thawed",
    ]

    for raw in test_cases:
        # Adım 1
        step1 = process_ingredients(raw)
        # Adım 2
        step2 = filter_culinary_terms(step1)
        # Adım 3
        step3 = nlp_process(step2)

        print(f"\n  Ham Girdi  : {raw}")
        print(f"  1) Temizlik: {step1}")
        print(f"  2) Filtre  : {step2}")
        print(f"  3) NLP     : {step3}")

    print("\n" + "=" * 60)
    print("preprocess_ingredient_list() — Tek Fonksiyon Testi")
    print("=" * 60)

    sample_list = [
        "2 cups of fresh tomatoes, chopped",
        "1/2 teaspoon salt",
        "3 large carrots, peeled and sliced",
        "1 pound boneless chicken breasts",
    ]

    processed = preprocess_ingredient_list(sample_list)
    for original, result in zip(sample_list, processed):
        print(f"  {original!r:45s}  ->  {result!r}")

    # Gerçek veri seti testi (dosya varsa)
    print("\n" + "=" * 60)
    print("Gerçek Veri Seti Testi")
    print("=" * 60)
    try:
        df_cleaned = basic_cleaning("data/full_format_recipes.json")
        df_processed = preprocess_dataset(df_cleaned)
        print(df_processed[["title", "ingredients", "ingredients_processed"]].head(3).to_string())
    except FileNotFoundError:
        print("Veri seti bulunamadı. Lütfen 'data/full_format_recipes.json' yolunu kontrol edin.")
    except Exception as e:
        print(f"Hata: {e}")
