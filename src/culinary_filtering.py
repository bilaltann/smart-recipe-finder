# Ölçü Birimleri ve Boyut Sıfatları Sözlüğü
# JSON veri setinden frekans analizine göre genişletilmiştir.
MEASUREMENTS = [
    'cup', 'cups', 'tablespoon', 'tablespoons', 'teaspoon', 'teaspoons',
    'ounce', 'ounces', 'oz', 'pound', 'pounds', 'lb', 'lbs', 
    'stick', 'sticks', 'inch', 'inches', 'package', 'packages', 
    'pinch', 'large', 'medium', 'small', 'whole'
]

# Durum/Hazırlık Terimleri Sözlüğü
# JSON veri setinden sık karşılaşılan hazırlık/durum kelimelerine göre genişletilmiştir.
PREP_TERMS = [
    'chopped', 'sliced', 'peeled', 'cut', 'minced', 'grated', 'divided',
    'halved', 'unsalted', 'dried', 'trimmed', 'drained', 'pieces', 
    'packed', 'seeded', 'toasted', 'diced', 'thinly', 'crumbled', 
    'quartered', 'finely', 'removed', 'pitted', 'crushed', 'rinsed', 
    'melted', 'thawed', 'cored', 'lengthwise', 'chilled', 'coarsely', 
    'softened', 'canned', 'fresh', 'boneless', 'skinless', 'ground',
    'room', 'temperature'
]

def filter_culinary_terms(text):
    """
    Temizlenmiş metni alıp, içindeki ölçü birimlerini ve "malzeme olmayan" 
    pişirme/durum terimlerini ayıklar.
    """
    if not isinstance(text, str):
        return text
        
    # Metni kelimelere ayır
    words = text.split()
    
    # Kelimeleri listelerle karşılaştır, olmayanları tut
    filtered_words = [
        word for word in words 
        if word not in MEASUREMENTS and word not in PREP_TERMS
    ]
    
    # Kelimeleri tekrar birleştir
    return " ".join(filtered_words)

# Test kısmı:
if __name__ == "__main__":
    test_text = "packages of ounce fresh tomatoes chopped and diced room temperature"
    result = filter_culinary_terms(test_text)
    print(f"Girdi: '{test_text}'")
    print(f"Çıktı: '{result}'")
