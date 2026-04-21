import json
import re
from collections import Counter

def extract_terms(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    measurements = Counter()
    prep_terms = Counter()
    
    for recipe in data:
        if 'ingredients' not in recipe or not recipe['ingredients']:
            continue
            
        for ing in recipe['ingredients']:
            if not isinstance(ing, str):
                continue
                
            ing = ing.lower()
            
            # Ölçü birimleri: Sayılardan veya kesirlerden hemen sonra gelen kelimeler
            # Örn: 1 cup, 1/2 teaspoon, 2 (10 ounce) -> "ounce" parantez içindeyse farklı ama
            # basitçe sayılardan sonra gelen ilk harfli kelimeyi alalım.
            measure_matches = re.findall(r'\d+(?:/\d+)?\s+(?:(?:\([^\)]+\)\s+)?)([a-z]+)', ing)
            for m in measure_matches:
                measurements[m] += 1
                
            # Hazırlık terimleri: Virgülle ayrılmış kısımlar (genelde sonda olur)
            # "fresh tomatoes, chopped", "carrot, peeled and chopped"
            if ',' in ing:
                parts = ing.split(',')
                for part in parts[1:]:
                    words = re.findall(r'\b[a-z]+\b', part)
                    for w in words:
                        if w not in ['and', 'or', 'to', 'taste']:
                            prep_terms[w] += 1
            
            # Ayrıca 'ed' ile biten yaygın kelimeler
            ed_words = re.findall(r'\b[a-z]+ed\b', ing)
            for w in ed_words:
                if w not in ['red', 'seed', 'weed']: # ignore some common non-prep ones
                    prep_terms[w] += 1

    print("Top 20 Measurements:")
    for w, c in measurements.most_common(20):
        print(f"'{w}': {c}")
        
    print("\nTop 40 Prep Terms:")
    for w, c in prep_terms.most_common(40):
        print(f"'{w}': {c}")

if __name__ == '__main__':
    extract_terms('data/full_format_recipes.json')
