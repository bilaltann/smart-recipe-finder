# Buzdolabı Asistanı: "Elimdekilerle Ne Pişirebilirim?"
## Proje Hakkında
Buzdolabı Asistanı, insanların buzdolabındaki mevcut malzemeleri israf etmeden değerlendirmelerini sağlamak amacıyla tasarlanmış doğal dil işleme (NLP) ve derin öğrenme tabanlı bir tavsiye sistemidir.
Özellikle öğrenciler, yalnız yaşayanlar ve yeni evlenenler için pratik bir mutfak çözümü sunmayı hedefleyen bu sistem; kullanıcının girdiği "tavuk, domates, soğan" gibi doğal dildeki ham malzeme listesini alır, işler ve bu malzemelerle yapılabilecek en uygun yemek tariflerini önerir.
## Veri Seti
Projede, tarif içerikleri ve malzeme listeleri için Kaggle'da açık kaynak olarak sunulan *Epirecipes (Epicurious Recipes)* veri seti kullanılmaktadır. Projenin NLP gereksinimleri doğrultusunda, tariflerin ham metinlerini hiyerarşik bir yapıda barındıran full_format_recipes.json dosyası üzerinden çalışılmaktadır.
 * *Veri Seti Linki:* https://www.kaggle.com/datasets/hugodarwood/epirecipes?select=full_format_recipes.json (Projede yaklaşık 20.000 tarif içeren veri kullanılmaktadır.)
## Teknik Kapsam ve Metodoloji
Projenin altyapısı, verinin işlenmesi ve anlamsal modelin kurulması olmak üzere iki temel mühendislik mimarisinden oluşmaktadır:
### 1. Doğal Dil İşleme (NLP) ve Veri Ön İşleme
Kullanıcının girdiği kelimeler ile veri setindeki karmaşık malzeme metinlerini ortak bir dilde buluşturmak için şu işlemler uygulanır:
 * *Veri Temizleme:* Tarif metinlerindeki ölçü birimlerinin (gram, cup, tablespoon), sayıların, kesirlerin ve noktalama işaretlerinin Regex kullanılarak temizlenmesi.
 * *Mutfak Terimleri Filtrelemesi:* Malzemenin kendisi olmayan, sadece durum bildiren "chopped", "fresh", "peeled" gibi kelimelerin alana özel olarak oluşturulan sözlüklerle metinden çıkarılması.
 * *Kök Bulma (Lemmatization):* NLTK/spaCy gibi kütüphaneler kullanılarak kelimelerin çoğul eklerinden arındırılması ve asıl köklerine (örn: "tomatoes" -> "tomato") dönüştürülmesi.
### 2. Derin Öğrenme ve Model Geliştirme
NLP aşamasından çıkan temizlenmiş anahtar kelimeler, tavsiye sisteminin temelini oluşturan anlamsal bir arama motoruna dönüştürülür:
 * *Vektörizasyon (Embedding):* BERT (veya Sentence-BERT) gibi önceden eğitilmiş bir dil modeli kullanılarak, 20.000 tarifin malzeme listesi çok boyutlu matematiksel vektörlere dönüştürülür.
 * *Anlamsal Eşleştirme:* Kullanıcının arayüze girdiği malzemeler aynı modelden geçirilerek vektör haline getirilir ve Kosinüs Benzerliği (Cosine Similarity) metriği ile kullanıcının elindeki malzemelere en yakın (en yüksek eşleşme oranına sahip) tarifler tespit edilir.
## Kullanılan Teknolojiler
 * *Dil:* Python
 * *Veri İşleme ve Analiz:* Pandas, Regex
 * *Doğal Dil İşleme (NLP):* NLTK
 * *Derin Öğrenme ve Vektörizasyon:* PyTorch, HuggingFace (Transformers)
 * *Kullanıcı Arayüzü:* Streamlit

## Projenin Çalıştırılması

Uygulamayı yerel bilgisayarınızda başlatmak için şu adımları izleyin:

1. **Gerekli Kütüphaneleri Yükleyin:**
   ```bash
   pip install pandas numpy scikit-learn sentence-transformers nltk streamlit
   ```

2. **Streamlit Web Arayüzünü Başlatın:**
   Bilgisayarınızdaki Python 3.12 ortamını hedef alarak ve Hugging Face bağlantı hatalarını engellemek için ayna sunucu ayarını içeren aşağıdaki komutları kullanabilirsiniz:

   * **Eğer `smart-recipe-finder` (ana proje dizini) klasöründeyseniz:**
     ```powershell
     $env:HF_ENDPOINT="https://hf-mirror.com"; C:\Users\ozbrk\AppData\Local\Programs\Python\Python312\python.exe -m streamlit run src/app.py
     ```

   * **Eğer terminalde zaten `src` klasörünün içindeyseniz:**
     ```powershell
     $env:HF_ENDPOINT="https://hf-mirror.com"; C:\Users\ozbrk\AppData\Local\Programs\Python\Python312\python.exe -m streamlit run app.py
     ```

3. **Uygulamaya Erişin:**
   Sunucu başarıyla ayağa kalktığında tarayıcınızda otomatik olarak açılacaktır. Açılmazsa manuel olarak şu adrese gidebilirsiniz:
   👉 **http://localhost:8501**
