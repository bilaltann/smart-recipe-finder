import streamlit as st
import os
import sys

# Ensure src is in the system path to allow proper imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app_inference import RecipeInferencePipeline

# Page configuration
st.set_page_config(
    page_title="Akıllı Tarif Bulucu - Smart Recipe Finder",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF4B2B 0%, #FF416C 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .recipe-card {
        background-color: #ffffff;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.03), 0 1px 3px rgba(0,0,0,0.06);
        border-left: 5px solid #FF416C;
        margin-bottom: 0.8rem;
        margin-top: 1rem;
    }
    
    .badge {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-block;
        margin-left: 0.8rem;
    }
    
    .ingredient-tag {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.2rem;
    }
    
    .tag-matched {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .tag-missing {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    
    /* Dark mode adjustments */
    @media (prefers-color-scheme: dark) {
        .recipe-card {
            background-color: #1e1e24;
            color: #f0f0f0;
            border-left: 5px solid #FF416C;
        }
        .subtitle {
            color: #ccc;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Load pipeline once and cache it across reruns
@st.cache_resource
def load_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    embeddings_file = os.path.join(base_dir, "data", "recipe_embeddings.pkl")
    recipes_file = os.path.join(base_dir, "data", "full_format_recipes.json")
    
    # Enable mirror just in case
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    return RecipeInferencePipeline(embeddings_file, recipes_file)

try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Sistem yüklenirken hata oluştu. Lütfen veri dosyalarını kontrol edin. Detay: {e}")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("### 🍳 Tarif Ayarları")
    top_n = st.slider("Önerilecek Tarif Sayısı", min_value=1, max_value=10, value=5)
    st.markdown("---")
    st.markdown(
        "**Akıllı Tarif Bulucu**, elinizdeki malzemeleri anlamlandırmak için **NLP** ve **Derin Öğrenme** yöntemlerini kullanır."
    )
    st.markdown("- **Model:** all-MiniLM-L6-v2 (Sentence-Transformers)")
    st.markdown("- **Algoritma:** Cosine Similarity")
    st.markdown("- **Preprocess:** NLTK Stopwords & Lemmatizer")

# Main Page Layout
st.markdown("<h1 class='main-title'>🍳 Akıllı Tarif Bulucu</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Buzdolabınızda kalan malzemeleri girin, en uyumlu tarifleri hemen listeleyelim!</p>", unsafe_allow_html=True)

# User input field
user_input = st.text_input(
    "Elinizdeki malzemeleri İngilizce olarak aralarında virgülle yazın:",
    placeholder="Örn: chicken, tomatoes, olive oil, garlic...",
    help="Ölçü birimi yazmanıza gerek yoktur, sistem otomatik temizler."
)

if st.button("En Uygun Tarifleri Bul", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("Lütfen arama yapmak için en az bir malzeme girin.")
    else:
        with st.spinner("En leziz tarifler semantik olarak aranıyor... 👩‍🍳"):
            result = pipeline.run_pipeline(user_input, top_n=top_n)
            
        if "error" in result:
            st.error(result["error"])
        else:
            st.success(f"Arama Başarılı! Temizlenen malzemeleriniz: **{result['clean_ingredients']}**")
            st.markdown("---")
            
            for idx, rec in enumerate(result["results"], 1):
                # Custom Header Card
                st.markdown(f"""
                <div class="recipe-card">
                    <h3 style="margin: 0; padding: 0;">{idx}. {rec['title']} <span class="badge">Eşleşme Oranı: %{rec['match_percentage']}</span></h3>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns([2, 3])
                
                with col1:
                    st.markdown("##### 🔍 Malzeme Durumu")
                    
                    # Custom Matched / Missing HTML Tags
                    matched_tags = "".join([f"<span class='ingredient-tag tag-matched'>✅ {w}</span>" for w in rec.get("matched_ingredients", [])])
                    missing_tags = "".join([f"<span class='ingredient-tag tag-missing'>❌ {w}</span>" for w in rec.get("missing_ingredients", [])])
                    
                    if matched_tags:
                        st.markdown(f"**Eşleşenler:** {matched_tags}", unsafe_allow_html=True)
                    if missing_tags:
                        st.markdown(f"**Sizde Olmayanlar:** {missing_tags}", unsafe_allow_html=True)
                        
                    st.markdown("##### 📋 Tarifteki Malzemeler")
                    st.write(", ".join(rec["ingredients"]))
                    
                with col2:
                    st.markdown("##### 📖 Hazırlanışı / Yapılış Talimatları")
                    st.info(rec["instructions"])
                
                st.markdown("---")
