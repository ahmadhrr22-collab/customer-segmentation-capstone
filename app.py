import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import datetime as dt
import requests
from streamlit_lottie import st_lottie

# --- 1. CONFIGURATION & TITLE ---
st.set_page_config(
    page_title="Clustify - Customer Segmentation Engine",
    page_icon="💎",
    layout="wide"
)

# --- FUNGSI LOAD ANIMASI LOTTIE ---
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# --- CSS STYLING (AGAR TAMPILAN CANTIK) ---
st.markdown("""
<style>
    /* Mengubah font judul */
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 3rem; 
        font-weight: 800; 
        color: #4B0082;
        margin-bottom: 0px;
    }
    
    /* Membuat Card untuk Metric agar ada bayangan */
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #CCCCCC;
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1); /* Bayangan */
        overflow-wrap: break-word;
    }

    /* Style untuk Box Strategi */
    .strategy-box {
        background-color: #f8f9fa; 
        padding: 20px; 
        border-radius: 15px; 
        border-left: 6px solid #4B0082;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Menghilangkan padding atas default Streamlit supaya lebih rapi */
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER SECTION DENGAN ANIMASI ---
col_header1, col_header2 = st.columns([1, 3])

with col_header1:
    # Animasi Robot/Data Analysis (Lottie)
    lottie_url = "https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json" 
    lottie_json = load_lottieurl(lottie_url)
    if lottie_json:
        st_lottie(lottie_json, height=150, key="coding")
    else:
        st.image("https://cdn-icons-png.flaticon.com/512/1904/1904425.png", width=100)

with col_header2:
    st.markdown('<p class="main-header">💎 Clustify</p>', unsafe_allow_html=True)
    st.markdown("**Automated Customer Segmentation & Marketing Strategy Engine**")
    st.markdown("Mengubah data transaksi mentah menjadi strategi pemasaran yang *actionable* menggunakan **K-Means Clustering**.")

st.markdown("---")

# --- 2. SIDEBAR & FILE UPLOAD ---
with st.sidebar:
    st.header("📂 Data Input")
    st.info("Upload data transaksi (CSV) untuk memulai analisis otomatis.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    st.markdown("---")
    st.markdown("### 📝 Panduan")
    st.markdown("""
    1. Siapkan data transaksi ritel.
    2. Pastikan format kolom sesuai standar UCI Dataset.
    3. Upload dan tunggu hasil analisis.
    """)
    st.caption("© 2025 Tim A25-CS332")

# --- 3. HELPER FUNCTIONS & INSIGHTS ---

# Database Insight (Sesuai Request Anda)
CLUSTER_INSIGHTS = {
    0: {
        "label": "Lost / Low Value",
        "description": "Pelanggan hampir hilang dan kontribusi sangat kecil.",
        "characteristics": ["Recency tinggi (>400 hari)", "Frequency rendah (1–2x)", "Monetary rendah"],
        "strategy": ["📉 Retargeting murah", "🏷️ Clearance Sale", "🚫 Efisiensi Budget"]
    },
    1: {
        "label": "VIP / Champions",
        "description": "Pelanggan paling loyal dan paling menguntungkan.",
        "characteristics": ["Recency rendah (Baru belanja)", "Frequency tinggi (Sering)", "Monetary Besar (Sultan)"],
        "strategy": ["👑 Program VIP & Loyalty", "🎁 Early Access Produk", "💎 Upselling Premium"]
    },
    2: {
        "label": "At Risk High Value",
        "description": "Pelanggan kaya yang sudah lama tidak kembali.",
        "characteristics": ["Recency tinggi (Menghilang)", "Frequency sedang", "Monetary Tinggi"],
        "strategy": ["🔔 'We Miss You' Campaign", "🎫 Diskon Agresif/Kupon", "❓ Survey Kepuasan"]
    },
    3: {
        "label": "New / Potential",
        "description": "Pelanggan baru dengan potensi tumbuh.",
        "characteristics": ["Recency sedang", "Frequency rendah", "Monetary kecil"],
        "strategy": ["👋 Welcome Series Email", "🛒 Cross-sell produk murah", "🎟️ Voucher Pembelian ke-2"]
    }
}

@st.cache_resource
def load_model():
    try:
        model = joblib.load('kmeans_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ File model tidak ditemukan!")
        return None, None

def calculate_rfm(df):
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    latest_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (latest_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalAmount': 'sum'
    }).reset_index()
    rfm.rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalAmount': 'Monetary'}, inplace=True)
    return rfm

def preprocess_data(rfm_df, scaler):
    rfm_log = rfm_df[['Recency', 'Frequency', 'Monetary']].copy()
    # Handle negative/zero values for log
    rfm_log = rfm_log.applymap(lambda x: x if x > 0 else 1) 
    rfm_log = np.log1p(rfm_log)
    rfm_scaled = scaler.transform(rfm_log)
    return pd.DataFrame(rfm_scaled, columns=['Recency', 'Frequency', 'Monetary'])

def get_label_from_id(cluster_id):
    return CLUSTER_INSIGHTS.get(cluster_id, {}).get("label", "Unknown")

# --- 4. MAIN LOGIC ---
model, scaler = load_model()

if uploaded_file is not None and model is not None:
    try:
        raw_data = pd.read_csv(uploaded_file)
        required_cols = ['CustomerID', 'InvoiceDate', 'Quantity', 'UnitPrice', 'InvoiceNo']
        
        if not all(col in raw_data.columns for col in required_cols):
            st.error(f"Format CSV salah! Pastikan ada kolom: {', '.join(required_cols)}")
        else:
            with st.spinner('🔄 Robot Clustify sedang menganalisis data...'):
                # Processing
                clean_data = raw_data.dropna(subset=['CustomerID'])
                clean_data = clean_data[(clean_data['Quantity'] > 0) & (clean_data['UnitPrice'] > 0)]
                rfm_data = calculate_rfm(clean_data)
                processed_data = preprocess_data(rfm_data, scaler)
                clusters = model.predict(processed_data)
                rfm_data['Cluster'] = clusters
                rfm_data['Segment'] = rfm_data['Cluster'].apply(get_label_from_id)
                
                # --- DASHBOARD UI ---
                st.success("✅ Analisis Selesai!")
                
                # Metrics Row (Dengan styling card CSS diatas)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total User", f"{len(rfm_data)}")
                m2.metric("Avg. Belanja", f"${rfm_data['Monetary'].mean():,.0f}")
                m3.metric("🏆 VIP User", f"{len(rfm_data[rfm_data['Cluster'] == 1])}")
                m4.metric("⚠️ At Risk", f"{len(rfm_data[rfm_data['Cluster'] == 2])}")

                st.markdown("---")
                
                # Charts Area
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("📊 Proporsi Segmen")
                    fig_pie = px.pie(rfm_data, names='Segment', 
                                     color='Segment',
                                     color_discrete_map={
                                         "VIP / Champions": "#2ecc71", # Hijau Keren
                                         "At Risk High Value": "#e74c3c", # Merah
                                         "New / Potential": "#3498db", # Biru
                                         "Lost / Low Value": "#95a5a6" # Abu
                                     },
                                     hole=0.5)
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with c2:
                    st.subheader("🧊 3D RFM Visualization")
                    fig_3d = px.scatter_3d(rfm_data, x='Recency', y='Frequency', z='Monetary',
                                           color='Segment', opacity=0.7)
                    st.plotly_chart(fig_3d, use_container_width=True)
                
                st.markdown("---")
                
                # Insight Section
                st.subheader("🚀 Rekomendasi Strategi")
                
                # Menggunakan Tabs agar lebih rapi daripada selectbox biasa
                tab1, tab2, tab3, tab4 = st.tabs(["🏆 VIP", "⚠️ At Risk", "🌱 New/Potential", "💤 Lost"])
                
                def display_strategy(cluster_id):
                    insight = CLUSTER_INSIGHTS[cluster_id]
                    # Menggunakan HTML Box custom
                    st.markdown(f"""
                    <div class="strategy-box">
                        <h3 style="color: #4B0082;">{insight['label']}</h3>
                        <p>{insight['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    sc1, sc2 = st.columns(2)
                    with sc1:
                        st.markdown("#### 🔍 Ciri-ciri")
                        for i in insight['characteristics']: st.info(i)
                    with sc2:
                        st.markdown("#### 💡 Action Plan")
                        for i in insight['strategy']: st.success(i)
                        
                    # Filter Data Button
                    st.markdown("---")
                    filtered_df = rfm_data[rfm_data['Cluster'] == cluster_id]
                    csv = filtered_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"📥 Download Data {insight['label']}",
                        data=csv,
                        file_name=f'Clustify_{insight["label"]}.csv',
                        mime='text/csv'
                    )

                with tab1: display_strategy(1)
                with tab2: display_strategy(2)
                with tab3: display_strategy(3)
                with tab4: display_strategy(0)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    # Tampilan Awal (Kosong) dengan Animasi
    col_empty1, col_empty2 = st.columns(2)
    with col_empty1:
         st.info("👋 Silakan upload file CSV di sebelah kiri.")
         st.code("Format: InvoiceNo, Quantity, InvoiceDate, UnitPrice, CustomerID")
    with col_empty2:
         # Animasi Upload (Lottie)
         lottie_upload = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_vktpl5cy.json")
         if lottie_upload: st_lottie(lottie_upload, height=200)
