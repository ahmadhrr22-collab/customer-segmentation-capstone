import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import datetime as dt

# --- 1. CONFIGURATION & TITLE ---
st.set_page_config(
    page_title="Clustify - Customer Segmentation Engine",
    page_icon="💎",
    layout="wide"
)

# Custom CSS untuk mempercantik tampilan
st.markdown("""
<style>
    .main-header {font-size: 3rem; font-weight: bold; color: #4B0082;}
    .sub-header {font-size: 1.5rem; font-weight: bold; color: #333;}
    .strategy-box {background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #4B0082;}
</style>
""", unsafe_allow_html=True)

st.title("💎 Clustify")
st.markdown("**Automated Customer Segmentation & Marketing Strategy Engine**")
st.markdown("""
Website ini membantu Anda mengidentifikasi segmen pelanggan potensial menggunakan algoritma **K-Means Clustering** dan memberikan rekomendasi strategi pemasaran yang dipersonalisasi.
""")

# --- 2. SIDEBAR & FILE UPLOAD ---
with st.sidebar:
    st.header("📂 Upload Data")
    st.info("Pastikan file CSV memiliki kolom: InvoiceNo, Quantity, InvoiceDate, UnitPrice, CustomerID")
    uploaded_file = st.file_uploader("Upload file CSV Transaksi", type=["csv"])
    st.markdown("---")
    st.write("Created by Tim A25-CS332")

# --- 3. HELPER FUNCTIONS & INSIGHTS DICTIONARY ---

# Database Insight & Strategi (Sesuai Request Anda)
CLUSTER_INSIGHTS = {
    0: {
        "label": "Lost / Low Value",
        "description": "Pelanggan hampir hilang dan kontribusi sangat kecil.",
        "characteristics": [
            "Recency tinggi (>400 hari, lama tidak belanja)",
            "Frequency rendah (1–2 kali)",
            "Monetary rendah (~Rp584 ribu)"
        ],
        "strategy": [
            "📉 Retargeting via WhatsApp/Email (Hemat Biaya)",
            "🏷️ Tawarkan produk 'Clearance Sale' atau harga rendah",
            "🚫 Jangan habiskan budget iklan besar di sini"
        ]
    },
    1: {
        "label": "VIP / Champions",
        "description": "Pelanggan paling loyal dan paling menguntungkan.",
        "characteristics": [
            "Recency rendah (~10 hari, baru saja belanja)",
            "Frequency tertinggi (~19 kali transaksi)",
            "Monetary fantastis (~Rp10.8 juta)"
        ],
        "strategy": [
            "👑 Program VIP Exclusive & Loyalty Points",
            "🎁 Early Access untuk produk baru",
            "💎 Upsell produk premium / Bundling eksklusif"
        ]
    },
    2: {
        "label": "At Risk High Value",
        "description": "Pelanggan mulai tidak aktif, namun punya potensi daya beli tinggi.",
        "characteristics": [
            "Recency cukup tinggi (~118 hari tidak belanja)",
            "Frequency menengah (~7–8 kali)",
            "Monetary menengah-tinggi (~Rp3.5 juta)"
        ],
        "strategy": [
            "🔔 Personal Reminder: 'We Miss You'",
            "🎫 Penawaran Diskon Targeted / Cashback agresif",
            "❓ Survey kepuasan: Tanyakan kenapa berhenti belanja"
        ]
    },
    3: {
        "label": "New / Potential",
        "description": "Pelanggan baru yang masih membangun hubungan dengan brand.",
        "characteristics": [
            "Recency sedang (~36 hari)",
            "Frequency rendah (~2 kali)",
            "Monetary kecil (~Rp776 ribu)"
        ],
        "strategy": [
            "👋 Welcome Journey & Edukasi Produk",
            "🛒 Cross-sell produk pelengkap yang murah",
            "🎟️ Voucher diskon untuk pembelian kedua"
        ]
    }
}

@st.cache_resource
def load_model():
    try:
        model = joblib.load('kmeans_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ File model (kmeans_model.pkl) atau scaler (scaler.pkl) tidak ditemukan!")
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
            with st.spinner('🔄 Clustify sedang menganalisis data Anda...'):
                # 1. Cleaning & RFM
                clean_data = raw_data.dropna(subset=['CustomerID'])
                clean_data = clean_data[(clean_data['Quantity'] > 0) & (clean_data['UnitPrice'] > 0)]
                rfm_data = calculate_rfm(clean_data)
                
                # 2. Prediction
                processed_data = preprocess_data(rfm_data, scaler)
                clusters = model.predict(processed_data)
                rfm_data['Cluster'] = clusters
                rfm_data['Segment'] = rfm_data['Cluster'].apply(get_label_from_id)
                
                # --- 5. DASHBOARD VISUALIZATION ---
                st.success("✅ Analisis Selesai! Berikut adalah hasil segmentasi pelanggan Anda.")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Pelanggan", f"{len(rfm_data)}")
                col2.metric("Rata-rata Transaksi", f"${rfm_data['Monetary'].mean():,.0f}")
                
                # Hitung jumlah user per cluster untuk metric
                vip_count = len(rfm_data[rfm_data['Cluster'] == 1])
                risk_count = len(rfm_data[rfm_data['Cluster'] == 2])
                
                col3.metric("🏆 Pelanggan VIP", f"{vip_count}", delta="High Value")
                col4.metric("⚠️ Pelanggan Berisiko", f"{risk_count}", delta_color="inverse")

                st.markdown("---")
                
                # Charts
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("📊 Proporsi Segmen")
                    fig_pie = px.pie(rfm_data, names='Segment', 
                                     color='Segment',
                                     color_discrete_map={
                                         "VIP / Champions": "green",
                                         "At Risk High Value": "orange",
                                         "New / Potential": "blue",
                                         "Lost / Low Value": "grey"
                                     },
                                     hole=0.4)
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with c2:
                    st.subheader("🧊 Sebaran 3D (RFM)")
                    fig_3d = px.scatter_3d(rfm_data, x='Recency', y='Frequency', z='Monetary',
                                           color='Segment', opacity=0.7)
                    fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0))
                    st.plotly_chart(fig_3d, use_container_width=True)
                
                st.markdown("---")
                
                # --- 6. INTELLIGENT ACTIONABLE INSIGHTS ---
                st.header("🚀 Actionable Insights & Strategy")
                
                # Pilihan Segmen
                segment_options = rfm_data['Segment'].unique()
                selected_segment_name = st.selectbox("🎯 Pilih Segmen untuk melihat Strategi Detail:", segment_options)
                
                # Cari ID Cluster berdasarkan nama segmen yang dipilih
                selected_cluster_id = [k for k, v in CLUSTER_INSIGHTS.items() if v['label'] == selected_segment_name][0]
                insight = CLUSTER_INSIGHTS[selected_cluster_id]
                
                # Tampilkan Insight dalam Box Rapi
                with st.container():
                    st.markdown(f"<div class='strategy-box'><h3>Analisis: {selected_segment_name}</h3><p>{insight['description']}</p></div>", unsafe_allow_html=True)
                    
                    col_i1, col_i2 = st.columns(2)
                    with col_i1:
                        st.subheader("🔍 Ciri-ciri Utama")
                        for char in insight['characteristics']:
                            st.write(f"- {char}")
                            
                    with col_i2:
                        st.subheader("💡 Rekomendasi Strategi")
                        for strat in insight['strategy']:
                            st.write(f"{strat}")

                st.markdown("---")
                
                # --- 7. DATA DOWNLOAD ---
                st.subheader(f"📥 Download Data: {selected_segment_name}")
                filtered_df = rfm_data[rfm_data['Cluster'] == selected_cluster_id]
                st.dataframe(filtered_df.head())
                
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"Download CSV ({len(filtered_df)} baris)",
                    data=csv,
                    file_name=f'Clustify_Segment_{selected_segment_name}.csv',
                    mime='text/csv',
                )

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        
else:
    # Tampilan Awal jika belum upload
    st.info("👋 Selamat datang di Clustify! Silakan upload data transaksi Anda di sidebar kiri untuk memulai.")
    st.markdown("### Format Data yang Dibutuhkan:")
    st.code("InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country")

