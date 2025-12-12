import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import datetime as dt

# --- 1. CONFIGURATION & TITLE ---
st.set_page_config(page_title="Customer Segmentation App", layout="wide")

st.title("🤖 Customer Segmentation Engine")
st.markdown("""
Aplikasi ini mengelompokkan pelanggan berdasarkan perilaku transaksi menggunakan algoritma **K-Means Clustering**.
Upload data transaksi terbaru Anda untuk mendapatkan insight strategi marketing.
""")

# --- 2. SIDEBAR & FILE UPLOAD ---
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload file CSV Transaksi", type=["csv"])

# Tombol untuk download template (opsional, agar user tahu formatnya)
# st.sidebar.download_button("Download Template CSV", ...) 

# --- 3. HELPER FUNCTIONS ---
@st.cache_resource
def load_model():
    # Pastikan file ini ada di folder yang sama dengan app.py
    try:
        model = joblib.load('kmeans_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("File model (kmeans_model.pkl) atau scaler (scaler.pkl) tidak ditemukan! Pastikan Anda sudah menjalankan langkah 'Simpan Model' di notebook.")
        return None, None

def calculate_rfm(df):
    # a. Pastikan format tanggal benar
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # b. Buat kolom TotalAmount (Monetary)
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    
    # c. Tentukan tanggal referensi (biasanya hari terakhir di data + 1 hari)
    latest_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    
    # d. Hitung RFM
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (latest_date - x.max()).days, # Recency
        'InvoiceNo': 'nunique', # Frequency
        'TotalAmount': 'sum'    # Monetary
    }).reset_index()
    
    rfm.rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalAmount': 'Monetary'
    }, inplace=True)
    
    return rfm

def preprocess_data(rfm_df, scaler):
    # a. Log Transformation (Sama persis seperti saat training)
    rfm_log = rfm_df[['Recency', 'Frequency', 'Monetary']].copy()
    rfm_log = np.log1p(rfm_log)
    
    # b. Scaling (Pakai scaler yang sudah dilatih)
    rfm_scaled = scaler.transform(rfm_log)
    
    return pd.DataFrame(rfm_scaled, columns=['Recency', 'Frequency', 'Monetary'])

def assign_label(cluster):
    # SESUAIKAN label ini dengan hasil analisis cluster Anda di Notebook!
    # Contoh di bawah ini hanya asumsi berdasarkan chat kita sebelumnya.
    if cluster == 1:
        return "VIP / Champions"
    elif cluster == 2:
        return "At Risk High Value"
    elif cluster == 3:
        return "New / Potential"
    elif cluster == 0:
        return "Lost / Low Value"
    return "Unknown"

# --- 4. MAIN LOGIC ---
model, scaler = load_model()

if uploaded_file is not None and model is not None:
    try:
        # Load Raw Data
        raw_data = pd.read_csv(uploaded_file)
        
        # Validasi Kolom Wajib
        required_cols = ['CustomerID', 'InvoiceDate', 'Quantity', 'UnitPrice', 'InvoiceNo']
        if not all(col in raw_data.columns for col in required_cols):
            st.error(f"Format CSV salah! Pastikan ada kolom: {', '.join(required_cols)}")
        else:
            with st.spinner('Sedang memproses data...'):
                # 1. Bersihkan Data (Hapus CustomerID kosong & Transaksi Negatif)
                clean_data = raw_data.dropna(subset=['CustomerID'])
                clean_data = clean_data[(clean_data['Quantity'] > 0) & (clean_data['UnitPrice'] > 0)]
                
                # 2. Hitung RFM
                rfm_data = calculate_rfm(clean_data)
                
                # 3. Preprocessing (Log + Scale)
                processed_data = preprocess_data(rfm_data, scaler)
                
                # 4. Prediksi
                clusters = model.predict(processed_data)
                rfm_data['Cluster'] = clusters
                rfm_data['Segment'] = rfm_data['Cluster'].apply(assign_label)
                
                # --- 5. DASHBOARD VISUALIZATION ---
                st.success("Segmentasi Selesai!")
                
                # Metrics Summary
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Pelanggan", f"{len(rfm_data)}")
                col2.metric("Rata-rata Belanja", f"${rfm_data['Monetary'].mean():,.0f}")
                col3.metric("Pelanggan VIP", f"{len(rfm_data[rfm_data['Cluster']==1])}") # Sesuaikan ID Cluster VIP
                col4.metric("Pelanggan Berisiko", f"{len(rfm_data[rfm_data['Cluster']==2])}") # Sesuaikan ID Cluster At Risk

                st.markdown("---")
                
                # Grafik Distribusi Segmen
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    st.subheader("Proporsi Segmen Pelanggan")
                    fig_pie = px.pie(rfm_data, names='Segment', title='Customer Segmentation Share')
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col_chart2:
                    st.subheader("Sebaran Pelanggan (3D)")
                    # Visualisasi 3D (Recency, Frequency, Monetary)
                    fig_3d = px.scatter_3d(rfm_data, x='Recency', y='Frequency', z='Monetary',
                                           color='Segment', opacity=0.7,
                                           title='RFM 3D Visualization')
                    st.plotly_chart(fig_3d, use_container_width=True)
                
                st.markdown("---")
                
                # --- 6. ACTIONABLE LISTS ---
                st.subheader("📋 Daftar Pelanggan per Segmen")
                
                # Filter Segmen
                selected_segment = st.selectbox("Pilih Segmen untuk Dilihat:", rfm_data['Segment'].unique())
                filtered_df = rfm_data[rfm_data['Segment'] == selected_segment]
                
                st.dataframe(filtered_df)
                
                # Download Button
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"Download Data {selected_segment}",
                    data=csv,
                    file_name=f'customer_segment_{selected_segment}.csv',
                    mime='text/csv',
                )

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        
else:
    st.info("Silakan upload file CSV transaksi di sidebar sebelah kiri.")
    st.write("Format CSV yang dibutuhkan (Header):")
    st.code("InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country")