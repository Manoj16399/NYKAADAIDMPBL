import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from lifetimes import BetaGeoFitter, GammaGammaFitter
from datetime import datetime

# ---------- PAGE CONFIG ---------
st.set_page_config(page_title="Nykaa Customer Insight Dashboard", layout="wide")
st.title("💄 Nykaa | Customer Intelligence Dashboard")
st.markdown("""
This dashboard offers strategic insights into customer behavior, segments, churn, and lifetime value.
Created for the Marketing and Sales Directors to guide targeted actions.
""")

# ---------- LOAD DATA ---------
@st.cache_data
def load_data():
    df = pd.read_csv("nykaa_synthetic_customer_data_v2.csv")

    # Clean and normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Rename standard columns
    df.rename(columns={
        'invoice_date': 'InvoiceDate',
        'customer_id': 'CustomerID',
        'total_amount': 'TotalAmount'
    }, inplace=True)

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df

df = load_data()

# ---------- SIDEBAR FILTERS ---------
st.sidebar.header("🔍 Filters")
region = st.sidebar.selectbox("Select Region", options=["All"] + sorted(df['region'].unique().tolist()))
date_range = st.sidebar.date_input("Select Invoice Date Range", [df['InvoiceDate'].min(), df['InvoiceDate'].max()])

# Apply filters
if region != "All":
    df = df[df['region'] == region]
df = df[(df['InvoiceDate'] >= pd.to_datetime(date_range[0])) & (df['InvoiceDate'] <= pd.to_datetime(date_range[1]))]

# ---------- TABS ----------
tabs = st.tabs(["Overview", "RFM Segmentation", "CLTV Prediction", "Churn Analysis"])

# ---------- OVERVIEW TAB ----------
with tabs[0]:
    st.subheader("1. 🧾 Sales Overview")
    st.markdown("Understand monthly trends and regional performance.")

    monthly_sales = df.groupby(df['InvoiceDate'].dt.to_period("M")).agg({'TotalAmount': 'sum'}).reset_index()
    monthly_sales['InvoiceDate'] = monthly_sales['InvoiceDate'].astype(str)
    fig = px.line(monthly_sales, x='InvoiceDate', y='TotalAmount', title='Monthly Sales Trend')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("2. 🌍 Sales by Region")
    st.markdown("See which regions generate the most revenue.")
    region_sales = df.groupby('region')['TotalAmount'].sum().sort_values(ascending=False)
    st.bar_chart(region_sales)

    st.subheader("3. 👥 Customer Count")
    st.markdown("How many unique buyers are we serving?")
    st.metric("Unique Customers", value=df['CustomerID'].nunique())

# ---------- RFM SEGMENTATION TAB ----------
with tabs[1]:
    st.subheader("4. 🎯 RFM Segmentation")
    st.markdown("Segment customers by Recency, Frequency and Monetary value.")

    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'invoiceno': 'nunique',
        'TotalAmount': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm_scaled = rfm[['Recency', 'Frequency', 'Monetary']].copy()
    rfm_scaled = (rfm_scaled - rfm_scaled.min()) / (rfm_scaled.max() - rfm_scaled.min())
    rfm['Segment'] = kmeans.fit_predict(rfm_scaled)

    st.markdown("**Customer Segments based on RFM Clustering**")
    fig = px.scatter_3d(rfm, x='Recency', y='Frequency', z='Monetary', color=rfm['Segment'].astype(str))
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(rfm.groupby('Segment').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'CustomerID': 'count'
    }).rename(columns={'CustomerID': 'Customer Count'}))

# ---------- CLTV TAB ----------
with tabs[2]:
    st.subheader("5. 💰 Customer Lifetime Value Prediction")
    st.markdown("We use BG/NBD & Gamma-Gamma models to estimate CLTV.")

    cltv_data = df.groupby('CustomerID').agg({
        'InvoiceDate': [np.min, np.max],
        'invoiceno': 'count',
        'TotalAmount': 'sum'
    })
    cltv_data.columns = ['min_date', 'max_date', 'frequency', 'monetary']
    cltv_data['T'] = (snapshot_date - cltv_data['min_date']).dt.days / 7
    cltv_data['recency'] = (cltv_data['max_date'] - cltv_data['min_date']).dt.days / 7
    cltv_data['monetary'] = cltv_data['monetary'] / cltv_data['frequency']
    cltv_data = cltv_data[cltv_data['frequency'] > 1]

    bgf = BetaGeoFitter()
    bgf.fit(cltv_data['frequency'], cltv_data['recency'], cltv_data['T'])
    cltv_data['pred_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(12, cltv_data['frequency'], cltv_data['recency'], cltv_data['T'])

    ggf = GammaGammaFitter()
    ggf.fit(cltv_data['frequency'], cltv_data['monetary'])
    cltv_data['pred_monetary'] = ggf.conditional_expected_average_profit(cltv_data['frequency'], cltv_data['monetary'])

    cltv_data['CLTV'] = cltv_data['pred_purchases'] * cltv_data['pred_monetary']

    top_cltv = cltv_data.sort_values(by='CLTV', ascending=False).head(20)
    fig = px.bar(top_cltv, x=top_cltv.index, y='CLTV', title="Top 20 Customers by Predicted CLTV")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(top_cltv[['frequency', 'monetary', 'CLTV']])

# ---------- CHURN TAB ----------
with tabs[3]:
    st.subheader("6. 📉 First-Time Buyer Churn Analysis")
    st.markdown("Identifying drop-off post first purchase.")

    txn_count = df.groupby('CustomerID')['invoiceno'].nunique()
    first_time_buyers = txn_count[txn_count == 1].index
    first_time_df = df[df['CustomerID'].isin(first_time_buyers)]

    churn_by_region = first_time_df.groupby('region')['CustomerID'].nunique().sort_values(ascending=False)
    fig = px.bar(churn_by_region, title="First-Time Buyers by Region")
    st.plotly_chart(fig, use_container_width=True)

    st.metric("% of First-Time Churners", f"{len(first_time_buyers) / df['CustomerID'].nunique():.2%}")
    st.markdown("You can target these users using re-engagement campaigns and offers.")
