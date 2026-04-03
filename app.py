import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Ρυθμίσεις Σελίδας ---
st.set_page_config(page_title="Kastoria Tourism Expert Analytics", layout="wide")

# --- 2. Φόρτωση Δεδομένων (Σωστή Σειρά) ---
@st.cache_data
def load_data():
    # Φόρτωση αρχείων
    rev = pd.read_csv("TripAdvisor_Kastoria.csv", sep=";")
    places = pd.read_csv("ThingsToDo.csv", sep=";")
    
    # Α) Καθαρισμός Ημερομηνιών
    rev['publishedDate'] = pd.to_datetime(rev['publishedDate'], dayfirst=True, errors='coerce')
    rev = rev.dropna(subset=['publishedDate']) # Αφαιρούμε όσα δεν έχουν ημερομηνία
    rev['year'] = rev['publishedDate'].dt.year
    rev['month'] = rev['publishedDate'].dt.month
    
    # Β) Καθαρισμός Αριθμητικών Δεδομένων & Κενών
    rev['rating'] = pd.to_numeric(rev['rating'], errors='coerce').fillna(0)
    rev['Photocount'] = pd.to_numeric(rev['Photocount'], errors='coerce').fillna(0)
    rev['placeInfo/name'] = rev['placeInfo/name'].fillna("Άγνωστο Αξιοθέατο")
    rev['user/contributions/totalContributions'] = pd.to_numeric(rev['user/contributions/totalContributions'], errors='coerce').fillna(0)
    rev['review_len'] = rev['text'].str.len().fillna(0)
    
    # Γ) Φόρτωση Stopwords
    try:
        with open("greek_stopwords.txt", "r", encoding="utf-8") as f:
            stopwords = set(f.read().split(","))
    except:
        stopwords = None
        
    return rev, places, stopwords

# Εκτέλεση φόρτωσης
try:
    rev, places, gr_stopwords = load_data()
except Exception as e:
    st.error(f"Σφάλμα κατά τη φόρτωση των δεδομένων: {e}")
    st.stop()

# --- 3. Sidebar & Φίλτρα ---
st.sidebar.header("🎯 Στρατηγική Επιλογή")
# Διόρθωση για το TypeError στο sorted (αφαίρεση NaN)
unique_names = rev['placeInfo/name'].unique()
all_places = ["Όλα τα Αξιοθέατα"] + sorted([str(name) for name in unique_names])
selected_place = st.sidebar.selectbox("Επιλέξτε Σημείο Ενδιαφέροντος:", all_places)

# Δημιουργία του df βάσει φίλτρου
if selected_place == "Όλα τα Αξιοθέατα":
    df = rev
else:
    df = rev[rev['placeInfo/name'] == selected_place]

# --- 4. Dashboard KPIs ---
st.title(f"🏛️ Dashboard Αναλύσεων: {selected_place}")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Συνολικές Κριτικές", f"{len(df):,}")
col2.metric("Μέση Βαθμολογία", f"{df['rating'].mean():.2f} ⭐")
resp_rate = (df['ownerResponse/text'].notna().sum() / len(df)) * 100 if len(df) > 0 else 0
col3.metric("Admin Response Rate", f"{resp_rate:.1f}%")
col4.metric("Helpful Votes", f"{int(df['user/contributions/helpfulVotes'].sum()):,}")

# --- 5. Heatmaps & Χρονική Ανάλυση ---
st.subheader("📅 Χρονική Εξέλιξη & Εποχικότητα")
c1, c2 = st.columns(2)
with c1:
    monthly_counts = df.groupby(['year', 'month']).size().reset_index(name='Κριτικές')
    fig_line = px.line(monthly_counts, x='month', y='Κριτικές', color='year', markers=True, title="Επισκέψεις ανά Μήνα")
    st.plotly_chart(fig_line, use_container_width=True)
with c2:
    pivot_heat = df.pivot_table(index='year', columns='month', values='rating', aggfunc='count').fillna(0)
    fig_heat = px.imshow(pivot_heat, text_auto=True, title="Heatmap Πυκνότητας Επισκέψεων")
    st.plotly_chart(fig_heat, use_container_width=True)

# --- 6. Γλώσσες & Προέλευση ---
st.subheader("🌍 Γεωγραφική & Γλωσσική Κατανομή")
c3, c4 = st.columns(2)
with c3:
    lang_dist = df['lang'].value_counts().reset_index()
    lang_dist.columns = ['Language', 'Count']
    fig_lang = px.pie(lang_dist, values='Count', names='Language', title="Κατανομή Γλωσσών", hole=0.3)
    st.plotly_chart(fig_lang, use_container_width=True)
with c4:
    top_cities = df['user/userLocation/name'].value_counts().head(20).reset_index()
    top_cities.columns = ['City', 'Count']
    fig_cities = px.bar(top_cities, x='Count', y='City', orientation='h', title="Top 20 Πόλεις Προέλευσης")
    st.plotly_chart(fig_cities, use_container_width=True)

# --- 7. Profile & Engagement (Pie Chart Fix) ---
st.subheader("👥 Προφίλ Επισκέπτη & Εμπλοκή")
c5, c6 = st.columns(2)
with c5:
    bins = [0, 1, 5, 20, 100, 1000000]
    labels = ["Newbie (1)", "Explorer (2-5)", "Contributor (6-20)", "Expert (21-100)", "Local Guide (100+)"]
    df['engagement_level'] = pd.cut(df['user/contributions/totalContributions'], bins=bins, labels=labels)
    eng_dist = df['engagement_level'].value_counts().reindex(labels).reset_index()
    eng_dist.columns = ['Level', 'Counts'] 
    fig_eng = px.pie(eng_dist, values='Counts', names='Level', title="Επίπεδο Εμπειρίας Χρηστών", hole=0.4)
    st.plotly_chart(fig_eng, use_container_width=True)
with c6:
    text_content = " ".join(df['text'].dropna().astype(str))
    if text_content:
        wc = WordCloud(width=800, height=400, background_color="white", stopwords=gr_stopwords).generate(text_content)
        fig_wc, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig_wc)

# --- 8. Regression Analysis (Statsmodels Fix) ---
st.subheader("📱 Ψηφιακό Αποτύπωμα")
try:
    fig_ols = px.scatter(df, x='review_len', y='rating', trendline="ols", 
                         title="Μέγεθος Κειμένου vs Βαθμολογία", opacity=0.4)
    st.plotly_chart(fig_ols, use_container_width=True)
except:
    st.write("Η ανάλυση τάσης απαιτεί περισσότερα δεδομένα ή τη βιβλιοθήκη statsmodels.")

st.caption("Kastoria Tourism Analytics Dashboard | Expert Edition")
