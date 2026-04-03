import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

df['publishedDate'] = pd.to_datetime(df['publishedDate'], errors='coerce')
df = df.dropna(subset=['publishedDate']) # Αφαιρεί κριτικές χωρίς ημερομηνία

# --- Ρυθμίσεις Σελίδας ---
st.set_page_config(page_title="Kastoria Tourism Expert Analytics", layout="wide")

# --- Custom CSS για KPIs ---
st.markdown("""
<style>
    .kpi-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2d4a6b;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- Φόρτωση Δεδομένων ---
@st.cache_data
def load_data():
    # Φόρτωση με σωστό delimiter
    rev = pd.read_csv("TripAdvisor_Kastoria.csv", sep=";")
    places = pd.read_csv("ThingsToDo.csv", sep=";")
    
    # Καθαρισμός Ημερομηνιών (Υποστήριξη για DD/MM/YYYY)
    rev['publishedDate'] = pd.to_datetime(rev['publishedDate'], dayfirst=True, errors='coerce')
    rev['year'] = rev['publishedDate'].dt.year
    rev['month'] = rev['publishedDate'].dt.month
    
    # Υπολογισμός μεγέθους κριτικής
    rev['review_len'] = rev['text'].str.len().fillna(0)
    
    # Φόρτωση Stopwords
    try:
        with open("greek_stopwords.txt", "r", encoding="utf-8") as f:
            stopwords = set(f.read().split(","))
    except:
        stopwords = None
        
    return rev, places, stopwords

try:
    rev, places, gr_stopwords = load_data()
except Exception as e:
    st.error(f"Σφάλμα φόρτωσης: {e}")
    st.stop()
    
# Καθαρισμός βασικών στηλών από NaN που προκαλούν σφάλματα στα γραφήματα
rev['rating'] = pd.to_numeric(rev['rating'], errors='coerce').fillna(0)
rev['Photocount'] = pd.to_numeric(rev['Photocount'], errors='coerce').fillna(0)
rev['placeInfo/name'] = rev['placeInfo/name'].fillna("Άγνωστο Αξιοθέατο")

# --- Sidebar Φίλτρα ---
st.sidebar.header("🎯 Στρατηγική Επιλογή")
# Αφαιρούμε τα NaN (dropna) και μετατρέπουμε σε string για σιγουριά
unique_names = rev['placeInfo/name'].dropna().unique()
all_places = ["Όλα τα Αξιοθέατα"] + sorted([str(name) for name in unique_names])

selected_place = st.sidebar.selectbox("Επιλέξτε Σημείο Ενδιαφέροντος:", all_places)

# Φιλτράρισμα Δεδομένων
if selected_place == "Όλα τα Αξιοθέατα":
    df = rev
else:
    df = rev[rev['placeInfo/name'] == selected_place]

# --- ΤΙΤΛΟΣ & KPIs ---
st.title(f"🏛️ Dashboard Αναλύσεων: {selected_place}")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Συνολικές Κριτικές", f"{len(df):,}")
with col2:
    st.metric("Μέση Βαθμολογία", f"{df['rating'].mean():.2f} ⭐")
with col3:
    resp_rate = (df['ownerResponse/text'].notna().sum() / len(df)) * 100
    st.metric("Admin Response Rate", f"{resp_rate:.1f}%")
with col4:
    st.metric("Helpful Votes", f"{int(df['user/contributions/helpfulVotes'].sum()):,}")

# --- SECTION 1: Χρονική Ανάλυση & Εποχικότητα ---
st.header("📅 Χρονική Εξέλιξη & Εποχικότητα")
c1, c2 = st.columns(2)

with c1:
    # Επισκέψεις ανά έτος/μήνα
    monthly_counts = df.groupby(['year', 'month']).size().reset_index(name='Κριτικές')
    fig_line = px.line(monthly_counts, x='month', y='Κριτικές', color='year', 
                       markers=True, title="Επισκέψεις (βάσει κριτικών) ανά Μήνα & Έτος")
    st.plotly_chart(fig_line, use_container_width=True)

with c2:
    # Heatmap: Έτος x Μήνας
    pivot_heat = df.pivot_table(index='year', columns='month', values='rating', aggfunc='count').fillna(0)
    fig_heat = px.imshow(pivot_heat, text_auto=True, aspect="auto",
                         title="Heatmap Κριτικών: Πότε επισκέπτονται την Καστοριά;",
                         labels=dict(x="Μήνας", y="Έτος", color="Πλήθος"))
    st.plotly_chart(fig_heat, use_container_width=True)

# --- SECTION 2: Γλώσσες & Προέλευση ---
st.header("🌍 Γεωγραφική & Γλωσσική Κατανομή")
c3, c4 = st.columns(2)

with c3:
    lang_avg = df.groupby('lang')['rating'].agg(['count', 'mean']).reset_index()
    fig_lang = px.bar(lang_avg, x='lang', y='count', color='mean',
                      title="Κριτικές ανά Γλώσσα & Μέση Βαθμολογία",
                      labels={'count':'Πλήθος', 'mean':'Rating'})
    st.plotly_chart(fig_lang, use_container_width=True)

with c4:
    top_cities = df['user/userLocation/name'].value_counts().head(20).reset_index()
    fig_cities = px.bar(top_cities, x='count', y='user/userLocation/name', orientation='h',
                        title="Top 20 Πόλεις Προέλευσης Επισκεπτών",
                        color_discrete_sequence=['#4c78a8'])
    fig_cities.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_cities, use_container_width=True)

# --- SECTION 3: Τύπος Ταξιδιού & Ικανοποίηση ---
st.header("👥 Προφίλ Επισκέπτη & Sentiment")
c5, c6 = st.columns(2)

with c5:
    trip_heat = df.pivot_table(index='tripType', columns='rating', values='id', aggfunc='count').fillna(0)
    fig_trip = px.imshow(trip_heat, text_auto=True, title="Heatmap: Τύπος Ταξιδιού x Βαθμολογία")
    st.plotly_chart(fig_trip, use_container_width=True)

with c6:
    # Word Cloud
    st.write("**Word Cloud Κριτικών (Key Themes)**")
    text_content = " ".join(df['text'].dropna().astype(str))
    if text_content:
        wc = WordCloud(width=800, height=400, background_color="white", 
                       stopwords=gr_stopwords, colormap='Dark2').generate(text_content)
        fig_wc, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig_wc)

# --- SECTION 4: Αλληλεπίδραση & Engagement ---
st.header("📱 Ψηφιακή Δραστηριότητα & Εμπλοκή")
c7, c8 = st.columns(2)

with c7:
    # Μέγεθος κριτικής vs Βαθμολογία με Trendline
    try:
        fig_ols = px.scatter(df, x='review_len', y='rating', trendline="ols",
                             title="Σχέση Μεγέθους Κειμένου & Βαθμολογίας",
                             labels={'review_len':'Μήκος Κειμένου (chars)', 'rating':'Βαθμολογία'},
                             opacity=0.4)
        st.plotly_chart(fig_ols, use_container_width=True)
    except:
        st.write("Απαιτούνται περισσότερα δεδομένα για τη γραμμή τάσης.")

with c8:
    # Επίπεδα Εμπλοκής Χρηστών
    # Διασφαλίζουμε ότι τα contributions είναι αριθμητικά
    rev['user/contributions/totalContributions'] = pd.to_numeric(rev['user/contributions/totalContributions'], errors='coerce').fillna(0)
    
    bins = [0, 1, 5, 20, 100, 1000000]
    labels = ["Newbie (1)", "Explorer (2-5)", "Contributor (6-20)", "Expert (21-100)", "Local Guide (100+)"]
    
    # Χρήση του φιλτραρισμένου df
    df['engagement_level'] = pd.cut(df['user/contributions/totalContributions'], bins=bins, labels=labels)
    
    # Σωστή προετοιμασία του dataframe για την Plotly
    eng_dist = df['engagement_level'].value_counts().reindex(labels).reset_index()
    
    # Ονομάζουμε ρητά τις στήλες για να τις βρει η Plotly Express
    eng_dist.columns = ['Level', 'Counts'] 
    
    fig_eng = px.pie(eng_dist, 
                     values='Counts', 
                     names='Level', 
                     title="Ποιοι μας κρίνουν; (Επίπεδο Εμπειρίας)",
                     hole=0.4, # Μετατροπή σε Donut chart για πιο modern look
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    
    st.plotly_chart(fig_eng, use_container_width=True)

# --- SECTION 5: Συγκριτική Ανάλυση Αξιοθέατων (Μόνο αν είναι επιλεγμένο το 'Όλα') ---
if selected_place == "Όλα τα Αξιοθέατα":
    st.header("📊 Συγκριτική Κατάταξη Αξιοθέατων")
    compare_df = df.groupby('placeInfo/name').agg({
        'rating': 'mean',
        'id': 'count'
    }).reset_index().rename(columns={'id': 'Πλήθος Κριτικών', 'rating': 'Μέση Βαθμολογία'})
    
    fig_compare = px.scatter(compare_df, x='Πλήθος Κριτικών', y='Μέση Βαθμολογία', 
                             text='placeInfo/name', size='Πλήθος Κριτικών',
                             title="Positioning Map: Δημοφιλία vs Ικανοποίηση")
    fig_compare.update_traces(textposition='top center')
    st.plotly_chart(fig_compare, use_container_width=True)

st.markdown("---")
st.caption("Strategic Analytics for Kastoria Tourism | Developed with Streamlit & Plotly")
