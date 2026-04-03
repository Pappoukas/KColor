import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- Ρυθμίσεις Σελίδας ---
st.set_page_config(page_title="Kastoria Tourism Expert Analytics", layout="wide")

# --- Φόρτωση Δεδομένων & Stopwords ---
@st.cache_data
def load_data():
    rev = pd.read_csv("TripAdvisor_Kastoria.csv", sep=";")
    places = pd.read_csv("ThingsToDo.csv", sep=";")
    
    # Καθαρισμός Ημερομηνιών
    rev['publishedDate'] = pd.to_datetime(rev['publishedDate'], dayfirst=True, errors='coerce')
    rev['year'] = rev['publishedDate'].dt.year
    rev['month'] = rev['publishedDate'].dt.month
    
    # Φόρτωση Stopwords
    try:
        with open("greek_stopwords.txt", "r", encoding="utf-8") as f:
            stopwords = set(f.read().split(","))
    except:
        stopwords = set()
    
    return rev, places, stopwords

rev, places, gr_stopwords = load_data()

# --- Sidebar ---
st.sidebar.header("📊 Στρατηγικά Φίλτρα")
selected_place = st.sidebar.selectbox("Επιλέξτε Αξιοθέατο:", ["Όλα"] + list(places['placeInfo/name'].unique()))

# Φιλτράρισμα
df = rev if selected_place == "Όλα" else rev[rev['placeInfo/name'] == selected_place]

# --- 1. Κεντρικά KPIs ---
st.title(f"🏛️ Ανάλυση: {selected_place}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Συνολικές Κριτικές", len(df))
c2.metric("Μέση Βαθμολογία", round(df['rating'].mean(), 2))
c3.metric("Helpful Votes", int(df['user/contributions/helpfulVotes'].sum()))
# Response Rate (Απαντήσεις Διαχειριστών)
resp_rate = (df['ownerResponse/text'].notna().sum() / len(df)) * 100
c4.metric("Admin Response %", f"{resp_rate:.1f}%")

# --- 2. Χρονική Ανάλυση & Heatmaps ---
st.subheader("📈 Χρονική Εξέλιξη & Εποχικότητα")
tab1, tab2 = st.tabs(["Επισκέψεις & Τάσεις", "Heatmap (Έτος x Μήνας)"])

with tab1:
    # Επισκέψεις ανά έτος/μήνα
    trend_data = df.groupby(['year', 'month']).size().reset_index(name='counts')
    fig_trend = px.line(trend_data, x='month', y='counts', color='year', title="Επισκέψεις ανά Μήνα και Έτος")
    st.plotly_chart(fig_trend, use_container_width=True)

with tab2:
    # Heatmap
    heat_data = df.pivot_table(index='year', columns='month', values='rating', aggfunc='count').fillna(0)
    fig_heat = px.imshow(heat_data, labels=dict(x="Μήνας", y="Έτος", color="Κριτικές"), title="Heatmap Πυκνότητας Κριτικών")
    st.plotly_chart(fig_heat, use_container_width=True)

# --- 3. Ανάλυση Βαθμολογιών ---
st.subheader("⭐ Ποιοτική Ανάλυση Βαθμολογίας")
col_a, col_b = st.columns(2)
with col_a:
    fig_dist = px.histogram(df, x='rating', nbins=5, title="Κατανομή Βαθμολογιών (1-5)")
    st.plotly_chart(fig_dist, use_container_width=True)
with col_b:
    # Διαχρονική τάση μέσης βαθμολογίας
    rating_trend = df.groupby('year')['rating'].mean().reset_index()
    fig_rt = px.area(rating_trend, x='year', y='rating', title="Διαχρονική Τάση Μέσης Βαθμολογίας")
    st.plotly_chart(fig_rt, use_container_width=True)

# --- 4. Γλωσσική & Γεωγραφική Προέλευση ---
st.subheader("🌍 Γλώσσες & Προέλευση Επισκεπτών")
c_lang, c_city = st.columns(2)

with c_lang:
    lang_dist = df['lang'].value_counts().reset_index()
    fig_lang = px.pie(lang_dist, values='count', names='lang', title="Κατανομή Γλωσσών")
    st.plotly_chart(fig_lang, use_container_width=True)

with c_city:
    # Top 20 Πόλεις
    top_cities = df['user/userLocation/name'].value_counts().head(20).reset_index()
    fig_city = px.bar(top_cities, x='count', y='user/userLocation/name', orientation='h', title="Top 20 Πόλεις Προέλευσης")
    st.plotly_chart(fig_city, use_container_width=True)

# --- 5. Τύπος Ταξιδιού & Συναίσθημα ---
st.subheader("👥 Προφίλ Επισκέπτη")
t1, t2 = st.columns(2)
with t1:
    fig_trip = px.sunburst(df.dropna(subset=['tripType']), path=['tripType', 'rating'], title="Τύπος Ταξιδιού & Ικανοποίηση")
    st.plotly_chart(fig_trip, use_container_width=True)
with t2:
    # Word Cloud με Stopwords
    valid_text = " ".join(df['text'].dropna().astype(str))
    if valid_text:
        wc = WordCloud(width=800, height=400, background_color="white", stopwords=gr_stopwords).generate(valid_text)
        fig_wc, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig_wc)

# --- 6. Αλληλεπίδραση & Δραστηριότητα ---
st.subheader("📱 Ψηφιακό Αποτύπωμα & Δραστηριότητα")
row1_1, row1_2 = st.columns(2)

with row1_1:
    # Φωτογραφίες ανά βαθμολογία
    fig_photo = px.box(df, x='rating', y='Photocount', 
                       title="Αριθμός Φωτογραφιών ανά Βαθμολογία",
                       color_discrete_sequence=['#2d4a6b'])
    st.plotly_chart(fig_photo, use_container_width=True)

with row1_2:
    # Μέγεθος κριτικής vs Βαθμολογία
    df['review_len'] = df['text'].str.len().fillna(0)
    
    # Χρήση try-except για την περίπτωση που λείπει το statsmodels ή δεν υπάρχουν αρκετά δεδομένα
    try:
        fig_len = px.scatter(df, x='review_len', y='rating', 
                           trendline="ols", 
                           title="Μέγεθος Κειμένου vs Βαθμολογία (Regression Analysis)",
                           labels={'review_len': 'Χαρακτήρες κειμένου', 'rating': 'Βαθμολογία'},
                           opacity=0.5)
        st.plotly_chart(fig_len, use_container_width=True)
    except Exception as e:
        # Fallback γράφημα χωρίς trendline αν αποτύχει η παλινδρόμηση
        fig_len_basic = px.scatter(df, x='review_len', y='rating', 
                                 title="Μέγεθος Κειμένου vs Βαθμολογία",
                                 opacity=0.5)
        st.plotly_chart(fig_len_basic, use_container_width=True)
        st.caption("Σημείωση: Η γραμμή τάσης δεν είναι διαθέσιμη (απαιτείται statsmodels).")

# --- 7. Επίπεδο Εμπλοκής (User Contributions) ---
st.subheader("🎖️ Προφίλ Εμπειρίας Χρηστών")
bins = [0, 1, 5, 20, 100, 10000]
labels = ["1 κριτική", "2-5", "6-20", "21-100", "100+"]
df['engagement'] = pd.cut(df['user/contributions/totalContributions'], bins=bins, labels=labels)
fig_eng = px.bar(df['engagement'].value_counts().reindex(labels), title="Αριθμός Κριτικών ανά Επίπεδο Εμπειρίας Χρήστη")
st.plotly_chart(fig_eng, use_container_width=True)
