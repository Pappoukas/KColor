import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- Ρυθμίσεις Σελίδας ---
st.set_page_config(
    page_title="Kastoria Tourism Visual Analytics",
    page_icon="🏛️",
    layout="wide"
)

# --- Συνάρτηση Φόρτωσης Δεδομένων ---
@st.cache_data
def load_data():
    # Φόρτωση βασικών αρχείων με βάση το delimiter ';' που είδαμε στα δείγματα
    df_reviews = pd.read_csv("TripAdvisor_Kastoria.csv", sep=";")
    df_places = pd.read_csv("ThingsToDo.csv", sep=";")
    df_colors = pd.read_csv("color_summary_Summary.csv", sep=";")
    df_clusters = pd.read_csv("color_summary_Clusters.csv", sep=";")
    
    # Μετατροπή ημερομηνιών
    df_reviews['publishedDate'] = pd.to_datetime(df_reviews['publishedDate'], errors='coerce')
    
    return df_reviews, df_places, df_colors, df_clusters

try:
    rev, places, colors, clusters = load_data()
except Exception as e:
    st.error(f"Σφάλμα κατά τη φόρτωση των αρχείων: {e}")
    st.stop()

# --- Sidebar / Φίλτρα ---
st.sidebar.header("🔍 Φίλτρα Αναζήτησης")
selected_place = st.sidebar.selectbox("Επιλέξτε Αξιοθέατο:", ["Όλα"] + list(places['placeInfo/name'].unique()))

# Φιλτράρισμα δεδομένων
if selected_place != "Όλα":
    rev_filtered = rev[rev['placeInfo/name'] == selected_place]
    # Εύρεση ID για τα χρώματα
    place_id = places[places['placeInfo/name'] == selected_place]['placeInfo/id'].values[0]
    # Σημείωση: Στο color_summary_Summary το '#' αντιστοιχεί στο ID ή index
else:
    rev_filtered = rev

# --- Κύριο Περιεχόμενο ---
st.title("🏛️ Kastoria Tourism Visual Analytics")
st.markdown("Ανάλυση τουριστικής εικόνας μέσω φωτογραφιών και κριτικών TripAdvisor.")

# --- Section 1: Στατιστικά Κριτικών (KPIs) ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Συνολικές Κριτικές", len(rev_filtered))
with col2:
    st.metric("Μέση Βαθμολογία", round(rev_filtered['rating'].mean(), 2))
with col3:
    st.metric("Αξιοθέατα", len(places))
with col4:
    st.metric("Φωτογραφίες που Αναλύθηκαν", "1.604")

# --- Section 2: Χρωματική Ταυτότητα (Από README) ---
st.subheader("🎨 Οπτική Ταυτότητα & Χρωματική Ανάλυση")
c1, c2 = st.columns([1, 1])

with c1:
    st.info("Κυρίαρχα χρώματα στις φωτογραφίες των επισκεπτών")
    # Παράδειγμα πίτας από το color_summary_Summary
    if selected_place == "Όλα":
        top_colors = colors['Color_1_Name'].value_counts().head(10)
        fig_col = px.pie(values=top_colors.values, names=top_colors.index, 
                         title="Συχνότητα Χρωμάτων (Top 10)",
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_col, use_container_width=True)

with c2:
    # Word Cloud από τις κριτικές
    st.info("Word Cloud Κριτικών (Sentiment)")
    text = " ".join(review for review in rev_filtered['text'].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(text)
    
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# --- Section 3: Ανάλυση Ικανοποίησης ---
st.subheader("📈 Τάσεις & Ικανοποίηση")
# Γράφημα βαθμολογιών ανά τύπο ταξιδιού
if 'tripType' in rev_filtered.columns:
    fig_trip = px.box(rev_filtered, x='tripType', y='rating', color='tripType',
                      title="Βαθμολογία ανά Τύπο Ταξιδιού")
    st.plotly_chart(fig_trip, use_container_width=True)

# --- Section 4: Γεωγραφική Κατανομή ---
st.subheader("📍 Τοποθεσία Αξιοθέατων")
if 'placeInfo/latitude' in places.columns:
    map_data = places[['placeInfo/latitude', 'placeInfo/longitude', 'placeInfo/name']].dropna()
    # Διόρθωση αν οι συντεταγμένες έχουν κόμμα αντί για τελεία
    map_data['lat'] = map_data['placeInfo/latitude'].astype(str).str.replace(',','.').astype(float)
    map_data['lon'] = map_data['placeInfo/longitude'].astype(str).str.replace(',','.').astype(float)
    st.map(map_data[['lat', 'lon']])

st.markdown("---")
st.caption("Data Source: TripAdvisor & Photo Color Analysis | Kastoria Tourism Dashboard")
