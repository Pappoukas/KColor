import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob

# Ρύθμιση σελίδας
st.set_page_config(page_title="🏛️ Kastoria Tourism Analytics Dashboard", layout="wide")
st.title("🏛️ Kastoria Tourism Visual Analytics Dashboard")
st.markdown("Ανάλυση κριτικών TripAdvisor & χρωματικών δεδομένων φωτογραφιών")

# ------------------------------
# 1. Φόρτωση δεδομένων
# ------------------------------
@st.cache_data
def load_data():
    # Φόρτωση TripAdvisor κριτικών
    df_reviews = pd.read_csv("TripAdvisor_Kastoria.csv", sep=";", encoding="utf-8", low_memory=False)
    # Φόρτωση αξιοθέατων
    df_places = pd.read_csv("ThingsToDo.csv", sep=";", encoding="utf-8")
    
    # Φόρτωση χρωματικών αρχείων (νέα ονόματα)
    df_means = pd.read_csv("color_summary_Means.csv", sep=";", decimal=",", encoding="utf-8")
    df_colors = pd.read_csv("color_summary_Colors.csv", sep=";", decimal=",", encoding="utf-8")
    df_stats = pd.read_csv("color_summary_Statistics.csv", sep=";", decimal=",", encoding="utf-8")
    df_info = pd.read_csv("color_summary_Info.csv", sep=";", encoding="utf-8")
    
    # Καθαρισμός ονομάτων στηλών
    df_means.columns = df_means.columns.str.strip()
    df_colors.columns = df_colors.columns.str.strip()
    df_stats.columns = df_stats.columns.str.strip()
    df_info.columns = df_info.columns.str.strip()
    
    # Μετατροπή ημερομηνίας
    df_reviews['publishedDate'] = pd.to_datetime(df_reviews['publishedDate'], format='%d/%m/%Y', errors='coerce')
    
    # Συγχώνευση με αξιοθέατα για ονόματα
    df_reviews = df_reviews.merge(df_places[['placeInfo/id', 'placeInfo/name']], on='placeInfo/id', how='left')
    
    # Στο color_summary_Info.csv, η στήλη "Review/ID" περιέχει το id της κριτικής
    df_info = df_info.rename(columns={'Review/ID': 'id'})
    df_info['id'] = df_info['id'].astype(str)
    df_reviews['id'] = df_reviews['id'].astype(str)
    
    # Συγχώνευση με info (σύνδεση χρωματικών δεδομένων με κριτικές)
    df_merged = df_reviews.merge(df_info, on='id', how='left', suffixes=('', '_color'))
    
    return df_reviews, df_places, df_means, df_colors, df_stats, df_info, df_merged

try:
    df_reviews, df_places, df_means, df_colors, df_stats, df_info, df_merged = load_data()
    st.success(f"✅ Φορτώθηκαν {len(df_reviews)} κριτικές και {len(df_places)} αξιοθέατα.")
except Exception as e:
    st.error(f"Σφάλμα φόρτωσης: {e}")
    st.stop()

# Προσθήκη κατηγοριών για τα αξιοθέατα (βάσει README)
category_map = {
    "Kastoria Lake": "🌿 Φυσικό",
    "Cave of Dragon (Spilia tou drakou)": "🌿 Φυσικό",
    "Prophet Elias": "🌿 Φυσικό",
    "Panagia Mavriotissa Monastery": "🏛️ Πολιτιστικό",
    "Byzantine Museum of Kastoria": "🏛️ Πολιτιστικό",
    "Folklore Museum of Kastoria": "🏛️ Πολιτιστικό",
    "Wax Museum of Mavrochoriou Kastorias": "🏛️ Πολιτιστικό",
    "Kastorian Byzantine Churches.": "🏛️ Πολιτιστικό",
    "Fossilized Forest": "🏛️ Πολιτιστικό",
    "Endymatologiko Mouseio": "🏛️ Πολιτιστικό",
    "Church of the Panagia Koumbelidiki": "🏛️ Πολιτιστικό",
    "Church of St. Taksiarkhov u Mitropolii": "🏛️ Πολιτιστικό",
    "Kastoria Aquarium": "🎯 Δραστηριότητα",
    "Adventure Kastoria": "🎯 Δραστηριότητα",
    "Kastoria Outdoors": "🎯 Δραστηριότητα",
    "Culture 8 Cultural City and Nature Guided Day Tours": "🎯 Δραστηριότητα",
    "Mountain Lunatics": "🎯 Δραστηριότητα",
    "PANIK RENTALS": "🎯 Δραστηριότητα"
}
df_reviews['Category'] = df_reviews['placeInfo/name_y'].map(category_map).fillna("🏛️ Πολιτιστικό")

# ------------------------------
# Sidebar φίλτρα
# ------------------------------
st.sidebar.header("🔍 Φίλτρα")
selected_attractions = st.sidebar.multiselect(
    "Επιλέξτε αξιοθέατα",
    options=df_reviews['placeInfo/name_y'].dropna().unique(),
    default=df_reviews['placeInfo/name_y'].value_counts().head(5).index.tolist()
)
if selected_attractions:
    df_filtered = df_reviews[df_reviews['placeInfo/name_y'].isin(selected_attractions)]
else:
    df_filtered = df_reviews.copy()

# ------------------------------
# Tabs
# ------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📊 Dashboard", "📈 Δημοφιλία", "⭐ Βαθμολογίες", "💬 Κείμενο & Sentiment",
    "🌍 Γλώσσα & Προέλευση", "👥 Τύπος Ταξιδιού", "📸 Αλληλεπίδραση", "🎨 Χρωματική Ανάλυση", "📐 Συγκριτικές"
])

# ------------------------------
# Tab 1: Dashboard
# ------------------------------
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Σύνολο κριτικών", len(df_filtered))
    col2.metric("Μέση βαθμολογία", f"{df_filtered['rating'].mean():.2f}")
    col3.metric("Ποσοστό 5★", f"{100 * (df_filtered['rating'] == 5).mean():.1f}%")
    col4.metric("Σύνολο helpful votes", df_filtered['helpfulVotes'].sum())
    
    st.subheader("📊 Κριτικές ανά αξιοθέατο (επιλεγμένα)")
    counts = df_filtered['placeInfo/name_y'].value_counts().reset_index()
    counts.columns = ['Αξιοθέατο', 'Πλήθος']
    fig = px.bar(counts, x='Αξιοθέατο', y='Πλήθος', title="Αριθμός κριτικών")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("⭐ Μέση βαθμολογία ανά αξιοθέατο")
    avg = df_filtered.groupby('placeInfo/name_y')['rating'].mean().sort_values(ascending=False).reset_index()
    avg.columns = ['Αξιοθέατο', 'Μέση βαθμολογία']
    fig = px.bar(avg, x='Μέση βαθμολογία', y='Αξιοθέατο', orientation='h')
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("🗺️ Χάρτης αξιοθέατων")
    if 'placeInfo/latitude' in df_places.columns and 'placeInfo/longitude' in df_places.columns:
        fig = px.scatter_mapbox(df_places, lat='placeInfo/latitude', lon='placeInfo/longitude',
                                hover_name='placeInfo/name', zoom=12, height=500)
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Tab 2: Δημοφιλία
# ------------------------------
with tab2:
    st.subheader("📅 Χρονική εξέλιξη κριτικών")
    monthly = df_filtered.groupby(df_filtered['publishedDate'].dt.to_period('M')).size().reset_index(name='count')
    monthly['publishedDate'] = monthly['publishedDate'].astype(str)
    fig = px.line(monthly, x='publishedDate', y='count', title="Αριθμός κριτικών ανά μήνα")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📆 Εποχικότητα (heatmap μήνας-έτος)")
    df_filtered['year'] = df_filtered['publishedDate'].dt.year
    df_filtered['month'] = df_filtered['publishedDate'].dt.month
    heat = df_filtered.groupby(['year', 'month']).size().reset_index(name='count')
    fig = px.density_heatmap(heat, x='month', y='year', z='count', title="Εποχικότητα")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Tab 3: Βαθμολογίες
# ------------------------------
with tab3:
    st.subheader("🏆 Κατανομή βαθμολογιών")
    rating_dist = df_filtered['rating'].value_counts().sort_index().reset_index()
    rating_dist.columns = ['rating', 'count']
    fig = px.bar(rating_dist, x='rating', y='count', title="Κατανομή 1-5★")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📈 Διαχρονική τάση βαθμολογίας")
    monthly_rating = df_filtered.groupby(df_filtered['publishedDate'].dt.to_period('M'))['rating'].mean().reset_index()
    monthly_rating['publishedDate'] = monthly_rating['publishedDate'].astype(str)
    fig = px.line(monthly_rating, x='publishedDate', y='rating', title="Μέση βαθμολογία ανά μήνα")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("👥 Βαθμολογία ανά τύπο ταξιδιού")
    trip_rating = df_filtered.groupby('tripType')['rating'].mean().reset_index()
    fig = px.bar(trip_rating, x='tripType', y='rating', title="Μέση βαθμολογία ανά τύπο")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Tab 4: Κείμενο & Sentiment
# ------------------------------
with tab4:
    st.subheader("☁️ Word Cloud από κριτικές")
    all_text = " ".join(df_filtered['text'].dropna().astype(str))
    if all_text.strip():
        wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(all_text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.info("Δεν υπάρχουν κείμενα για word cloud.")
    
    st.subheader("📊 Ανάλυση συναισθήματος (Sentiment)")
    def sentiment_score(text):
        blob = TextBlob(str(text))
        return blob.sentiment.polarity
    df_filtered['sentiment'] = df_filtered['text'].apply(sentiment_score)
    st.write("Μέση πόλωση συναισθήματος:", df_filtered['sentiment'].mean())
    fig = px.histogram(df_filtered, x='sentiment', nbins=30, title="Κατανομή πόλωσης συναισθήματος")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Tab 5: Γλώσσα & Προέλευση
# ------------------------------
with tab5:
    st.subheader("🌐 Κατανομή γλωσσών κριτικών")
    lang_counts = df_filtered['lang'].value_counts().reset_index()
    lang_counts.columns = ['lang', 'count']
    fig = px.pie(lang_counts, values='count', names='lang', title="Γλώσσες κριτικών")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("🗺️ Θερμογραφικός χάρτης γλώσσας ανά αξιοθέατο")
    cross = pd.crosstab(df_filtered['placeInfo/name_y'], df_filtered['lang'])
    fig = px.imshow(cross, text_auto=True, aspect="auto", title="Πλήθος κριτικών ανά γλώσσα & αξιοθέατο")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Tab 6: Τύπος Ταξιδιού
# ------------------------------
with tab6:
    st.subheader("👨‍👩‍👧‍👦 Προτιμήσεις τύπου ταξιδιού ανά αξιοθέατο")
    trip_att = df_filtered.groupby(['placeInfo/name_y', 'tripType']).size().reset_index(name='count')
    fig = px.bar(trip_att, x='placeInfo/name_y', y='count', color='tripType', title="Τύπος ταξιδιού ανά αξιοθέατο", barmode='stack')
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📊 Κατανομή τύπων ταξιδιού")
    trip_total = df_filtered['tripType'].value_counts().reset_index()
    trip_total.columns = ['tripType', 'count']
    fig = px.pie(trip_total, values='count', names='tripType', title="Συνολική κατανομή")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Tab 7: Αλληλεπίδραση
# ------------------------------
with tab7:
    st.subheader("📸 Φωτογραφίες ανά κριτική")
    fig = px.box(df_filtered, x='placeInfo/name_y', y='Photocount', title="Κατανομή αριθμού φωτογραφιών ανά αξιοθέατο")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("👍 Χρήσιμες ψήφοι ανά αξιοθέατο")
    fig = px.box(df_filtered, x='placeInfo/name_y', y='helpfulVotes', title="Χρήσιμες ψήφοι ανά αξιοθέατο")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("💬 Απαντήσεις διαχειριστών")
    has_response = df_filtered['ownerResponse/text'].notna().sum()
    st.metric("Κριτικές με απάντηση", has_response)

# ------------------------------
# Tab 8: Χρωματική Ανάλυση (ανανεωμένο με βάση νέα αρχεία)
# ------------------------------
with tab8:
    st.subheader("🎨 Χρωματικές παλέτες ανά αξιοθέατο (από color_summary_Colors.csv)")
    
    # Συγχώνευση colors με info και reviews για να πάρουμε το αξιοθέατο
    df_color_place = df_colors.merge(df_info, on='#', how='inner')
    df_color_place = df_color_place.merge(df_reviews[['id', 'placeInfo/name_y']], on='id', how='inner')
    
    # Για κάθε αξιοθέατο, συλλέγουμε όλα τα χρώματα (HEX) και τα ποσοστά (%)
    color_data = []
    for _, row in df_color_place.iterrows():
        place = row['placeInfo/name_y']
        if pd.isna(place):
            continue
        # Στα color_summary_Colors.csv υπάρχουν στήλες 'HEX' και '%'
        if pd.notna(row['HEX']) and pd.notna(row['%']):
            color_data.append({
                'place': place,
                'HEX': row['HEX'],
                'pct': row['%']
            })
    
    if color_data:
        df_colors_agg = pd.DataFrame(color_data)
        # Για κάθε αξιοθέατο, παίρνουμε τα top 5 χρώματα βάσει ποσοστού
        top_colors = df_colors_agg.groupby('place').apply(lambda g: g.nlargest(5, 'pct')).reset_index(drop=True)
        
        for place in top_colors['place'].unique():
            place_colors = top_colors[top_colors['place'] == place]
            st.write(f"**{place}**")
            cols = st.columns(len(place_colors))
            for i, (_, row) in enumerate(place_colors.iterrows()):
                cols[i].color_picker(f"Χρώμα {i+1}", value=row['HEX'], disabled=True, key=f"{place}_{i}_{row['HEX']}")
    else:
        st.info("Δεν βρέθηκαν χρωματικά δεδομένα για τα επιλεγμένα αξιοθέατα.")
    
    st.subheader("📊 Διασπορά κορεσμού (S%) και φωτεινότητας (V%)")
    # Χρησιμοποιούμε τα statistics που ήδη έχουμε
    if 'S%_mean' in df_stats.columns and 'V%_mean' in df_stats.columns:
        stats_merged = df_stats.merge(df_info, on='#').merge(df_reviews[['id', 'placeInfo/name_y']], on='id', how='inner')
        fig = px.scatter(stats_merged, x='S%_mean', y='V%_mean', color='placeInfo/name_y',
                         title="Κορεσμός vs Φωτεινότητα ανά φωτογραφία")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Δεν υπάρχουν δεδομένα S% ή V%.")
    
    st.subheader("🌈 Διάγραμμα Chroma (LCH)")
    if 'Chroma' in df_stats.columns and 'Lightness' in df_stats.columns:
        stats_merged = df_stats.merge(df_info, on='#').merge(df_reviews[['id', 'placeInfo/name_y']], on='id', how='inner')
        fig = px.scatter(stats_merged, x='Lightness', y='Chroma', color='placeInfo/name_y',
                         title="Lightness vs Chroma")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Δεν υπάρχουν δεδομένα Chroma ή Lightness.")
    
    st.subheader("🔵🔴 Διάγραμμα a* b* (χώρος Lab)")
    if 'Green-Red' in df_stats.columns and 'Blue-Yellow' in df_stats.columns:
        stats_merged = df_stats.merge(df_info, on='#').merge(df_reviews[['id', 'placeInfo/name_y']], on='id', how='inner')
        fig = px.scatter(stats_merged, x='Green-Red', y='Blue-Yellow', color='placeInfo/name_y',
                         title="a* (πράσινο-κόκκινο) vs b* (μπλε-κίτρινο)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Δεν υπάρχουν δεδομένα a* ή b*.")

# ------------------------------
# Tab 9: Συγκριτικές
# ------------------------------
with tab9:
    st.subheader("🏛️ Σύγκριση κατηγοριών: Φυσικά vs Πολιτιστικά vs Δραστηριότητες")
    cat_rating = df_reviews.groupby('Category')['rating'].mean().reset_index()
    fig = px.bar(cat_rating, x='Category', y='rating', title="Μέση βαθμολογία ανά κατηγορία")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📈 Απλή γραμμική παλινδρόμηση: βαθμολογία vs χρόνος")
    df_time = df_filtered.groupby(df_filtered['publishedDate'].dt.year)['rating'].mean().reset_index()
    df_time = df_time.dropna()
    if len(df_time) > 1:
        x = df_time['publishedDate'].astype(int)
        y = df_time['rating']
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        df_time['trend'] = p(x)
        fig = px.scatter(df_time, x='publishedDate', y='rating', title="Τάση βαθμολογίας ανά έτος")
        fig.add_trace(go.Scatter(x=df_time['publishedDate'], y=df_time['trend'], mode='lines', name='Γραμμή τάσης'))
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📊 Συσχέτιση μεταξύ βαθμολογίας και αριθμού φωτογραφιών")
    corr = df_filtered[['rating', 'Photocount']].corr().iloc[0,1]
    st.metric("Συντελεστής συσχέτισης Pearson", f"{corr:.3f}")
    fig = px.scatter(df_filtered, x='Photocount', y='rating', opacity=0.5, title="Rating vs Photocount")
    st.plotly_chart(fig, use_container_width=True)
