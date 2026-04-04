import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Σελίδα & τίτλος
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="🏛️ Kastoria Tourism Analytics",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏛️ Kastoria Tourism Visual Analytics Dashboard")
st.markdown(
    "Ανάλυση **2.578 κριτικών TripAdvisor** & **χρωματικών δεδομένων** από 1.604 φωτογραφίες "
    "επισκεπτών σε 18 αξιοθέατα της Καστοριάς."
)

# ──────────────────────────────────────────────
# Κατηγορίες & σταθερά δεδομένα
# ──────────────────────────────────────────────
CATEGORY_MAP = {
    "Kastoria Lake": "🌿 Φυσικό",
    "Cave of Dragon (Spilia tou drakou)": "🌿 Φυσικό",
    "Prophet Elias": "🌿 Φυσικό",
    "Panagia Mavriotissa Monastery": "🏛️ Πολιτιστικό",
    "Byzantine Museum of Kastoria": "🏛️ Πολιτιστικό",
    "Folklore Museum of Kastoria": "🏛️ Πολιτιστικό",
    "Wax Museum  of Mavrochoriou Kastorias": "🏛️ Πολιτιστικό",
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
    "PANIK RENTALS": "🎯 Δραστηριότητα",
}
CAT_COLORS = {
    "🌿 Φυσικό": "#27ae60",
    "🏛️ Πολιτιστικό": "#c0392b",
    "🎯 Δραστηριότητα": "#2980b9",
}
MONTH_EL = {
    1: "Ιαν", 2: "Φεβ", 3: "Μαρ", 4: "Απρ", 5: "Μάι", 6: "Ιουν",
    7: "Ιουλ", 8: "Αυγ", 9: "Σεπ", 10: "Οκτ", 11: "Νοε", 12: "Δεκ",
}
SEASON_MAP = {
    12: "🌨 Χειμώνας", 1: "🌨 Χειμώνας", 2: "🌨 Χειμώνας",
    3: "🌸 Άνοιξη", 4: "🌸 Άνοιξη", 5: "🌸 Άνοιξη",
    6: "☀️ Καλοκαίρι", 7: "☀️ Καλοκαίρι", 8: "☀️ Καλοκαίρι",
    9: "🍂 Φθινόπωρο", 10: "🍂 Φθινόπωρο", 11: "🍂 Φθινόπωρο",
}

# ──────────────────────────────────────────────
# Φόρτωση δεδομένων
# ──────────────────────────────────────────────
@st.cache_data(show_spinner="Φόρτωση δεδομένων…")
def load_data():
    # TripAdvisor κριτικές
    rev = pd.read_csv(
        "TripAdvisor_Kastoria.csv", sep=";", encoding="utf-8-sig", low_memory=False
    )
    rev.columns = rev.columns.str.strip()

    # Αξιοθέατα
    places = pd.read_csv("ThingsToDo.csv", sep=";", encoding="utf-8-sig")
    places.columns = places.columns.str.strip()
    for col in ["placeInfo/latitude", "placeInfo/longitude"]:
        places[col] = (
            places[col].astype(str).str.replace(",", ".", regex=False)
            .replace("nan", np.nan).astype(float)
        )

    # Αν το TripAdvisor CSV έχει ήδη placeInfo/name, αφαιρούμε πριν το merge
    if "placeInfo/name" in rev.columns:
        rev = rev.drop(columns=["placeInfo/name"])

    # Συγχώνευση για ονόματα / συντεταγμένες
    rev = rev.merge(
        places[["placeInfo/id", "placeInfo/name",
                "placeInfo/latitude", "placeInfo/longitude"]],
        on="placeInfo/id", how="left",
    )
    rev["place_name"] = rev["placeInfo/name"].fillna("Άγνωστο")

    # Παραγόμενες στήλες
    rev["Category"] = rev["place_name"].map(CATEGORY_MAP).fillna("🏛️ Πολιτιστικό")
    rev["publishedDate"] = pd.to_datetime(
        rev["publishedDate"], format="%d/%m/%Y", errors="coerce"
    )
    rev["year"] = rev["publishedDate"].dt.year
    rev["month"] = rev["publishedDate"].dt.month
    rev["month_name"] = rev["month"].map(MONTH_EL)
    rev["season"] = rev["month"].map(SEASON_MAP)
    rev["text_len"] = rev["text"].astype(str).str.len()
    rev["has_response"] = rev["ownerResponse/text"].notna().astype(int)
    rev["travelDate_dt"] = pd.to_datetime(rev["travelDate"], format="%Y-%m", errors="coerce")
    rev["review_lag_days"] = (rev["publishedDate"] - rev["travelDate_dt"]).dt.days
    rev["contributions"] = pd.to_numeric(
        rev["user/contributions/totalContributions"], errors="coerce"
    )

    def contrib_level(n):
        if pd.isna(n):
            return "Άγνωστο"
        n = int(n)
        if n <= 1:
            return "Νέος (1)"
        if n <= 10:
            return "Αρχάριος (2–10)"
        if n <= 50:
            return "Ενεργός (11–50)"
        if n <= 200:
            return "Έμπειρος (51–200)"
        return "Πολύ Έμπειρος (200+)"

    rev["contrib_level"] = rev["contributions"].apply(contrib_level)

    # Χρωματικά αρχεία
    info = pd.read_csv("color_summary_Info.csv", sep=";", encoding="utf-8-sig")
    info.columns = info.columns.str.strip()
    info = info.rename(columns={"Review/ID": "id"})
    info["id"] = info["id"].astype(str)
    rev["id"] = rev["id"].astype(str)

    if "W×H" in info.columns:
        dims = info["W×H"].astype(str).str.split("×", expand=True)
        if dims.shape[1] >= 2:
            info["photo_w"] = pd.to_numeric(dims[0], errors="coerce")
            info["photo_h"] = pd.to_numeric(dims[1], errors="coerce")
            info["megapixels"] = (info["photo_w"] * info["photo_h"]) / 1_000_000

    colors_p = pd.read_csv(
        "color_summary_Colors.csv", sep=";", decimal=",", encoding="utf-8-sig"
    )
    colors_p.columns = colors_p.columns.str.strip()
    colors_p["pct_num"] = pd.to_numeric(
        colors_p["%"].astype(str).str.replace(",", ".", regex=False), errors="coerce"
    )

    clusters_p = pd.read_csv(
        "color_summary_Clusters.csv", sep=";", decimal=",", encoding="utf-8-sig"
    )
    clusters_p.columns = clusters_p.columns.str.strip()

    means_p = pd.read_csv(
        "color_summary_Means.csv", sep=";", decimal=",", encoding="utf-8-sig"
    )
    means_p.columns = means_p.columns.str.strip()

    stats_p = pd.read_csv(
        "color_summary_Statistics.csv", sep=";", decimal=",", encoding="utf-8-sig"
    )
    stats_p.columns = stats_p.columns.str.strip()

    return rev, places, info, colors_p, clusters_p, means_p, stats_p


try:
    rev, places, info, colors_p, clusters_p, means_p, stats_p = load_data()
    st.success(
        f"✅ Φορτώθηκαν **{len(rev):,}** κριτικές · "
        f"**{len(info):,}** φωτογραφίες · "
        f"**{rev['place_name'].nunique()}** αξιοθέατα"
    )
except Exception as e:
    st.error(f"❌ Σφάλμα φόρτωσης δεδομένων: {e}")
    st.stop()

# ──────────────────────────────────────────────
# Sidebar φίλτρα
# ──────────────────────────────────────────────
st.sidebar.header("🔍 Φίλτρα")

all_places = sorted(rev["place_name"].dropna().unique())
sel_places = st.sidebar.multiselect(
    "Αξιοθέατα", options=all_places, default=all_places
)

valid_years = rev["year"].dropna().astype(int)
yr_min, yr_max = int(valid_years.min()), int(valid_years.max())
sel_years = st.sidebar.slider("Εύρος ετών", yr_min, yr_max, (yr_min, yr_max))

all_cats = sorted(rev["Category"].dropna().unique())
sel_cats = st.sidebar.multiselect("Κατηγορίες", options=all_cats, default=all_cats)

all_langs = sorted(rev["lang"].dropna().unique())
default_langs = all_langs[:min(8, len(all_langs))]
sel_langs = st.sidebar.multiselect("Γλώσσες", options=all_langs, default=default_langs)


def apply_filters(df):
    p = sel_places if sel_places else all_places
    c = sel_cats if sel_cats else all_cats
    l = sel_langs if sel_langs else all_langs
    return df[
        df["place_name"].isin(p)
        & df["year"].between(sel_years[0], sel_years[1])
        & df["Category"].isin(c)
        & df["lang"].isin(l)
    ].copy()


df = apply_filters(rev)

if df.empty:
    st.warning("⚠️ Κανένα αποτέλεσμα με τα επιλεγμένα φίλτρα.")
    st.stop()

st.sidebar.markdown(f"**{len(df):,}** κριτικές επιλεγμένες")

# ──────────────────────────────────────────────
# Βοηθητικές συναρτήσεις
# ──────────────────────────────────────────────
def palette_html(hex_list, pct_list, height=34):
    total = sum(pct_list) or 1
    parts = "".join(
        f'<div title="{h} ({p:.1f}%)" '
        f'style="background:{h};width:{100*p/total:.1f}%;height:{height}px;display:inline-block;"></div>'
        for h, p in zip(hex_list, pct_list)
    )
    return f'<div style="width:100%;display:flex;border-radius:4px;overflow:hidden;">{parts}</div>'


def sentiment_score(text):
    try:
        return TextBlob(str(text)).sentiment.polarity
    except Exception:
        return 0.0


# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
(tab1, tab2, tab3, tab4,
 tab5, tab6, tab7, tab8, tab9) = st.tabs([
    "📊 Dashboard", "📈 Δημοφιλία", "⭐ Βαθμολογίες",
    "💬 Κείμενο & Sentiment", "🌍 Γλώσσα & Προέλευση",
    "👥 Τύπος Ταξιδιού", "📸 Αλληλεπίδραση",
    "🎨 Χρωματική Ανάλυση", "📐 Συγκριτικές & Insights",
])

# ════════════════════════════════════════════════
# TAB 1 — Dashboard
# ════════════════════════════════════════════════
with tab1:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Κριτικές", f"{len(df):,}")
    c2.metric("Μέση βαθμολογία", f"{df['rating'].mean():.2f} ⭐")
    c3.metric("Ποσοστό 5★", f"{100*(df['rating']==5).mean():.1f}%")
    c4.metric("Helpful votes", f"{df['helpfulVotes'].sum():,}")
    c5.metric("Αξιοθέατα", df["place_name"].nunique())
    c6.metric("Γλώσσες", df["lang"].nunique())

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("📊 Κριτικές ανά αξιοθέατο")
        counts = (
            df["place_name"].value_counts()
            .reset_index()
            .rename(columns={"place_name": "Αξιοθέατο", "count": "Κριτικές"})
        )
        counts["Κατηγορία"] = counts["Αξιοθέατο"].map(CATEGORY_MAP).fillna("🏛️ Πολιτιστικό")
        fig = px.bar(
            counts, x="Κριτικές", y="Αξιοθέατο", orientation="h",
            color="Κατηγορία", color_discrete_map=CAT_COLORS, height=500,
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, legend_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("⭐ Μέση βαθμολογία ανά αξιοθέατο")
        avg = (
            df.groupby("place_name")["rating"]
            .agg(mean="mean", count="count")
            .reset_index()
            .rename(columns={"place_name": "Αξιοθέατο", "mean": "Βαθμολογία", "count": "n"})
            .sort_values("Βαθμολογία")
        )
        avg["Κατηγορία"] = avg["Αξιοθέατο"].map(CATEGORY_MAP).fillna("🏛️ Πολιτιστικό")
        fig = px.bar(
            avg, x="Βαθμολογία", y="Αξιοθέατο", orientation="h",
            color="Κατηγορία", color_discrete_map=CAT_COLORS,
            text=avg["Βαθμολογία"].round(2),
            hover_data={"n": True}, height=500,
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(xaxis_range=[0, 5.5], legend_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🗺️ Χάρτης αξιοθέατων Καστοριάς")
    map_data = (
        df.groupby("place_name")
        .agg(
            rating=("rating", "mean"),
            count=("rating", "count"),
            lat=("placeInfo/latitude", "first"),
            lon=("placeInfo/longitude", "first"),
        )
        .reset_index()
        .dropna(subset=["lat", "lon"])
    )
    map_data["Κατηγορία"] = map_data["place_name"].map(CATEGORY_MAP).fillna("🏛️ Πολιτιστικό")
    if not map_data.empty:
        fig = px.scatter_mapbox(
            map_data, lat="lat", lon="lon",
            hover_name="place_name",
            size="count", color="Κατηγορία",
            color_discrete_map=CAT_COLORS,
            hover_data={"rating": ":.2f", "count": True, "lat": False, "lon": False},
            zoom=12, height=500, size_max=40,
        )
        fig.update_layout(mapbox_style="open-street-map", legend_title="")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Δεν υπάρχουν έγκυρες συντεταγμένες για εμφάνιση χάρτη.")

# ════════════════════════════════════════════════
# TAB 2 — Δημοφιλία
# ════════════════════════════════════════════════
with tab2:
    st.subheader("📅 Χρονική εξέλιξη κριτικών (μηνιαία)")
    monthly = (
        df.groupby(df["publishedDate"].dt.to_period("M"))
        .size().reset_index(name="Κριτικές")
    )
    monthly["Μήνας"] = monthly["publishedDate"].astype(str)
    fig = px.area(monthly, x="Μήνας", y="Κριτικές",
                  title="Αριθμός κριτικών ανά μήνα")
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("📆 Heatmap: μήνας × έτος")
        heat = df.groupby(["year", "month"]).size().reset_index(name="Κριτικές")
        pivot = heat.pivot_table(
            index="year", columns="month", values="Κριτικές", aggfunc="sum"
        ).fillna(0)
        pivot.columns = [MONTH_EL.get(c, c) for c in pivot.columns]
        fig = px.imshow(
            pivot, text_auto=True, aspect="auto",
            color_continuous_scale="Blues",
            title="Heatmap: κριτικές ανά μήνα & έτος",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("🌸 Κατανομή ανά εποχή")
        seas = df["season"].value_counts().reset_index()
        seas.columns = ["Εποχή", "Κριτικές"]
        fig = px.pie(
            seas, values="Κριτικές", names="Εποχή",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            title="Κριτικές ανά εποχή",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Ετήσια εξέλιξη ανά αξιοθέατο (top 6)")
    top6 = df["place_name"].value_counts().head(6).index.tolist()
    yearly = (
        df[df["place_name"].isin(top6)]
        .groupby(["year", "place_name"])
        .size().reset_index(name="Κριτικές")
    )
    fig = px.line(
        yearly, x="year", y="Κριτικές", color="place_name",
        markers=True, title="Ετήσιες κριτικές — top 6 αξιοθέατα",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔵 Bubble chart: κριτικές ανά αξιοθέατο & έτος")
    yr_rank = df.groupby(["year", "place_name"]).size().reset_index(name="Κριτικές")
    yr_rank["Κατηγορία"] = yr_rank["place_name"].map(CATEGORY_MAP).fillna("🏛️ Πολιτιστικό")
    fig = px.scatter(
        yr_rank, x="year", y="place_name",
        size="Κριτικές", color="Κατηγορία",
        color_discrete_map=CAT_COLORS,
        title="Bubble: κριτικές ανά αξιοθέατο & έτος",
    )
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════
# TAB 3 — Βαθμολογίες
# ════════════════════════════════════════════════
with tab3:
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("🏆 Κατανομή βαθμολογιών 1–5★")
        rd = df["rating"].value_counts().sort_index().reset_index()
        rd.columns = ["Βαθμολογία", "Κριτικές"]
        rd["pct"] = (rd["Κριτικές"] / rd["Κριτικές"].sum() * 100).round(1)
        fig = px.bar(
            rd, x="Βαθμολογία", y="Κριτικές",
            text=rd["pct"].astype(str) + "%",
            color="Βαθμολογία",
            color_continuous_scale=["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"],
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("📦 Box plot ανά αξιοθέατο")
        fig = px.box(
            df, x="rating", y="place_name",
            color="Category", color_discrete_map=CAT_COLORS,
            orientation="h", title="Κατανομή βαθμολογιών",
        )
        fig.update_layout(yaxis={"categoryorder": "median ascending"}, legend_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Τάση μέσης βαθμολογίας (τριμηνιαία)")
    trend = (
        df.groupby(df["publishedDate"].dt.to_period("Q"))["rating"]
        .mean().reset_index()
    )
    trend["Τρίμηνο"] = trend["publishedDate"].astype(str)
    fig = px.line(
        trend, x="Τρίμηνο", y="rating", markers=True,
        title="Μέση βαθμολογία ανά τρίμηνο",
    )
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(range=[1, 5])
    st.plotly_chart(fig, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("🌸 Βαθμολογία ανά εποχή")
        s_avg = df.groupby("season")["rating"].mean().reset_index()
        fig = px.bar(
            s_avg, x="season", y="rating", color="season",
            text=s_avg["rating"].round(2),
            title="Μέση βαθμολογία ανά εποχή",
        )
        fig.update_traces(textposition="outside")
        fig.update_yaxes(range=[0, 5.5])
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("🗓️ Ποσοστό 5★ ανά μήνα")
        m5 = (
            df.groupby("month")
            .apply(lambda g: 100 * (g["rating"] == 5).mean())
            .reset_index(name="5★ %")
        )
        m5["Μήνας"] = m5["month"].map(MONTH_EL)
        fig = px.bar(
            m5, x="Μήνας", y="5★ %",
            text=m5["5★ %"].round(1),
            color="5★ %", color_continuous_scale="RdYlGn",
            title="Ποσοστό 5★ ανά μήνα",
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════
# TAB 4 — Κείμενο & Sentiment
# ════════════════════════════════════════════════
with tab4:
    st.subheader("☁️ Word Cloud κριτικών")
    all_text = " ".join(df["text"].dropna().astype(str).tolist())
    if all_text.strip():
        wc = WordCloud(
            width=1000, height=420, background_color="white",
            max_words=120, colormap="tab20",
        ).generate(all_text)
        fig_wc, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig_wc)
        plt.close(fig_wc)
    else:
        st.info("Δεν υπάρχουν κείμενα για word cloud.")

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("📏 Κατανομή μήκους κριτικής")
        fig = px.histogram(
            df, x="text_len", nbins=50,
            color="Category", color_discrete_map=CAT_COLORS,
            title="Μήκος κριτικής σε χαρακτήρες",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("📏 Μέσο μήκος κριτικής ανά αξιοθέατο")
        tl = (
            df.groupby("place_name")["text_len"]
            .mean().sort_values().reset_index()
        )
        tl.columns = ["Αξιοθέατο", "Μέσο μήκος"]
        fig = px.bar(
            tl, x="Μέσο μήκος", y="Αξιοθέατο", orientation="h",
            title="Μέσο μήκος κειμένου ανά αξιοθέατο",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🎭 Sentiment Analysis (TextBlob — αγγλικές κριτικές)")
    with st.spinner("Υπολογισμός sentiment…"):
        en_df = df[df["lang"] == "en"].copy()
        if not en_df.empty:
            en_df["sentiment"] = en_df["text"].apply(sentiment_score)
            col_l, col_r = st.columns(2)
            with col_l:
                st.metric("Μέση πόλωση (αγγλ.)", f"{en_df['sentiment'].mean():.3f}")
                fig = px.histogram(
                    en_df, x="sentiment", nbins=40,
                    color_discrete_sequence=["#3498db"],
                    title="Κατανομή sentiment score",
                )
                st.plotly_chart(fig, use_container_width=True)
            with col_r:
                sp = en_df.groupby("place_name")["sentiment"].mean().sort_values().reset_index()
                sp.columns = ["Αξιοθέατο", "Sentiment"]
                fig = px.bar(
                    sp, x="Sentiment", y="Αξιοθέατο", orientation="h",
                    color="Sentiment", color_continuous_scale="RdYlGn",
                    title="Μέσο sentiment ανά αξιοθέατο",
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Δεν υπάρχουν αγγλικές κριτικές στα τρέχοντα φίλτρα.")

# ════════════════════════════════════════════════
# TAB 5 — Γλώσσα & Προέλευση
# ════════════════════════════════════════════════
with tab5:
    top_langs = df["lang"].value_counts().head(10).index.tolist()

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("🌐 Κατανομή γλωσσών (top 15)")
        lc = df["lang"].value_counts().head(15).reset_index()
        lc.columns = ["Γλώσσα", "Κριτικές"]
        fig = px.pie(
            lc, values="Κριτικές", names="Γλώσσα",
            title="Γλώσσες κριτικών",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("📍 Top 15 πόλεις επισκεπτών")
        loc_col = "user/userLocation/name"
        if loc_col in df.columns:
            loc = df[loc_col].dropna().value_counts().head(15).reset_index()
            loc.columns = ["Τοποθεσία", "Κριτικές"]
            fig = px.bar(
                loc, x="Κριτικές", y="Τοποθεσία", orientation="h",
                title="Top 15 τοποθεσίες χρηστών",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔥 Heatmap: αξιοθέατο × γλώσσα (top 10 γλώσσες)")
    cross = pd.crosstab(
        df[df["lang"].isin(top_langs)]["place_name"],
        df[df["lang"].isin(top_langs)]["lang"],
    )
    fig = px.imshow(
        cross, text_auto=True, aspect="auto",
        color_continuous_scale="Blues",
        title="Κριτικές: αξιοθέατο × γλώσσα",
    )
    st.plotly_chart(fig, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("📈 Εξέλιξη top 5 γλωσσών ανά έτος")
        top5_langs = df["lang"].value_counts().head(5).index.tolist()
        lang_yr = (
            df[df["lang"].isin(top5_langs)]
            .groupby(["year", "lang"])
            .size().reset_index(name="Κριτικές")
        )
        fig = px.line(
            lang_yr, x="year", y="Κριτικές", color="lang",
            markers=True, title="Εξέλιξη γλωσσών ανά έτος",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("⭐ Βαθμολογία ανά γλώσσα (top 10)")
        lr = (
            df[df["lang"].isin(top_langs)]
            .groupby("lang")["rating"]
            .agg(mean="mean", count="count")
            .reset_index()
            .sort_values("mean", ascending=False)
        )
        lr.columns = ["Γλώσσα", "Μέση βαθμολογία", "n"]
        fig = px.bar(
            lr, x="Γλώσσα", y="Μέση βαθμολογία",
            text=lr["Μέση βαθμολογία"].round(2),
            hover_data={"n": True},
            title="Μέση βαθμολογία ανά γλώσσα",
        )
        fig.update_traces(textposition="outside")
        fig.update_yaxes(range=[0, 5.5])
        st.plotly_chart(fig, use_container_width=True)

    if "publishedPlatform" in df.columns:
        st.subheader("📱 Platform ανά γλώσσα (top 5)")
        plat = (
            df[df["lang"].isin(top5_langs)]
            .groupby(["lang", "publishedPlatform"])
            .size().reset_index(name="n")
        )
        fig = px.bar(
            plat, x="lang", y="n", color="publishedPlatform",
            barmode="stack", title="Platform αξιολόγησης ανά γλώσσα",
        )
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════
# TAB 6 — Τύπος Ταξιδιού
# ════════════════════════════════════════════════
with tab6:
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("👨‍👩‍👧 Κατανομή τύπων ταξιδιού")
        tt = df["tripType"].value_counts().reset_index()
        tt.columns = ["Τύπος", "Κριτικές"]
        fig = px.pie(
            tt, values="Κριτικές", names="Τύπος",
            color_discrete_sequence=px.colors.qualitative.Safe,
            title="Τύπος ταξιδιού — σύνολο",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("⭐ Βαθμολογία ανά τύπο ταξιδιού")
        tr = (
            df.groupby("tripType")["rating"]
            .agg(mean="mean", count="count")
            .reset_index()
            .sort_values("mean", ascending=False)
        )
        tr.columns = ["Τύπος", "Μέση βαθμολογία", "n"]
        fig = px.bar(
            tr, x="Τύπος", y="Μέση βαθμολογία",
            text=tr["Μέση βαθμολογία"].round(2),
            hover_data={"n": True}, color="Τύπος",
            title="Μέση βαθμολογία ανά τύπο ταξιδιού",
        )
        fig.update_traces(textposition="outside")
        fig.update_yaxes(range=[0, 5.5])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Τύπος ταξιδιού ανά αξιοθέατο (stacked bar)")
    ta = df.groupby(["place_name", "tripType"]).size().reset_index(name="n")
    fig = px.bar(
        ta, x="place_name", y="n", color="tripType",
        barmode="stack", title="Τύπος ταξιδιού ανά αξιοθέατο",
    )
    fig.update_xaxes(tickangle=35)
    st.plotly_chart(fig, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("📈 Εξέλιξη τύπων ταξιδιού ανά έτος")
        ty = df.groupby(["year", "tripType"]).size().reset_index(name="n")
        fig = px.line(
            ty, x="year", y="n", color="tripType",
            markers=True, title="Τύποι ταξιδιού ανά έτος",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("📏 Μέσο μήκος κριτικής ανά τύπο ταξιδιού")
        tl2 = df.groupby("tripType")["text_len"].mean().reset_index()
        tl2.columns = ["Τύπος", "Μέσο μήκος"]
        fig = px.bar(
            tl2, x="Τύπος", y="Μέσο μήκος",
            color="Τύπος", title="Μήκος κειμένου ανά τύπο",
        )
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════
# TAB 7 — Αλληλεπίδραση
# ════════════════════════════════════════════════
with tab7:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Σύνολο φωτογραφιών", f"{df['Photocount'].sum():,.0f}")
    c2.metric("Μέσες φωτ./κριτική", f"{df['Photocount'].mean():.2f}")
    c3.metric("Ποσοστό απαντήσεων", f"{100*df['has_response'].mean():.1f}%")
    c4.metric("Μέσα helpful votes", f"{df['helpfulVotes'].mean():.2f}")

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("📸 Φωτογραφίες ανά κριτική (box plot)")
        fig = px.box(
            df, x="place_name", y="Photocount",
            color="Category", color_discrete_map=CAT_COLORS,
            title="Φωτογραφίες ανά αξιοθέατο",
        )
        fig.update_xaxes(tickangle=35)
        fig.update_layout(legend_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("💬 Ποσοστό απαντήσεων ανά αξιοθέατο")
        rr = (
            df.groupby("place_name")["has_response"]
            .mean().mul(100).sort_values().reset_index()
        )
        rr.columns = ["Αξιοθέατο", "% Απαντήσεων"]
        fig = px.bar(
            rr, x="% Απαντήσεων", y="Αξιοθέατο",
            orientation="h", color="% Απαντήσεων",
            color_continuous_scale="Greens",
            title="Ποσοστό κριτικών με απάντηση διαχειριστή",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("⏱️ Καθυστέρηση δημοσίευσης (ημέρες από ταξίδι)")
        lag = df["review_lag_days"].dropna()
        lag = lag[(lag >= 0) & (lag <= 730)]
        if not lag.empty:
            fig = px.histogram(
                lag, nbins=50,
                title="Ημέρες μεταξύ ταξιδιού & δημοσίευσης (0–730)",
                labels={"value": "Ημέρες"},
                color_discrete_sequence=["#9b59b6"],
            )
            fig.add_vline(
                x=lag.median(), line_dash="dash", line_color="red",
                annotation_text=f"Διάμεσος: {lag.median():.0f}d",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Δεν υπάρχουν έγκυρα δεδομένα lag.")

    with col_r:
        st.subheader("👤 Επίπεδο συνεισφοράς χρηστών TripAdvisor")
        level_order = [
            "Νέος (1)", "Αρχάριος (2–10)", "Ενεργός (11–50)",
            "Έμπειρος (51–200)", "Πολύ Έμπειρος (200+)", "Άγνωστο",
        ]
        cl = df["contrib_level"].value_counts().reset_index()
        cl.columns = ["Επίπεδο", "Κριτικές"]
        cl["Επίπεδο"] = pd.Categorical(
            cl["Επίπεδο"], categories=level_order, ordered=True
        )
        cl = cl.sort_values("Επίπεδο")
        fig = px.bar(
            cl, x="Επίπεδο", y="Κριτικές",
            color="Επίπεδο", title="Κριτικές ανά επίπεδο χρήστη",
        )
        st.plotly_chart(fig, use_container_width=True)

    if "publishedPlatform" in df.columns:
        st.subheader("📱 Κατανομή πλατφόρμας δημοσίευσης")
        plat = df["publishedPlatform"].value_counts().reset_index()
        plat.columns = ["Platform", "Κριτικές"]
        fig = px.bar(
            plat, x="Platform", y="Κριτικές",
            text="Κριτικές", color="Platform",
            title="Κριτικές ανά πλατφόρμα",
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🏆 Top 20 χρήστες βάσει συνεισφορών")
    top_users = (
        df.groupby("user/name")
        .agg(
            κριτικές=("rating", "count"),
            μέση_βαθμολογία=("rating", "mean"),
            συνεισφορές=("contributions", "first"),
        )
        .sort_values("συνεισφορές", ascending=False)
        .head(20)
        .reset_index()
        .rename(columns={"user/name": "Χρήστης"})
    )
    st.dataframe(
        top_users.style.format({"μέση_βαθμολογία": "{:.2f}", "συνεισφορές": "{:,.0f}"}),
        use_container_width=True,
    )

# ════════════════════════════════════════════════
# TAB 8 — Χρωματική Ανάλυση
# ════════════════════════════════════════════════
with tab8:
    # Σύνδεση χρωμάτων με αξιοθέατα
    sel_places_set = set(sel_places if sel_places else all_places)

    color_joined = (
        colors_p.merge(info[["#", "id"]], on="#", how="inner")
        .merge(rev[["id", "place_name", "Category", "rating"]], on="id", how="inner")
    )
    color_joined = color_joined[color_joined["place_name"].isin(sel_places_set)]

    means_joined = (
        means_p.merge(info[["#", "id"]], on="#", how="inner")
        .merge(rev[["id", "place_name", "Category", "rating"]], on="id", how="inner")
    )
    means_joined = means_joined[means_joined["place_name"].isin(sel_places_set)]

    st.subheader("🎨 Χρωματικές παλέτες ανά αξιοθέατο")
    for pname in sorted(color_joined["place_name"].unique()):
        sub = (
            color_joined[color_joined["place_name"] == pname]
            .groupby("HEX")["pct_num"].mean()
            .reset_index()
            .sort_values("pct_num", ascending=False)
            .head(8)
        )
        if not sub.empty:
            st.markdown(f"**{pname}**")
            st.markdown(
                palette_html(sub["HEX"].tolist(), sub["pct_num"].tolist()),
                unsafe_allow_html=True,
            )
            labels = "  ".join(
                f'<span style="font-size:11px;color:{h}">■ {h} ({p:.1f}%)</span>'
                for h, p in zip(sub["HEX"], sub["pct_num"])
            )
            st.markdown(labels, unsafe_allow_html=True)
            st.markdown("")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    num_cols = [
        c for c in means_joined.columns
        if c not in ["#", "id", "place_name", "Category", "rating"]
        and pd.api.types.is_numeric_dtype(means_joined[c])
    ]

    with col_l:
        st.subheader("🔵 Κορεσμός (S%) vs Φωτεινότητα (V%)")
        s_col = next((c for c in means_joined.columns if "S%" in c), None)
        v_col = next((c for c in means_joined.columns if "V%" in c), None)
        if s_col and v_col:
            fig = px.scatter(
                means_joined, x=s_col, y=v_col,
                color="place_name", opacity=0.55,
                title="S% vs V% ανά φωτογραφία",
                labels={s_col: "Κορεσμός (%)", v_col: "Φωτεινότητα (%)"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Δεν βρέθηκαν στήλες S%/V%.")

    with col_r:
        st.subheader("🌈 Chroma (C) vs Lightness (L)")
        l_col = next((c for c in means_joined.columns if c in ["L_mean", "L"]), None)
        c_col = next((c for c in means_joined.columns if c in ["C_mean", "C"]), None)
        if l_col and c_col:
            fig = px.scatter(
                means_joined, x=l_col, y=c_col,
                color="place_name", opacity=0.55,
                title="Lightness vs Chroma (LCH)",
                labels={l_col: "Lightness", c_col: "Chroma"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Δεν βρέθηκαν στήλες L/C.")

    if num_cols:
        st.subheader("📊 Μέσα χρωματικά χαρακτηριστικά ανά αξιοθέατο")
        metric_sel = st.selectbox("Επιλέξτε μετρική χρώματος", options=num_cols)
        means_agg = means_joined.groupby("place_name")[num_cols].mean().reset_index()
        fig = px.bar(
            means_agg.sort_values(metric_sel),
            x=metric_sel, y="place_name", orientation="h",
            color=metric_sel, color_continuous_scale="Viridis",
            title=f"Μέση τιμή {metric_sel} ανά αξιοθέατο",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    if "megapixels" in info.columns:
        st.subheader("📐 Ανάλυση φωτογραφιών (megapixels) ανά αξιοθέατο")
        mp_joined = (
            info[["#", "id", "megapixels"]]
            .merge(rev[["id", "place_name"]], on="id", how="inner")
        )
        mp_joined = mp_joined[mp_joined["place_name"].isin(sel_places_set)]
        if not mp_joined.empty:
            fig = px.box(
                mp_joined, x="place_name", y="megapixels",
                title="Megapixels φωτογραφιών ανά αξιοθέατο",
            )
            fig.update_xaxes(tickangle=35)
            st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════
# TAB 9 — Συγκριτικές & Insights
# ════════════════════════════════════════════════
with tab9:
    st.subheader("🏛️ Σύγκριση κατηγοριών: Φυσικό / Πολιτιστικό / Δραστηριότητα")
    cat_agg = (
        df.groupby("Category")
        .agg(
            κριτικές=("rating", "count"),
            μέση_βαθμολογία=("rating", "mean"),
            photos=("Photocount", "sum"),
            response_rate=("has_response", "mean"),
        )
        .reset_index()
    )

    col_l, col_r = st.columns(2)
    with col_l:
        fig = px.bar(
            cat_agg, x="Category", y="μέση_βαθμολογία",
            color="Category", color_discrete_map=CAT_COLORS,
            text=cat_agg["μέση_βαθμολογία"].round(2),
            title="Μέση βαθμολογία ανά κατηγορία",
        )
        fig.update_traces(textposition="outside")
        fig.update_yaxes(range=[0, 5.5])
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig = px.scatter(
            cat_agg, x="κριτικές", y="μέση_βαθμολογία",
            size="photos", color="Category",
            color_discrete_map=CAT_COLORS, text="Category",
            title="Bubble: κριτικές vs βαθμολογία vs φωτογραφίες",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("📈 Γραμμική παλινδρόμηση βαθμολογίας vs έτος")
        dt = df.groupby("year")["rating"].mean().reset_index().dropna()
        if len(dt) > 2:
            x = dt["year"].astype(int).values
            y = dt["rating"].values
            z = np.polyfit(x, y, 1)
            dt["trend"] = np.poly1d(z)(x)
            slope_dir = "📈 Ανοδική" if z[0] > 0 else "📉 Καθοδική"
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dt["year"], y=dt["rating"],
                mode="markers+lines", name="Πραγματική",
                marker=dict(size=8),
            ))
            fig.add_trace(go.Scatter(
                x=dt["year"], y=dt["trend"],
                mode="lines", name="Τάση",
                line=dict(dash="dash", color="red"),
            ))
            fig.update_layout(
                title=f"Τάση βαθμολογίας ({slope_dir}: {z[0]:+.4f}/έτος)",
                yaxis_range=[1, 5],
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("💬 Επίδραση απάντησης διαχειριστή στη βαθμολογία")
        resp_impact = (
            df.groupby(["place_name", "has_response"])["rating"]
            .mean().unstack(fill_value=np.nan)
        )
        resp_impact.columns = ["Χωρίς απάντηση", "Με απάντηση"]
        resp_impact = resp_impact.reset_index()
        fig = px.scatter(
            resp_impact,
            x="Χωρίς απάντηση", y="Με απάντηση",
            text="place_name",
            title="Βαθμολογία: με vs χωρίς απάντηση διαχειριστή",
        )
        fig.add_shape(
            type="line", x0=1, y0=1, x1=5, y1=5,
            line=dict(dash="dash", color="grey"),
        )
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Πίνακας συσχέτισης Pearson")
    corr_cols = ["rating", "Photocount", "helpfulVotes", "text_len",
                 "has_response", "contributions"]
    available_corr = [c for c in corr_cols if c in df.columns]
    corr_data = df[available_corr].dropna()
    if len(corr_data) > 10:
        corr_matrix = corr_data.corr()
        fig = px.imshow(
            corr_matrix, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Pearson r μεταξύ βασικών μεταβλητών",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🎨 Συσχέτιση χρωματικών χαρακτηριστικών με βαθμολογία")
    if not means_joined.empty and num_cols:
        color_corr = []
        for col in num_cols:
            sub = means_joined[["rating", col]].dropna()
            if len(sub) > 10:
                r = sub["rating"].corr(sub[col])
                color_corr.append({"Χαρακτηριστικό": col, "Pearson r": round(r, 3)})
        if color_corr:
            cc_df = pd.DataFrame(color_corr).sort_values("Pearson r")
            fig = px.bar(
                cc_df, x="Pearson r", y="Χαρακτηριστικό",
                orientation="h", color="Pearson r",
                color_continuous_scale="RdBu_r",
                title="Συσχέτιση χρωμάτων με βαθμολογία",
            )
            fig.add_vline(x=0, line_color="black")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Δεν υπάρχουν διαθέσιμα χρωματικά δεδομένα για τα επιλεγμένα φίλτρα.")
