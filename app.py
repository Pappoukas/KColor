# app.py (πλήρως διορθωμένο)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import re

# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Kastoria Color Analytics",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4A627A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #F8F9FA;
        border-radius: 0.75rem;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stPlotlyChart {
        background-color: #FFFFFF;
        border-radius: 0.75rem;
        padding: 0.5rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Data loading with robust handling
# ------------------------------------------------------------
@st.cache_data
def load_summary():
    df = pd.read_csv("color_summary_batch_Summary.csv", sep=";", encoding="utf-8")
    df.columns = df.columns.str.strip()
    
    # Convert all percentage columns (Color X %) to numeric, handling comma decimal
    for col in df.columns:
        if re.search(r"Color\s+\d+\s*%", col):
            # Replace comma with dot, then convert to float
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

@st.cache_data
def get_color_columns(df):
    """Return list of dicts with hex, pct, name columns for ranks 1-5."""
    hex_cols = {}
    pct_cols = {}
    name_cols = {}
    for col in df.columns:
        # Match "Color" followed by a number, optional spaces, then "HEX"
        match_hex = re.match(r"Color\s*(\d+)\s*HEX", col, re.IGNORECASE)
        if match_hex:
            rank = int(match_hex.group(1))
            hex_cols[rank] = col
        # Match "Color" + number + optional spaces + "%"
        match_pct = re.match(r"Color\s*(\d+)\s*%", col, re.IGNORECASE)
        if match_pct:
            rank = int(match_pct.group(1))
            pct_cols[rank] = col
        # Match "Color" + number + optional spaces + "Name"
        match_name = re.match(r"Color\s*(\d+)\s*Name", col, re.IGNORECASE)
        if match_name:
            rank = int(match_name.group(1))
            name_cols[rank] = col
    
    color_cols = []
    for rank in range(1, 6):
        if rank in hex_cols and rank in pct_cols and rank in name_cols:
            color_cols.append({
                "rank": rank,
                "hex_col": hex_cols[rank],
                "pct_col": pct_cols[rank],
                "name_col": name_cols[rank]
            })
    return color_cols

@st.cache_data
def load_clusters():
    df = pd.read_csv("color_summary_batch_Clusters.csv", sep=";", encoding="utf-8")
    df.columns = df.columns.str.strip()
    # Convert percentage column
    if "%" in df.columns:
        df["%"] = df["%"].astype(str).str.replace(",", ".").astype(float)
    numeric_cols = ["Pixels", "%", "Red", "Green", "Blue", "Hue", "Saturation", "Value", "Lightness", "Green-Red", "Blue-Yellow"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

@st.cache_data
def load_info():
    df = pd.read_csv("color_summary_batch_Info.csv", sep=";", encoding="utf-8")
    df.columns = df.columns.str.strip()
    if "placeInfo/name" in df.columns:
        df["placeInfo/name"] = df["placeInfo/name"].astype(str).str.strip()
    return df

@st.cache_data
def load_statistics():
    df = pd.read_csv("color_summary_batch_Statistics.csv", sep=";", encoding="utf-8")
    df.columns = df.columns.str.strip()
    for col in df.columns:
        if col not in ["#", "Space", "Channel"]:
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# Load data
df_summary = load_summary()
df_clusters = load_clusters()
df_info = load_info()
df_stats = load_statistics()

# Detect color columns
color_columns = get_color_columns(df_summary)

# Build long format for top 5 colors
records = []
for _, row in df_summary.iterrows():
    image_id = row["#"]
    for col_info in color_columns:
        hex_val = row[col_info["hex_col"]]
        pct_val = row[col_info["pct_col"]]
        name_val = row[col_info["name_col"]]
        if pd.notna(hex_val) and pd.notna(pct_val):
            records.append({
                "#": image_id,
                "rank": col_info["rank"],
                "hex": hex_val,
                "percentage": pct_val,
                "color_name": name_val
            })

df_colors_long = pd.DataFrame(records)

# Ensure percentage is numeric (already, but double-check)
df_colors_long["percentage"] = pd.to_numeric(df_colors_long["percentage"], errors="coerce")
df_colors_long = df_colors_long.dropna(subset=["percentage"])

# Merge monument info
df_colors_long = df_colors_long.merge(df_info[["#", "placeInfo/name"]], on="#", how="left")

# Overall color frequency
if not df_colors_long.empty:
    overall_color_freq = df_colors_long.groupby("hex").agg(
        total_percentage=("percentage", "sum"),
        color_name=("color_name", "first")
    ).reset_index().sort_values("total_percentage", ascending=False)
else:
    overall_color_freq = pd.DataFrame(columns=["hex", "total_percentage", "color_name"])

# Per-monument color frequency
if not df_colors_long.empty:
    monument_color_freq = df_colors_long.groupby(["placeInfo/name", "hex"]).agg(
        total_percentage=("percentage", "sum"),
        color_name=("color_name", "first")
    ).reset_index()
else:
    monument_color_freq = pd.DataFrame(columns=["placeInfo/name", "hex", "total_percentage", "color_name"])

# Dominant cluster per image (highest %)
df_clusters["%"] = pd.to_numeric(df_clusters["%"], errors="coerce")
df_clusters_dominant = df_clusters.loc[df_clusters.groupby("#")["%"].idxmax()]
df_clusters_dominant = df_clusters_dominant[["#", "Cluster"]].rename(columns={"Cluster": "dominant_cluster"})

# Merge into summary and stats
df_summary_with_info = df_summary.merge(df_info[["#", "placeInfo/name"]], on="#", how="left")
df_summary_with_cluster = df_summary_with_info.merge(df_clusters_dominant, on="#", how="left")
df_stats_with_cluster = df_stats.merge(df_clusters_dominant, on="#", how="left")
df_stats_with_cluster = df_stats_with_cluster.merge(df_info[["#", "placeInfo/name"]], on="#", how="left")

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.image("https://via.placeholder.com/150x80?text=Kastoria+Color", use_column_width=True)
st.sidebar.title("🎨 Navigation")
page = st.sidebar.radio(
    "Select a view",
    ["🏠 Overview", "📊 Color Frequency", "🥧 Dominant Colors (Pie)", "🔥 Intensity Heatmap", "📈 Brightness vs Saturation", "🔮 Cluster Explorer"]
)
st.sidebar.markdown("---")
st.sidebar.caption("Data source: TripAdvisor visitor photos from Kastoria, Greece")
st.sidebar.caption("Color analysis performed on 100 images")

# ------------------------------------------------------------
# Page: Overview
# ------------------------------------------------------------
if page == "🏠 Overview":
    st.markdown('<div class="main-header">Kastoria Color Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Understanding the chromatic signature of visitor experiences</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Images", len(df_summary))
    with col2:
        unique_monuments = df_info["placeInfo/name"].nunique()
        st.metric("Monuments / Attractions", unique_monuments)
    with col3:
        unique_colors = df_colors_long["hex"].nunique() if not df_colors_long.empty else 0
        st.metric("Unique Colors (Top-5)", unique_colors)
    with col4:
        avg_saturation = df_stats[df_stats["Channel"] == "Saturation"]["Mean"].values[0] if not df_stats[df_stats["Channel"] == "Saturation"].empty else 0
        st.metric("Avg Image Saturation", f"{avg_saturation:.1f}%")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 🎯 Most Frequent Colors (Overall)")
        if not overall_color_freq.empty:
            top10 = overall_color_freq.head(10)
            fig = px.bar(top10, x="total_percentage", y="hex", orientation='h',
                         color="total_percentage", color_continuous_scale="Viridis",
                         title="Top 10 Colors by Aggregated Area %",
                         labels={"total_percentage": "Total % across images", "hex": "Color (HEX)"})
            fig.update_layout(height=500, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No color data available.")
    with col_b:
        st.markdown("### 🏛️ Monuments Overview")
        monument_counts = df_info["placeInfo/name"].value_counts().reset_index()
        monument_counts.columns = ["Monument", "Number of Images"]
        fig2 = px.bar(monument_counts, x="Monument", y="Number of Images", color="Monument",
                      title="Images per Monument", text_auto=True)
        fig2.update_layout(showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🌈 Color Diversity Snapshot")
    if not overall_color_freq.empty:
        top12 = overall_color_freq.head(12)
        color_swatches = []
        for _, row in top12.iterrows():
            color_swatches.append(f"<div style='background-color:{row['hex']}; width:60px; height:60px; border-radius:8px; margin:4px; display:inline-block;'></div>")
        st.markdown("".join(color_swatches), unsafe_allow_html=True)

# ------------------------------------------------------------
# Page: Color Frequency Bar Chart
# ------------------------------------------------------------
elif page == "📊 Color Frequency":
    st.markdown('<div class="main-header">Color Frequency Analysis</div>', unsafe_allow_html=True)
    freq_option = st.radio("Select scope", ["Overall Top 10 Colors", "Top 5 per Monument"], horizontal=True)

    if freq_option == "Overall Top 10 Colors":
        if not overall_color_freq.empty:
            top_n = st.slider("Number of top colors to show", 5, 20, 10)
            data = overall_color_freq.head(top_n).copy()
            fig = px.bar(data, x="total_percentage", y="hex", orientation='h',
                         color="total_percentage", color_continuous_scale="Plasma",
                         title=f"Top {top_n} Colors (Aggregated % across all images)",
                         labels={"total_percentage": "Total percentage", "hex": "Color"})
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No color frequency data available.")

    else:  # per monument
        monuments = sorted(df_info["placeInfo/name"].unique())
        selected_monument = st.selectbox("Choose a monument", monuments)
        monument_data = monument_color_freq[monument_color_freq["placeInfo/name"] == selected_monument]
        top5 = monument_data.sort_values("total_percentage", ascending=False).head(5)
        if not top5.empty:
            fig = px.bar(top5, x="total_percentage", y="hex", orientation='h',
                         color="total_percentage", color_continuous_scale="Magma",
                         title=f"Top 5 Colors for {selected_monument}",
                         labels={"total_percentage": "Aggregated %", "hex": "Color"})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### 🎨 Color Palette")
            cols = st.columns(5)
            for idx, (_, row) in enumerate(top5.iterrows()):
                with cols[idx]:
                    st.markdown(f"<div style='background-color:{row['hex']}; height:80px; border-radius:12px;'></div>", unsafe_allow_html=True)
                    st.caption(f"{row['hex']}<br>{row['total_percentage']:.1f}%", unsafe_allow_html=True)
        else:
            st.warning("No color data found for this monument.")

# ------------------------------------------------------------
# Page: Dominant Colors Pie Chart
# ------------------------------------------------------------
elif page == "🥧 Dominant Colors (Pie)":
    st.markdown('<div class="main-header">Dominant Color Composition per Image</div>', unsafe_allow_html=True)
    monuments = sorted(df_info["placeInfo/name"].unique())
    selected_monument = st.selectbox("Filter by monument", ["All"] + monuments)
    if selected_monument == "All":
        filtered_images = df_summary_with_info
    else:
        filtered_images = df_summary_with_info[df_summary_with_info["placeInfo/name"] == selected_monument]

    image_options = filtered_images["#"].tolist()
    if not image_options:
        st.warning("No images found for this monument.")
    else:
        selected_id = st.selectbox("Select image ID", image_options, format_func=lambda x: f"Image #{x}")
        img_colors = df_colors_long[df_colors_long["#"] == selected_id].sort_values("rank")
        if not img_colors.empty:
            fig = px.pie(img_colors, values="percentage", names="hex", title=f"Image #{selected_id} - Color Distribution",
                         color="hex", hole=0.4)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

            img_info = df_info[df_info["#"] == selected_id].iloc[0]
            st.markdown(f"**Monument:** {img_info['placeInfo/name']}  |  **Resolution:** {img_info['W×H']}")
            if "URL" in img_info and pd.notna(img_info["URL"]):
                st.markdown(f"[View original image on TripAdvisor]({img_info['URL']})")
        else:
            st.warning("No color data for this image.")

# ------------------------------------------------------------
# Page: Heatmap images vs color intensity
# ------------------------------------------------------------
elif page == "🔥 Intensity Heatmap":
    st.markdown('<div class="main-header">Color Intensity Matrix: Images vs Metrics</div>', unsafe_allow_html=True)
    stats_pivot = df_stats[df_stats["Channel"].isin(["Saturation", "Value", "Lightness", "Chroma"])]
    if not stats_pivot.empty:
        heatmap_data = stats_pivot.pivot(index="#", columns="Channel", values="Mean").reset_index()
        heatmap_data = heatmap_data.merge(df_info[["#", "placeInfo/name"]], on="#", how="left")
        metrics = ["Saturation", "Value", "Lightness", "Chroma"]
        for m in metrics:
            if m in heatmap_data.columns:
                min_val = heatmap_data[m].min()
                max_val = heatmap_data[m].max()
                if max_val > min_val:
                    heatmap_data[f"{m}_norm"] = (heatmap_data[m] - min_val) / (max_val - min_val)
                else:
                    heatmap_data[f"{m}_norm"] = 0
        norm_cols = [f"{m}_norm" for m in metrics if f"{m}_norm" in heatmap_data.columns]
        if norm_cols:
            heatmap_display = heatmap_data.set_index("#")[norm_cols]
            heatmap_display.columns = [c.replace("_norm", "") for c in norm_cols]
            fig = px.imshow(heatmap_display.T, aspect="auto", text_auto=True, color_continuous_scale="Blues",
                            title="Color Intensity per Image (normalized per metric)",
                            labels={"x": "Image ID", "y": "Metric", "color": "Normalized Intensity"})
            fig.update_layout(height=500, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Higher normalized values indicate stronger intensity (e.g., high saturation, high chroma). Each metric is scaled independently.")
        else:
            st.warning("Required metrics not found in data.")
    else:
        st.warning("No intensity data available.")

# ------------------------------------------------------------
# Page: Scatter plot brightness vs saturation
# ------------------------------------------------------------
elif page == "📈 Brightness vs Saturation":
    st.markdown('<div class="main-header">Brightness vs Saturation</div>', unsafe_allow_html=True)
    sat_df = df_stats[df_stats["Channel"] == "Saturation"][["#", "Mean"]].rename(columns={"Mean": "Saturation_mean"})
    val_df = df_stats[df_stats["Channel"] == "Value"][["#", "Mean"]].rename(columns={"Mean": "Value_mean"})
    if not sat_df.empty and not val_df.empty:
        scatter_data = sat_df.merge(val_df, on="#", how="inner")
        scatter_data = scatter_data.merge(df_info[["#", "placeInfo/name"]], on="#", how="left")
        scatter_data = scatter_data.merge(df_clusters_dominant, on="#", how="left")

        fig = px.scatter(scatter_data, x="Saturation_mean", y="Value_mean", color="placeInfo/name",
                         hover_data=["#", "dominant_cluster"], size_max=15,
                         title="Per‑Image Average Saturation vs Brightness (Value)",
                         labels={"Saturation_mean": "Average Saturation (%)", "Value_mean": "Average Brightness (Value %)"},
                         trendline="ols")
        fig.update_layout(legend_title_text="Monument")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Points near the top‑right are vivid and bright; bottom‑left indicates dark and muted images. Trendline shows general correlation.")
    else:
        st.warning("Insufficient data for brightness vs saturation plot.")

# ------------------------------------------------------------
# Page: Cluster Visualization
# ------------------------------------------------------------
elif page == "🔮 Cluster Explorer":
    st.markdown('<div class="main-header">Color Profile Clusters</div>', unsafe_allow_html=True)
    st.markdown("Images grouped by dominant color profile using K‑means clustering (pre‑computed).")

    if not df_clusters_dominant.empty:
        cluster_counts = df_clusters_dominant["dominant_cluster"].value_counts().sort_index()
        fig_counts = px.bar(x=cluster_counts.index, y=cluster_counts.values, color=cluster_counts.index.astype(str),
                            title="Number of Images per Dominant Cluster",
                            labels={"x": "Cluster", "y": "Count"})
        st.plotly_chart(fig_counts, use_container_width=True)

        # PCA on color features - remove duplicates before pivot
        df_stats_unique = df_stats.drop_duplicates(subset=["#", "Channel"])
        stats_wide = df_stats_unique.pivot(index="#", columns="Channel", values="Mean").reset_index()
        
        feature_cols = ["Red", "Green", "Blue", "Saturation", "Value", "Lightness", "Chroma"]
        available_features = [c for c in feature_cols if c in stats_wide.columns]
        features = stats_wide[["#"] + available_features].dropna()
        features = features.merge(df_clusters_dominant, on="#", how="inner")
        features = features.merge(df_info[["#", "placeInfo/name"]], on="#", how="left")

        if len(features) > 1:
            X = features[available_features]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(X_scaled)
            features["PC1"] = pca_result[:, 0]
            features["PC2"] = pca_result[:, 1]

            fig_pca = px.scatter(features, x="PC1", y="PC2", color="dominant_cluster",
                                 hover_data=["#", "placeInfo/name"],
                                 title="2D PCA Projection of Color Features (colored by dominant cluster)",
                                 color_continuous_scale="Turbo")
            fig_pca.update_layout(legend_title_text="Cluster")
            st.plotly_chart(fig_pca, use_container_width=True)
        else:
            st.info("Not enough data for PCA (need at least 2 images with complete features).")

        st.markdown("### 🎨 Cluster Color Signatures")
        cluster_colors = []
        for cluster_id in sorted(features["dominant_cluster"].unique()):
            images_in_cluster = features[features["dominant_cluster"] == cluster_id]["#"].tolist()
            top_colors_in_cluster = df_colors_long[df_colors_long["#"].isin(images_in_cluster)]
            if not top_colors_in_cluster.empty:
                cluster_top = top_colors_in_cluster.groupby("hex").agg(total_pct=("percentage", "sum")).reset_index()
                cluster_top = cluster_top.sort_values("total_pct", ascending=False).head(4)
                cluster_colors.append({
                    "cluster": cluster_id,
                    "colors": cluster_top["hex"].tolist(),
                    "count": len(images_in_cluster)
                })

        if cluster_colors:
            cols = st.columns(len(cluster_colors))
            for idx, cc in enumerate(cluster_colors):
                with cols[idx]:
                    st.markdown(f"**Cluster {cc['cluster']}** ({cc['count']} images)")
                    for c in cc["colors"]:
                        st.markdown(f"<div style='background-color:{c}; height:30px; border-radius:6px; margin:4px;'></div>", unsafe_allow_html=True)
                        st.caption(c)
            st.caption("Above: Most frequent HEX colors among images belonging to each cluster.")
        else:
            st.info("No cluster color signatures could be derived.")
    else:
        st.warning("No cluster data available.")
