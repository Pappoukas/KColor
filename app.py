# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Kastoria Color Analytics",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a more polished look
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
# Data loading and preprocessing
# ------------------------------------------------------------
@st.cache_data
def load_summary():
    df = pd.read_csv("color_summary_batch_Summary.csv", sep=";", decimal=",")
    # Rename columns: remove trailing spaces if any
    df.columns = df.columns.str.strip()
    # Ensure numeric columns
    numeric_cols = ["R mean", "G mean", "B mean", "H° mean", "S% mean", "V% mean", "L mean", "C mean"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Color percentage columns (Color 1% ... Color 5%)
    for i in range(1, 6):
        col = f"Color {i}%"
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

@st.cache_data
def load_clusters():
    df = pd.read_csv("color_summary_batch_Clusters.csv", sep=";", decimal=",")
    df.columns = df.columns.str.strip()
    # Ensure numeric columns
    numeric_cols = ["Pixels", "%", "Red", "Green", "Blue", "Hue", "Saturation", "Value", "Lightness", "Green-Red", "Blue-Yellow"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

@st.cache_data
def load_info():
    df = pd.read_csv("color_summary_batch_Info.csv", sep=";", decimal=",")
    df.columns = df.columns.str.strip()
    # The placeInfo/name column might have trailing spaces
    if "placeInfo/name" in df.columns:
        df["placeInfo/name"] = df["placeInfo/name"].astype(str).str.strip()
    return df

@st.cache_data
def load_statistics():
    df = pd.read_csv("color_summary_batch_Statistics.csv", sep=";", decimal=",")
    df.columns = df.columns.str.strip()
    # Identify numeric columns (most columns after first)
    for col in df.columns:
        if col not in ["#", "Space", "Channel"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# Load all dataframes
df_summary = load_summary()
df_clusters = load_clusters()
df_info = load_info()
df_stats = load_statistics()

# Merge info into summary for monument names
df_summary_with_info = df_summary.merge(df_info[["#", "placeInfo/name"]], on="#", how="left")

# Prepare long format for top 5 colors per image
color_cols = []
for i in range(1, 6):
    color_cols.append({
        "hex_col": f"Color {i} HEX",
        "pct_col": f"Color {i}%",
        "name_col": f"Color {i} Name",
        "rank": i
    })

records = []
for _, row in df_summary.iterrows():
    image_id = row["#"]
    for c in color_cols:
        hex_val = row[c["hex_col"]]
        pct_val = row[c["pct_col"]]
        name_val = row[c["name_col"]]
        if pd.notna(hex_val) and pd.notna(pct_val):
            records.append({
                "#": image_id,
                "rank": c["rank"],
                "hex": hex_val,
                "percentage": pct_val,
                "color_name": name_val
            })
df_colors_long = pd.DataFrame(records)

# Merge monument info into long colors
df_colors_long = df_colors_long.merge(df_info[["#", "placeInfo/name"]], on="#", how="left")

# For overall color frequency: sum percentages per hex across all images
overall_color_freq = df_colors_long.groupby("hex").agg(
    total_percentage=("percentage", "sum"),
    color_name=("color_name", "first")
).reset_index().sort_values("total_percentage", ascending=False)

# For per-monument top colors
monument_color_freq = df_colors_long.groupby(["placeInfo/name", "hex"]).agg(
    total_percentage=("percentage", "sum"),
    color_name=("color_name", "first")
).reset_index()

# Derive dominant cluster for each image from Clusters.csv
# For each image, the cluster with highest "%"
df_clusters_dominant = df_clusters.loc[df_clusters.groupby("#")["%"].idxmax()]
df_clusters_dominant = df_clusters_dominant[["#", "Cluster"]].rename(columns={"Cluster": "dominant_cluster"})

# Merge dominant cluster into summary and stats
df_summary_with_cluster = df_summary_with_info.merge(df_clusters_dominant, on="#", how="left")
df_stats_with_cluster = df_stats.merge(df_clusters_dominant, on="#", how="left")
df_stats_with_cluster = df_stats_with_cluster.merge(df_info[["#", "placeInfo/name"]], on="#", how="left")

# ------------------------------------------------------------
# Sidebar filters
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
        unique_colors = df_colors_long["hex"].nunique()
        st.metric("Unique Colors (Top-5)", unique_colors)
    with col4:
        avg_saturation = df_stats[df_stats["Channel"] == "Saturation"]["Mean"].values[0] if not df_stats[df_stats["Channel"] == "Saturation"].empty else 0
        st.metric("Avg Image Saturation", f"{avg_saturation:.1f}%")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 🎯 Most Frequent Colors (Overall)")
        top10 = overall_color_freq.head(10)
        fig = px.bar(top10, x="total_percentage", y="hex", orientation='h',
                     color="total_percentage", color_continuous_scale="Viridis",
                     title="Top 10 Colors by Aggregated Area %",
                     labels={"total_percentage": "Total % across images", "hex": "Color (HEX)"})
        fig.update_layout(height=500, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
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
    # sample palette of top 12 colors
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
        top_n = st.slider("Number of top colors to show", 5, 20, 10)
        data = overall_color_freq.head(top_n).copy()
        fig = px.bar(data, x="total_percentage", y="hex", orientation='h',
                     color="total_percentage", color_continuous_scale="Plasma",
                     title=f"Top {top_n} Colors (Aggregated % across all images)",
                     labels={"total_percentage": "Total percentage", "hex": "Color"})
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    else:  # per monument
        monuments = sorted(df_info["placeInfo/name"].unique())
        selected_monument = st.selectbox("Choose a monument", monuments)
        # Get top 5 colors for that monument
        monument_data = monument_color_freq[monument_color_freq["placeInfo/name"] == selected_monument]
        top5 = monument_data.sort_values("total_percentage", ascending=False).head(5)
        if not top5.empty:
            fig = px.bar(top5, x="total_percentage", y="hex", orientation='h',
                         color="total_percentage", color_continuous_scale="Magma",
                         title=f"Top 5 Colors for {selected_monument}",
                         labels={"total_percentage": "Aggregated %", "hex": "Color"})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No color data found for this monument.")

    # Additional insight: Show a sample of the color palette for the selected monument
    if freq_option == "Top 5 per Monument" and "selected_monument" in locals():
        st.markdown("#### 🎨 Color Palette")
        cols = st.columns(5)
        for idx, (_, row) in enumerate(top5.iterrows()):
            with cols[idx]:
                st.markdown(f"<div style='background-color:{row['hex']}; height:80px; border-radius:12px;'></div>", unsafe_allow_html=True)
                st.caption(f"{row['hex']}<br>{row['total_percentage']:.1f}%", unsafe_allow_html=True)

# ------------------------------------------------------------
# Page: Dominant Colors Pie Chart (per image)
# ------------------------------------------------------------
elif page == "🥧 Dominant Colors (Pie)":
    st.markdown('<div class="main-header">Dominant Color Composition per Image</div>', unsafe_allow_html=True)
    # Filter images by monument first
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
        # Get colors for that image
        img_colors = df_colors_long[df_colors_long["#"] == selected_id].sort_values("rank")
        if not img_colors.empty:
            fig = px.pie(img_colors, values="percentage", names="hex", title=f"Image #{selected_id} - Color Distribution",
                         color="hex", hole=0.4)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

            # Show additional info about the image
            img_info = df_info[df_info["#"] == selected_id].iloc[0]
            st.markdown(f"**Monument:** {img_info['placeInfo/name']}  |  **Resolution:** {img_info['W×H']}")
            # Show the actual image URL if needed (optional)
            if "URL" in img_info and pd.notna(img_info["URL"]):
                st.markdown(f"[View original image on TripAdvisor]({img_info['URL']})")
        else:
            st.warning("No color data for this image.")

# ------------------------------------------------------------
# Page: Heatmap images vs color intensity
# ------------------------------------------------------------
elif page == "🔥 Intensity Heatmap":
    st.markdown('<div class="main-header">Color Intensity Matrix: Images vs Metrics</div>', unsafe_allow_html=True)
    # Prepare data: for each image, get mean Saturation, Chroma, Lightness, Value
    # We'll take from Statistics.csv where Channel is one of these
    stats_pivot = df_stats[df_stats["Channel"].isin(["Saturation", "Value", "Lightness", "Chroma"])]
    heatmap_data = stats_pivot.pivot(index="#", columns="Channel", values="Mean").reset_index()
    # Merge monument info for optional grouping
    heatmap_data = heatmap_data.merge(df_info[["#", "placeInfo/name"]], on="#", how="left")
    # Normalize each metric for better heatmap contrast
    metrics = ["Saturation", "Value", "Lightness", "Chroma"]
    for m in metrics:
        if m in heatmap_data.columns:
            heatmap_data[f"{m}_norm"] = (heatmap_data[m] - heatmap_data[m].min()) / (heatmap_data[m].max() - heatmap_data[m].min())
    # Use normalized columns for heatmap
    norm_cols = [f"{m}_norm" for m in metrics]
    heatmap_display = heatmap_data.set_index("#")[norm_cols]
    heatmap_display.columns = metrics  # rename back for display
    fig = px.imshow(heatmap_display.T, aspect="auto", text_auto=True, color_continuous_scale="Blues",
                    title="Color Intensity per Image (normalized per metric)",
                    labels={"x": "Image ID", "y": "Metric", "color": "Normalized Intensity"})
    fig.update_layout(height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Interpretation")
    st.caption("Higher normalized values indicate stronger intensity (e.g., high saturation, high chroma). Each metric is scaled independently for visualization.")

# ------------------------------------------------------------
# Page: Scatter plot brightness vs saturation
# ------------------------------------------------------------
elif page == "📈 Brightness vs Saturation":
    st.markdown('<div class="main-header">Brightness vs Saturation</div>', unsafe_allow_html=True)
    # Get mean saturation and value per image from Statistics
    sat_df = df_stats[df_stats["Channel"] == "Saturation"][["#", "Mean"]].rename(columns={"Mean": "Saturation_mean"})
    val_df = df_stats[df_stats["Channel"] == "Value"][["#", "Mean"]].rename(columns={"Mean": "Value_mean"})
    scatter_data = sat_df.merge(val_df, on="#", how="inner")
    scatter_data = scatter_data.merge(df_info[["#", "placeInfo/name"]], on="#", how="left")
    # Add dominant cluster if available
    scatter_data = scatter_data.merge(df_clusters_dominant, on="#", how="left")

    fig = px.scatter(scatter_data, x="Saturation_mean", y="Value_mean", color="placeInfo/name",
                     hover_data=["#", "dominant_cluster"], size_max=15,
                     title="Per‑Image Average Saturation vs Brightness (Value)",
                     labels={"Saturation_mean": "Average Saturation (%)", "Value_mean": "Average Brightness (Value %)"},
                     trendline="ols")
    fig.update_layout(legend_title_text="Monument")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Observations")
    st.caption("Points near the top‑right are vivid and bright; bottom‑left indicates dark and muted images. Trendline shows general correlation across all images.")

# ------------------------------------------------------------
# Page: Cluster Visualization
# ------------------------------------------------------------
elif page == "🔮 Cluster Explorer":
    st.markdown('<div class="main-header">Color Profile Clusters</div>', unsafe_allow_html=True)
    st.markdown("Images grouped by dominant color profile using K‑means clustering (pre‑computed).")

    # Show cluster sizes
    cluster_counts = df_clusters_dominant["dominant_cluster"].value_counts().sort_index()
    fig_counts = px.bar(x=cluster_counts.index, y=cluster_counts.values, color=cluster_counts.index.astype(str),
                        title="Number of Images per Dominant Cluster",
                        labels={"x": "Cluster", "y": "Count"})
    st.plotly_chart(fig_counts, use_container_width=True)

    # PCA on color features
    # Build feature matrix: mean R, G, B, Saturation, Value, Lightness, Chroma from Statistics
    # Pivot statistics to get each channel as column
    stats_wide = df_stats.pivot(index="#", columns="Channel", values="Mean").reset_index()
    feature_cols = ["Red", "Green", "Blue", "Saturation", "Value", "Lightness", "Chroma"]
    available_features = [c for c in feature_cols if c in stats_wide.columns]
    features = stats_wide[["#"] + available_features].dropna()
    # Merge dominant cluster
    features = features.merge(df_clusters_dominant, on="#", how="inner")
    features = features.merge(df_info[["#", "placeInfo/name"]], on="#", how="left")
    # Standardize and PCA
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

    # Show cluster characteristics: top colors per cluster
    st.markdown("### 🎨 Cluster Color Signatures")
    # For each cluster, find most frequent HEX among dominant images
    cluster_colors = []
    for cluster_id in sorted(features["dominant_cluster"].unique()):
        images_in_cluster = features[features["dominant_cluster"] == cluster_id]["#"].tolist()
        # Get top colors from these images (from long format)
        top_colors_in_cluster = df_colors_long[df_colors_long["#"].isin(images_in_cluster)]
        cluster_top = top_colors_in_cluster.groupby("hex").agg(total_pct=("percentage", "sum")).reset_index()
        cluster_top = cluster_top.sort_values("total_pct", ascending=False).head(4)
        cluster_colors.append({
            "cluster": cluster_id,
            "colors": cluster_top["hex"].tolist(),
            "count": len(images_in_cluster)
        })

    cols = st.columns(len(cluster_colors))
    for idx, cc in enumerate(cluster_colors):
        with cols[idx]:
            st.markdown(f"**Cluster {cc['cluster']}** ({cc['count']} images)")
            for c in cc["colors"]:
                st.markdown(f"<div style='background-color:{c}; height:30px; border-radius:6px; margin:4px;'></div>", unsafe_allow_html=True)
                st.caption(c)
    st.caption("Above: Most frequent HEX colors among images belonging to each cluster.")

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit • Color analysis from TripAdvisor photos")
