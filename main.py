import pathlib
import pandas as pd
import plotly.express as px
import streamlit as st
from src.transform import Transform
from src.eda import RetailEDA
from src.segmenter import Segmenter
from src.models import ClusteringPipeline



st.set_page_config(
    page_title="Online Retail Customer Analytics & Segmentation",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

css_path = pathlib.Path("assets/style.css")
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_data
def compute_rfm_pipeline(filtered_reg: pd.DataFrame):
    segmenter = Segmenter(filtered_reg)
    segmenter.buildRFM()
    segmenter.computeRFM()
    segmenter.applySegmentationMapping()
    scaled_matrix = segmenter.preprocessor()

    return segmenter.rfm_df, scaled_matrix


@st.cache_data
def load_and_clean_data():
    csv_path = pathlib.Path("data/OnlineRetail.csv")
    if not csv_path.exists():
        csv_path = pathlib.Path("../data/OnlineRetail.csv")

    df_raw = pd.read_csv(csv_path)
    transformer = Transform(df_raw)
    transformer.clean()
    return transformer.transform_all()


@st.cache_data
def compute_country_aggregates(df_v_encoded):
    country_summary = df_v_encoded.groupby('Country', as_index=False).agg(
        Total_Revenue=('TotalLineRevenue', 'sum'),
        Order_Count=('InvoiceNo', 'nunique')
    )

    iso_mapping = {
        'United Kingdom': 'GBR', 'Germany': 'DEU', 'France': 'FRA', 'EIRE': 'IRL',
        'Spain': 'ESP', 'Netherlands': 'NLD', 'Belgium': 'BEL', 'Switzerland': 'CHE',
        'Portugal': 'PRT', 'Australia': 'AUS', 'Norway': 'NOR', 'Italy': 'ITA',
        'Channel Islands': 'GBR', 'Finland': 'FIN', 'Cyprus': 'CYP', 'Sweden': 'SWE',
        'Austria': 'AUT', 'Denmark': 'DNK', 'Japan': 'JPN', 'Poland': 'POL',
        'USA': 'USA', 'Israel': 'ISR', 'Unspecified': 'ATA', 'Greece': 'GRC',
        'Iceland': 'ISL', 'Canada': 'CAN', 'Malta': 'MLT', 'Bermuda': 'BMU',
        'Singapore': 'SGP', 'Bahrain': 'BHR', 'Thailand': 'THA', 'RSA': 'ZAF',
        'Lebanon': 'LBN', 'Lithuania': 'LTU', 'Brazil': 'BRA', 'Czech Republic': 'CZE',
        'European Union': 'FRA', 'United Arab Emirates': 'ARE', 'Saudi Arabia': 'SAU'
    }
    country_summary['ISO'] = country_summary['Country'].map(iso_mapping)
    return country_summary.dropna(subset=['ISO'])


def renderGlobe(df_v):
    country_summary = compute_country_aggregates(df_v)

    fig = px.choropleth(
        country_summary,
        locations="ISO",
        color="Total_Revenue",
        hover_name="Country",
        color_continuous_scale="Teal",
        projection="orthographic",
        title="<b>Global Revenue & Order Density Globe</b>"
    )

    fig.update_geos(
        showland=True,
        landcolor="rgb(242, 245, 248)",
        showcountries=True,
        countrycolor="rgb(205, 212, 220)",
        showocean=True,
        oceancolor="rgb(250, 252, 255)",
        showlakes=False,
        projection_rotation=dict(lon=10, lat=30, roll=0)
    )

    fig.update_layout(
        template='plotly_white',
        height=520,
        margin=dict(t=40, b=0, l=0, r=0),
        coloraxis_colorbar=dict(title="Revenue ($)", thickness=15, len=0.6)
    )
    return fig


def run_stapp():
    with st.spinner("🚀 Initializing high-performance retail engine..."):
        datasets = load_and_clean_data()
        df_reg = datasets["df_reg"]
        df_cl = datasets["df_cl"]
        df_v = datasets["df_v"]

    # Sidebar Dashboard Controls
    st.sidebar.header("🎛️ Dashboard Controls")

    market_scope = st.sidebar.radio(
        "Market Scope Filter",
        ["All Global Markets", "United Kingdom Only", "Rest of World (International)"]
    )

    if market_scope == "United Kingdom Only":
        df_v_filtered = df_v[df_v['Country'] == 'United Kingdom'].copy()
        df_reg_filtered = df_reg[df_reg['Country'] == 'United Kingdom'].copy()
        df_cl_filtered = df_cl[df_cl['Country'] == 'United Kingdom'].copy()
    elif market_scope == "Rest of World (International)":
        df_v_filtered = df_v[df_v['Country'] != 'United Kingdom'].copy()
        df_reg_filtered = df_reg[df_reg['Country'] != 'United Kingdom'].copy()
        df_cl_filtered = df_cl[df_cl['Country'] != 'United Kingdom'].copy()
    else:
        df_v_filtered = df_v
        df_reg_filtered = df_reg.copy()
        df_cl_filtered = df_cl.copy()

    countries = ['All'] + sorted(df_v_filtered['Country'].dropna().unique().tolist())
    selected_country = st.sidebar.selectbox("Drill-down by Specific Country", countries)

    if selected_country != 'All':
        filtered_reg = df_reg_filtered[df_reg_filtered['Country'] == selected_country].copy()
        filtered_cl = df_cl_filtered[df_cl_filtered['Country'] == selected_country].copy()
        filtered_v = df_v_filtered[df_v_filtered['Country'] == selected_country].copy()
    else:
        filtered_reg = df_reg_filtered
        filtered_cl = df_cl_filtered
        filtered_v = df_v_filtered

    eda_macro = RetailEDA(filtered_v)
    eda_canc = RetailEDA(filtered_cl)

    st.title("🛍️ Online Retail Customer Analytics & Segmentation")
    st.markdown(
        "Executive intelligence platform analyzing macroeconomic trends, product concentration, returns, and RFM customer lifetime value."
    )
    st.markdown("---")

    # Top KPI Metrics Bar
    col1, col2, col3, col4 = st.columns(4)
    total_rev = filtered_v['TotalLineRevenue'].sum()
    total_orders = filtered_v['InvoiceNo'].nunique()
    total_customers = filtered_reg['CustomerID'].nunique()
    total_returns = len(filtered_cl)

    with col1:
        st.metric(label="💰 Total Store Net Revenue", value=f"${total_rev:,.2f}")
    with col2:
        st.metric(label="📦 Store Unique Orders", value=f"{total_orders:,}")
    with col3:
        st.metric(label="👥 Registered Active Customers", value=f"{total_customers:,}")
    with col4:
        st.metric(label="↩️ Total Cancellations", value=f"{total_returns:,}")

    st.markdown("---")

    # Interactive Globe View Section
    with st.expander("🌍 Interactive 3D Global Revenue Distribution Globe", expanded=True):
        st.plotly_chart(renderGlobe(filtered_v), use_container_width=True)

    st.markdown("---")

    tab_macro, tab_products, tab_returns, tab_rfm = st.tabs([
        "📈 Macro Trends & EDA",
        "🏆 Product Concentration & Traffic",
        "⚠️ Returns & Cancellations",
        "🎯 RFM Segmentation & AI Clustering"
    ])

    with tab_macro:
        st.subheader("Macroeconomic Performance & Seasonality")
        col_a, col_b = st.columns([2, 1])
        with col_a:
            fig_monthly = eda_macro.monthlyRevenue()
            fig_monthly.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_monthly, use_container_width=True)
        with col_b:
            st.markdown("### 💡 Executive Takeaway")
            st.info(
                "Revenue displays a massive hockey-stick surge heading into Q4 (October–December), peaking heavily in November ahead of the holiday shutdown."
            )

    with tab_products:
        st.subheader("Product Revenue Concentration (80/20 Pareto Rule)")
        fig_pareto = eda_macro.paretoRevenue()
        fig_pareto.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pareto, use_container_width=True)

        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown("#### 🕒 Hourly Traffic & Cadence")
            fig_traffic, fig_cadence = eda_macro.analyze_hourly_traffic()
            fig_traffic.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_traffic, use_container_width=True)
        with col_d:
            st.markdown("#### 🛒 Average Order Value (AOV)")
            fig_aov = eda_macro.average_order_values()
            fig_aov.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_aov, use_container_width=True)

        st.plotly_chart(fig_cadence, use_container_width=True)

    with tab_returns:
        st.subheader("Product Returns & Cancellation Impact Analysis")
        if len(filtered_cl) > 0:
            fig_canc, canc_stats = eda_canc.cancellations()
            fig_canc.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_canc, use_container_width=True)

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.metric(label="Total Cancelled Transactions", value=f"{canc_stats['cancelled']:,}")
            with col_r2:
                st.metric(label="Estimated Revenue Lost", value=f"${canc_stats['revenue_lost']:,.2f}")
        else:
            st.info(f"No cancellation records found for selection.")

    with tab_rfm:
        st.subheader("Behavioral RFM Analysis & Multi-Model Clustering")

        segmenter = Segmenter(filtered_reg)
        rfm_raw = segmenter.buildRFM()

        if len(rfm_raw) < 3:
            st.warning(
                f"⚠️ Selected region has too few active registered customers ({len(rfm_raw)}) to perform robust clustering. Please broaden your filter."
            )
        else:
            rfm_df, scaled_matrix = compute_rfm_pipeline(filtered_reg)
            segmenter.rfm_df = rfm_df

            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.plotly_chart(segmenter.plot_RFM3D(), use_container_width=True)
            with col_v2:
                st.plotly_chart(segmenter.SegmentRevenues(), use_container_width=True)

            st.markdown("---")
            st.markdown("### 🤖 Unsupervised Machine Learning Clustering")

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                model_choice = st.selectbox(
                    "Select Clustering Algorithm",
                    ["kmeans", "hierarchical", "dbscan"]
                )

            with col_m2:
                max_k = min(10, max(2, len(rfm_raw)))
                hyperparams = {}

                if model_choice == "kmeans":
                    hyperparams["optimal_k"] = st.slider("Number of Clusters ($K$)", min_value=2, max_value=max_k,
                                                         value=min(4, max_k))
                elif model_choice == "hierarchical":
                    hyperparams["optimal_k"] = st.slider("Number of Clusters ($K$)", min_value=2, max_value=max_k,
                                                         value=min(4, max_k))
                    hyperparams["linkage_method"] = st.selectbox("Linkage Criterion",
                                                                 ["ward", "complete", "average", "single"])
                elif model_choice == "dbscan":
                    hyperparams["eps"] = st.slider(r"DBSCAN epsilon ($\epsilon$)", min_value=0.1, max_value=3.0,
                                                   value=0.8, step=0.1)
                    hyperparams["min_samples"] = st.slider("DBSCAN min_samples", min_value=2, max_value=15, value=5)

            cluster_pipeline = ClusteringPipeline(scaled_matrix, segmenter.rfm_df)


            if model_choice == "kmeans":
                cluster_pipeline.tuneKMeans(**hyperparams)
                with st.expander("📈 View K-Means Diagnostics (Elbow & Silhouette)", expanded=False):
                    st.plotly_chart(cluster_pipeline.plotKMeansElbows(max_k=max_k), use_container_width=True)
            elif model_choice == "hierarchical":
                cluster_pipeline.Hierarchical(**hyperparams)
                with st.expander("📈 View Hierarchical Dendrogram", expanded=False):
                    st.plotly_chart(
                        cluster_pipeline.Dendrogram(linkage_method=hyperparams.get("linkage_method", "ward")),
                        use_container_width=True)
            elif model_choice == "dbscan":
                cluster_pipeline.DBscanFit(**hyperparams)
                with st.expander("📈 View DBSCAN k-Distance Graph", expanded=False):
                    st.plotly_chart(
                        cluster_pipeline.DBscanKDistancePlotter(min_samples=hyperparams.get("min_samples", 5)),
                        use_container_width=True)

            st.markdown("### 🔍 Customer Cohort Segmentation Preview")
            st.dataframe(cluster_pipeline.rfm_df.head(100), use_container_width=True)
            st.success(f"✨ {model_choice.upper()} segmentation model successfully executed and evaluated!")


if __name__ == "__main__":
    run_stapp()