import pathlib
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
def load_data():
    """Loads and runs the data cleaning and transformation pipeline."""
    csv_path = pathlib.Path("data/OnlineRetail.csv")
    if not csv_path.exists():
        csv_path = pathlib.Path("../data/OnlineRetail.csv")

    df_raw = pd.read_csv(csv_path)
    transformer = Transform(df_raw)
    transformer.clean()
    transformed_sets = transformer.transform_all()
    return transformed_sets["df_reg"]


def run_stapp():
    with st.spinner("🚀 Loading & optimizing retail data pipeline..."):
        df = load_data()


    st.sidebar.header("🎛️ Dashboard Controls")
    countries = ['All'] + sorted(df['Country'].dropna().unique().tolist())
    selected_country = st.sidebar.selectbox("Filter by Country", countries)

    filtered_df = df[df['Country'] == selected_country].copy() if selected_country != 'All' else df.copy()
    eda = RetailEDA(filtered_df)


    st.title("🛍️ Online Retail Customer Analytics & Segmentation")
    st.markdown(
        "Executive intelligence platform analyzing macroeconomic trends, product concentration, returns, and RFM customer lifetime value."
    )
    st.markdown("---")


    col1, col2, col3, col4 = st.columns(4)
    total_rev = filtered_df['TotalLineRevenue'].sum()
    total_orders = filtered_df['InvoiceNo'].nunique()
    total_customers = filtered_df['CustomerID'].nunique()
    total_returns = len(filtered_df[filtered_df['Cancelled'] == True])

    with col1:
        st.metric(label="💰 Total Net Revenue", value=f"${total_rev:,.2f}")
    with col2:
        st.metric(label="📦 Unique Orders", value=f"{total_orders:,}")
    with col3:
        st.metric(label="👥 Active Customers", value=f"{total_customers:,}")
    with col4:
        st.metric(label="↩️ Total Cancellations", value=f"{total_returns:,}")

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
            fig_monthly = eda.monthlyRevenue()
            fig_monthly.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_monthly, use_container_width=True)
        with col_b:
            st.markdown("### 💡 Executive Takeaway")
            st.info(
                "Revenue displays a massive hockey-stick surge heading into Q4 (October–December), peaking heavily in November ahead of the holiday shutdown."
            )

    with tab_products:
        st.subheader("Product Revenue Concentration (80/20 Pareto Rule)")
        fig_pareto = eda.paretoRevenue()
        fig_pareto.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pareto, use_container_width=True)

        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown("#### 🕒 Hourly Traffic & Cadence")
            fig_traffic, fig_cadence = eda.analyze_hourly_traffic()
            fig_traffic.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_traffic, use_container_width=True)
        with col_d:
            st.markdown("#### 🛒 Average Order Value (AOV)")
            fig_aov = eda.average_order_values()
            fig_aov.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_aov, use_container_width=True)

        st.plotly_chart(fig_cadence, use_container_width=True)

    with tab_returns:
        st.subheader("Product Returns & Cancellation Impact Analysis")
        fig_canc, canc_stats = eda.cancellations()
        fig_canc.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_canc, use_container_width=True)

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.metric(label="Total Cancelled Transactions", value=f"{canc_stats['cancelled']:,}")
        with col_r2:
            st.metric(label="Estimated Revenue Lost", value=f"${canc_stats['revenue_lost']:,.2f}")

    with tab_rfm:
        st.subheader("Behavioral RFM Analysis & Multi-Model Clustering")

        segmenter = Segmenter(filtered_df)
        rfm_raw = segmenter.buildRFM()

        if len(rfm_raw) < 3:
            st.warning(
                f"⚠️ Selected country '{selected_country}' has too few active customers ({len(rfm_raw)}) to perform robust clustering. Please select 'All' or a larger region."
            )
        else:
            segmenter.computeRFM()
            segmenter.applySegmentationMapping()

            # Display Business Matrix Visualizations
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
                    hyperparams["eps"] = st.slider("DBSCAN epsilon ($\epsilon$)", min_value=0.1, max_value=3.0,
                                                   value=0.8, step=0.1)
                    hyperparams["min_samples"] = st.slider("DBSCAN min_samples", min_value=2, max_value=15, value=5)

            # Preprocess and fit model
            scaled_matrix = segmenter.preprocessor()
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