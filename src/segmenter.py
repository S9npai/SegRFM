import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler



class Segmenter:
    def __init__(self, df_registered: pd.DataFrame):
        self.df = df_registered.dropna(subset=['CustomerID']).copy()
        self.rfm_df = None
        self.scaled_matrix = None


    def buildRFM(self):
        snapshot_date = self.df['InvoiceDate'].max() + pd.Timedelta(days=1)

        if "TotalLineRevenue" not in self.df.columns:
            self.df["TotalLineRevenue"] = self.df["Quantity"] * self.df["UnitPrice"]

        rfm = self.df.groupby('CustomerID').agg(
            Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
            Frequency=('InvoiceNo', 'nunique'),
            Monetary=('TotalLineRevenue', 'sum')
        ).reset_index()

        self.rfm_df = rfm[rfm['Monetary'] > 0].copy()
        return self.rfm_df


    def computeRFM(self):
        if self.rfm_df is None:
            self.buildRFM()

        r = self.rfm_df.copy()

        def safe_qcut(series, q=5, inverted=False):
            labels = [5, 4, 3, 2, 1] if inverted else [1, 2, 3, 4, 5]
            try:
                _, bins = pd.qcut(series, q, retbins=True, duplicates='drop')
                n_bins = len(bins) - 1
                current_labels = labels[:n_bins]
                if inverted:
                    current_labels = sorted(current_labels, reverse=True)
                else:
                    current_labels = sorted(current_labels)
                return pd.qcut(series, q, labels=current_labels, duplicates='drop').astype(int)
            except Exception:
                ranks = series.rank(method='first')
                min_q = min(q, len(series))
                return pd.qcut(ranks, min_q, labels=False, duplicates='drop') + (0 if not inverted else 1)

        r['R_Score'] = safe_qcut(r['Recency'], q=5, inverted=True)
        r['F_Score'] = safe_qcut(r['Frequency'], q=5, inverted=False)
        r['M_Score'] = safe_qcut(r['Monetary'], q=5, inverted=False)

        r['RFM_Cell'] = r['R_Score'].astype(str) + r['F_Score'].astype(str) + r['M_Score'].astype(str)
        self.rfm_df = r
        return r


    @staticmethod
    def _assign_segments(row) -> str:
        r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]

        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif f >= 3 and m >= 3 and r >= 3:
            return "Loyal Customers"
        elif r <= 2 and (f >= 3 or m >= 3):
            return "At-Risk / High-Value Churn"
        elif r >= 4 and f <= 2:
            return "New Inflow"
        else:
            return "General / Potential Loyalists"


    def applySegmentationMapping(self):
        if 'R_Score' not in self.rfm_df.columns:
            self.computeRFM()

        self.rfm_df["Segment"] = self.rfm_df.apply(self._assign_segments, axis=1)
        return self.rfm_df


    def plot_RFM3D(self):
        if 'Segment' not in self.rfm_df.columns:
            self.applySegmentationMapping()

        fig = px.scatter_3d(
            self.rfm_df,
            x='Recency',
            y='Frequency',
            z='Monetary',
            color='Segment',
            hover_data=['CustomerID', 'R_Score', 'F_Score', 'M_Score', 'RFM_Cell'],
            title='<b>3D Customer RFM Distribution by Business Segment</b>',
            labels={'Recency': 'Recency (Days)', 'Frequency': 'Frequency', 'Monetary': 'Monetary ($)'},
            template='plotly_white'
        )
        fig.update_layout(height=700)
        return fig


    def SegmentRevenues(self):
        if 'Segment' not in self.rfm_df.columns:
            self.applySegmentationMapping()

        seg_summary = self.rfm_df.groupby('Segment').agg(
            Total_Revenue=('Monetary', 'sum'),
            Customer_Count=('CustomerID', 'count'),
            Avg_Monetary=('Monetary', 'mean')
        ).reset_index().sort_values(by='Total_Revenue', ascending=False)

        fig = px.bar(
            seg_summary,
            x='Segment',
            y='Total_Revenue',
            color='Customer_Count',
            color_continuous_scale='Teal',
            text=seg_summary['Total_Revenue'].apply(lambda x: f"${x:,.0f}"),
            title='<b>Revenue Contribution & Customer Count by Business Segment Matrix</b>',
            labels={'Total_Revenue': 'Total Revenue ($)', 'Segment': 'Business Segment',
                    'Customer_Count': 'Customer Count'}
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(template='plotly_white', xaxis_tickangle=-20)
        return fig


    def preprocessor(self):
        if self.rfm_df is None:
            self.buildRFM()

        features = self.rfm_df[['Recency', 'Frequency', 'Monetary']]
        log_features = np.log1p(features)

        scaler = StandardScaler()
        self.scaled_matrix = scaler.fit_transform(log_features)
        return self.scaled_matrix

