import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import linkage
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import pathlib
import pandas as pd
from src.transform import Transform
from src.segmenter import Segmenter


class ClusteringPipeline:
    def __init__(self, scaled_matrix: np.ndarray, rfm_df: pd.DataFrame):
        self.X = scaled_matrix
        self.rfm_df = rfm_df.copy()


    def KMeansEval(self, max_k=10):
        wcss = []
        silhouette_scores = {}

        K = range(2, max_k + 1)
        for k in K:
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
            labels = kmeans.fit_predict(self.X)
            wcss.append(kmeans.inertia_)
            silhouette_scores[k] = silhouette_score(self.X, labels)

        return {"wcss": wcss, "silhouette_scores": silhouette_scores, "k_range": list(K)}


    def plotKMeansElbows(self, max_k=10):
        eval_results = self.KMeansEval(max_k=max_k)
        k_range = eval_results["k_range"]
        wcss = eval_results["wcss"]
        sil_scores = list(eval_results["silhouette_scores"].values())

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('<b>K-Means: Elbow Method (WCSS)</b>', '<b>K-Means: Silhouette Score by k</b>')
        )

        fig.add_trace(
            go.Scatter(x=k_range, y=wcss, mode='lines+markers', name='WCSS', line=dict(color='#27c8e8', width=3)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=k_range, y=sil_scores, mode='lines+markers', name='Silhouette', line=dict(color='#4fbd77', width=3)),
            row=1, col=2
        )

        fig.update_xaxes(title_text="Number of Clusters (k)", row=1, col=1)
        fig.update_yaxes(title_text="WCSS (Inertia)", row=1, col=1)
        fig.update_xaxes(title_text="Number of Clusters (k)", row=1, col=2)
        fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)

        fig.update_layout(template='plotly_white', title_text="<b>K-Means Model Evaluation Diagnostics</b>", showlegend=False)
        return fig


    def Dendrogram(self, linkage_method='ward', p=5):
        dend = ff.create_dendrogram(
            self.X,
            orientation='top',
            linkagefun=lambda x: linkage(x, method=linkage_method)
        )
        dend.update_layout(
            title=f'<b>Hierarchical Clustering Dendrogram (Linkage: {linkage_method})</b>',
            xaxis_title='Sample Index / Cluster',
            yaxis_title='Euclidean Distance',
            template='plotly_white',
            height=500
        )
        return dend


    def DBscanKDistancePlotter(self, min_samples=5):
        neigh = NearestNeighbors(n_neighbors=min_samples)
        nbrs = neigh.fit(self.X)
        distances, _ = nbrs.kneighbors(self.X)

        k_distances = np.sort(distances[:, min_samples - 1], axis=0)

        fig = px.line(
            x=np.arange(len(k_distances)),
            y=k_distances,
            title=f'<b>DBSCAN: k-Distance Graph (min_samples={min_samples})</b>',
            labels={'x': 'Data Points sorted by distance', 'y': f'{min_samples}-th Nearest Neighbor Distance'}
        )
        fig.update_traces(line=dict(color='#C0392B', width=2))
        fig.update_layout(template='plotly_white')
        return fig


    def tuneKMeans(self, optimal_k=4):
        kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42)
        labels = kmeans.fit_predict(self.X)
        score = silhouette_score(self.X, labels)
        self.rfm_df['KMeans_Cluster'] = labels
        print(f"-> K-Means (k={optimal_k}) fitted. Silhouette Score: {score:.4f}")
        return labels


    def Hierarchical(self, optimal_k=4, linkage_method='ward'):
        model = AgglomerativeClustering(n_clusters=optimal_k, metric='euclidean', linkage=linkage_method)
        labels = model.fit_predict(self.X)
        score = silhouette_score(self.X, labels)
        self.rfm_df['Hierarchical_Cluster'] = labels
        print(f"-> Hierarchical (k={optimal_k}, linkage={linkage_method}) fitted. Silhouette Score: {score:.4f}")
        return labels


    def DBscanFit(self, eps=0.8, min_samples=5):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels = dbscan.fit_predict(self.X)
        self.rfm_df['DBSCAN_Cluster'] = labels

        mask = labels != -1
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        print(f"-> DBSCAN (eps={eps}, min_samples={min_samples}) fitted.")
        print(f"   Found {n_clusters} clusters and {n_noise} noise points.")
        if n_clusters > 1:
            score = silhouette_score(self.X[mask], labels[mask])
            print(f"   Silhouette Score (excluding noise): {score:.4f}")
        return labels



def tester():
    df = pd.read_csv(pathlib.Path("../data/OnlineRetail.csv"))
    transformer = Transform(df)
    transformer.clean()
    df_reg = transformer.transform_all()["df_reg"]

    segmenter = Segmenter(df_reg)
    segmenter.buildRFM()
    segmenter.computeRFM()
    segmenter.applySegmentationMapping()

    fig_3d = segmenter.plot_RFM3D()
    fig_3d.show()

    fig_rev = segmenter.SegmentRevenues()
    fig_rev.show()

    scaled_data = segmenter.preprocessor()
    cluster_pipeline = ClusteringPipeline(scaled_data, segmenter.rfm_df)

    fig_elbow = cluster_pipeline.plotKMeansElbows(max_k=10)
    fig_elbow.show()

    fig_dendro = cluster_pipeline.Dendrogram(linkage_method='ward')
    fig_dendro.show()

    fig_dbscan = cluster_pipeline.DBscanKDistancePlotter(min_samples=5)
    fig_dbscan.show()



if __name__=="__main__":
    tester()

