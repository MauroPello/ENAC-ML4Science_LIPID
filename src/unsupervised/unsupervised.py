"""
Unsupervised learning analysis actions.
Includes Canonical Correlation Analysis (CCA), Clustering, Manifold Learning, and Anomaly Detection.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score,
    mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    confusion_matrix,
)


# Try importing UMAP, fallback to t-SNE if not available
try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


class UnsupervisedAnalyzer:
    """Encapsulates unsupervised learning methods for the analysis pivot."""

    def __init__(self):
        self.cca_model = None
        self.pca_model = None

    def run_pca(
        self, X: pd.DataFrame, n_components: float | int = 0.95
    ) -> tuple[PCA, pd.DataFrame, np.ndarray]:
        """
        Run Principal Component Analysis.

        Args:
            X: Data to analyze.
            n_components: Number of components or explained variance ratio (if < 1.0).

        Returns:
            Tuple with fitted PCA model, transformed data, and explained variance ratio.
        """
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        self.pca_model = pca

        explained_var = pca.explained_variance_ratio_
        cum_var = np.cumsum(explained_var)

        print(
            f"PCA: {pca.n_components_} components explain {cum_var[-1]:.2%} of variance."
        )

        # Create a DataFrame for easier handling
        cols = [f"PC{i+1}" for i in range(pca.n_components_)]
        X_pca_df = pd.DataFrame(X_pca, columns=cols, index=X.index)

        return pca, X_pca_df, explained_var

    def plot_pca_start(self, pca: PCA) -> None:
        """Plot explained variance (Scree Plot)."""
        exp_var = pca.explained_variance_ratio_
        cum_var = np.cumsum(exp_var)

        plt.figure(figsize=(10, 5))
        plt.bar(
            range(1, len(exp_var) + 1),
            exp_var,
            alpha=0.5,
            align="center",
            label="Individual explained variance",
        )
        plt.step(
            range(1, len(exp_var) + 1),
            cum_var,
            where="mid",
            label="Cumulative explained variance",
        )
        plt.ylabel("Explained variance ratio")
        plt.xlabel("Principal component index")
        plt.legend(loc="best")
        plt.title("PCA Scree Plot")
        plt.tight_layout()
        plt.show()

    def run_kmeans(
        self, data: pd.DataFrame, k_range: range = range(2, 11)
    ) -> dict[str, list]:
        """
        Run K-Means for a range of clusters and return metrics for Elbow/Silhouette analysis.
        """
        inertias = []
        sil_scores = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
            if k > 1:
                score = silhouette_score(data, kmeans.labels_)
                sil_scores.append(score)
            else:
                sil_scores.append(0)

        # Plotting
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = "tab:red"
        ax1.set_xlabel("Number of Clusters (k)")
        ax1.set_ylabel("Inertia (Elbow)", color=color)
        ax1.plot(k_range, inertias, color=color, marker="o")
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()
        color = "tab:blue"
        ax2.set_ylabel("Silhouette Score", color=color)
        ax2.plot(k_range, sil_scores, color=color, marker="x")
        ax2.tick_params(axis="y", labelcolor=color)

        plt.title("K-Means Optimization: Elbow Method and Silhouette Score")
        plt.show()

        return {"k": list(k_range), "inertia": inertias, "silhouette": sil_scores}

    def run_gmm(
        self, data: pd.DataFrame, n_components_range: range = range(2, 11)
    ) -> dict[str, list]:
        """
        Run Gaussian Mixture Models and calculate BIC/AIC to find optimal components.
        """
        bics = []
        aics = []

        for n in n_components_range:
            gmm = GaussianMixture(n_components=n, random_state=42, n_init=5)
            gmm.fit(data)
            bics.append(gmm.bic(data))
            aics.append(gmm.aic(data))

        plt.figure(figsize=(10, 6))
        plt.plot(n_components_range, bics, label="BIC", marker="o")
        plt.plot(n_components_range, aics, label="AIC", marker="x")
        plt.xlabel("Number of components")
        plt.ylabel("Score")
        plt.legend()
        plt.title("GMM Model Selection (BIC/AIC)")
        plt.show()

        return {"n": list(n_components_range), "bic": bics, "aic": aics}

    def evaluate_clustering(
        self, data: pd.DataFrame, labels: np.ndarray
    ) -> dict[str, float]:
        """Calculate internal clustering validation metrics."""
        # Ignore noise points (-1) for evaluation if possible,
        # but sklearn metrics usually handle them or penalize them.
        # For DBSCAN, -1 is noise.

        n_labels = len(set(labels)) - (1 if -1 in labels else 0)

        if n_labels < 2:
            return {"silhouette": -1, "calinski": -1}

        sil = silhouette_score(data, labels)
        cal = calinski_harabasz_score(data, labels)

        print(f"Clustering Evaluation (n={n_labels}):")
        print(f"  Silhouette Score: {sil:.4f}")
        print(f"  Calinski-Harabasz Score: {cal:.4f}")

        return {"silhouette": sil, "calinski": cal}

    def run_cca(
        self, X: pd.DataFrame, Y: pd.DataFrame, n_components: int = 2
    ) -> tuple[CCA, np.ndarray, np.ndarray]:
        """
        Run Canonical Correlation Analysis between two views.

        Args:
            X: First view (e.g., Environmental Quality).
            Y: Second view (e.g., Morphology).
            n_components: Number of components to keep.

        Returns:
            Tuple with the fitted CCA model, X variates, and Y variates.
        """
        cca = CCA(n_components=n_components)
        cca.fit(X, Y)
        X_c, Y_c = cca.transform(X, Y)
        self.cca_model = cca

        # Calculate correlations for each component
        corrs = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(n_components)]
        print(f"CCA Correlations (Top {n_components} components): {corrs}")

        return cca, X_c, Y_c

    def plot_cca_results(
        self, X_c: np.ndarray, Y_c: np.ndarray, component_idx: int = 0
    ) -> None:
        """Plot the relationship between X and Y for a specific CCA component."""
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X_c[:, component_idx], y=Y_c[:, component_idx], alpha=0.6)
        plt.xlabel(f"Canonical Variable X (Component {component_idx+1})")
        plt.ylabel(f"Canonical Variable Y (Component {component_idx+1})")
        plt.title(f"CCA Component {component_idx+1} Correlation")
        plt.grid(True, alpha=0.3)
        plt.show()

    def run_hierarchical_clustering(
        self, data: pd.DataFrame, method: str = "ward", metric: str = "euclidean"
    ) -> None:
        """Run hierarchical clustering and plot the dendrogram."""
        plt.figure(figsize=(10, 7))
        plt.title("Hierarchical Clustering Dendrogram")
        Z = linkage(data, method=method, metric=metric)
        dendrogram(Z, no_labels=True)
        plt.xlabel("Sample Index")
        plt.ylabel("Distance")
        plt.show()

    def run_dbscan(
        self, data: pd.DataFrame, eps: float = 0.5, min_samples: int = 5
    ) -> np.ndarray:
        """Run DBSCAN clustering."""
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(data)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print(f"DBSCAN: Found {n_clusters} clusters and {n_noise} noise points.")
        return labels

    def compare_clusters(
        self, labels_true: np.ndarray, labels_pred: np.ndarray
    ) -> None:
        """Compare two sets of cluster labels (e.g., Environment vs Morphology clusters)."""
        ari = adjusted_rand_score(labels_true, labels_pred)
        mi = mutual_info_score(labels_true, labels_pred)
        print(f"Cluster Comparison - Adjusted Rand Index: {ari:.4f}")
        print(f"Cluster Comparison - Mutual Information: {mi:.4f}")

    def run_manifold_learning(
        self,
        data: pd.DataFrame,
        color_by: pd.Series | np.ndarray | None = None,
        method: str = "tsne",
        n_components: int = 2,
    ) -> None:
        """
        Project high-dimensional data into 2D using t-SNE or UMAP.

        Args:
            data: The data to project.
            color_by: Optional array to color points by (e.g., cluster labels or an EQ feature).
            method: 'tsne' or 'umap'.
            n_components: Dimensions to project to (default 2).
        """
        if method == "umap":
            if not HAS_UMAP:
                print("UMAP not installed, falling back to t-SNE.")
                method = "tsne"
            else:
                reducer = umap.UMAP(n_components=n_components, random_state=42)
                embedding = reducer.fit_transform(data)

        if method == "tsne":
            reducer = TSNE(
                n_components=n_components,
                random_state=42,
                init="pca",
                learning_rate="auto",
            )
            embedding = reducer.fit_transform(data)

        plt.figure(figsize=(10, 8))
        if color_by is not None:
            sns.scatterplot(
                x=embedding[:, 0],
                y=embedding[:, 1],
                hue=color_by,
                palette="viridis",
                legend="full",
                alpha=0.7,
            )
        else:
            plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7)

        plt.title(f"{method.upper()} Projection")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.show()

    def run_anomaly_detection(
        self, data: pd.DataFrame, contamination: float = 0.05
    ) -> pd.DataFrame:
        """
        Run Isolation Forest to detect anomalies.

        Returns:
            DataFrame with original data and an 'anomaly' column (-1 for anomaly, 1 for normal).
        """
        iso = IsolationForest(contamination=contamination, random_state=42)
        preds = iso.fit_predict(data)

        result = data.copy()
        result["anomaly"] = preds
        n_anomalies = (preds == -1).sum()
        print(
            f"Isolation Forest: Detected {n_anomalies} anomalies out of {len(data)} samples."
        )
        return result

    def plot_cluster_profile(
        self,
        data: pd.DataFrame,
        labels: np.ndarray,
        title: str = "Cluster Profile (Z-Scores)",
    ) -> None:
        """
        Plot a heatmap of the mean values (or z-scores if input is scaled) for each feature per cluster.
        """
        if len(set(labels)) <= 1:
            print("Not enough clusters to profile.")
            return

        df_p = pd.DataFrame(data, columns=data.columns)
        df_p["Cluster"] = labels
        # Remove noise if present (label -1)
        df_p = df_p[df_p["Cluster"] != -1]

        if df_p.empty:
            print("No valid clusters to profile (all noise).")
            return

        cluster_means = df_p.groupby("Cluster").mean()

        plt.figure(figsize=(12, 8))
        sns.heatmap(cluster_means.T, cmap="RdBu_r", center=0, annot=True, fmt=".2f")
        plt.title(title)
        plt.xlabel("Cluster Label")
        plt.tight_layout()
        plt.show()

    def plot_cca_loadings(
        self, X: pd.DataFrame, Y: pd.DataFrame, cca: CCA = None, component: int = 0
    ) -> None:
        """
        Plot bar charts of the correlations between original features and the canonical variates (loadings).
        """
        if cca is None:
            cca = self.cca_model
            if cca is None:
                print("No CCA model provided or trained.")
                return

        # Transform to get the canonical variates
        # Note: transform returns X_c, Y_c
        X_c, Y_c = cca.transform(X, Y)

        # Calculate loadings (correlation between original vars and canonical component)
        x_loadings = [np.corrcoef(X[col], X_c[:, component])[0, 1] for col in X.columns]
        y_loadings = [np.corrcoef(Y[col], Y_c[:, component])[0, 1] for col in Y.columns]

        fig, axes = plt.subplots(1, 2, figsize=(15, 8))

        # Plot X loadings
        sns.barplot(
            x=x_loadings,
            y=X.columns,
            ax=axes[0],
            palette="vlag",
            hue=X.columns,
            legend=False,
        )
        axes[0].set_title(f"Env Features Correlation with CC{component+1}")
        axes[0].axvline(0, color="k", linestyle="--")

        # Plot Y loadings
        sns.barplot(
            x=y_loadings,
            y=Y.columns,
            ax=axes[1],
            palette="vlag",
            hue=Y.columns,
            legend=False,
        )
        axes[1].set_title(f"Morph Features Correlation with CC{component+1}")
        axes[1].axvline(0, color="k", linestyle="--")

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, title: str = "Confusion Matrix"
    ) -> None:
        """Plot the confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(title)
        plt.show()
