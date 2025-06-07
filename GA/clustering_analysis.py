import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import os
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


os.chdir('Statistical-Learning/GA')

SEED = 4645251


# wine_quality = fetch_ucirepo(id=186) 
# X = wine_quality.data.features 
# y = wine_quality.data.targets

data = pd.read_csv('data/wines.csv')
X = data.drop(columns=['quality', 'colour'])


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_X = PCA()
pca_X.fit(X_scaled)
scores = pca_X.transform(X_scaled)

three_PCs = scores[:, :3]


def run_KMeans():
    wcss = []
    pve = []
    tss = np.sum(np.square(three_PCs))
    k_vals = range(2,11)
    for k in k_vals:
        kmeans = KMeans(n_clusters=k, random_state=SEED, n_init=50).fit(three_PCs)
        wcss.append(kmeans.inertia_)
        pve.append(1 - kmeans.inertia_/tss)

    plt.figure(figsize=(12,8))
    plt.plot(k_vals, wcss, marker='o', color='dodgerblue')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.title('Scree plot of WCSS against $k \\in [2, 10]$')
    plt.savefig('plots/scree_plot.png')

    plt.figure(figsize=(12,8))
    plt.plot(k_vals[1:], np.diff(wcss), marker='o', color='seagreen')
    plt.xlabel('Number of clusters')
    plt.ylabel('$\\frac{dWCSS}{dk}$')
    plt.title('Scree plot of $\\frac{dWCSS}{dk}$ against $k \\in [2, 10]$')
    plt.savefig('plots/scree_plot_deriv.png')

    plt.figure(figsize=(12,8))
    plt.plot(k_vals, pve, marker='o', color='maroon')
    plt.xlabel('Number of clusters')
    plt.ylabel('PVE')
    plt.title('Scree plot of PVE against $k \\in [2, 10]$')
    plt.savefig('plots/scree_plot_pve.png')


def silhouette_analysis(range_n_clusters):

    for n_clusters in range_n_clusters:

        fig, (ax1, ax2) = plt.subplots(1,2)
        fig.set_size_inches(18, 7)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=SEED)
        cluster_labels = kmeans.fit_predict(three_PCs)

        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            "For n_clusters ="
            , n_clusters
            , "The average silhouette_score is:"
            , silhouette_avg
        )

        sample_silhouette_values = silhouette_samples(three_PCs, cluster_labels)
        ax1.set_xlim([min(sample_silhouette_values), 1])
        ax1.set_ylim([0, len(three_PCs) + (n_clusters + 1)*10])

        y_lower = 10
        for i in range(n_clusters):
            # aggregate the silhouette scores for samples belonging to cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            colour = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper)
                , 0
                , ith_cluster_silhouette_values
                , facecolor=colour
                , edgecolor=colour
                , alpha=0.7
            )
            ax1.text(-0.05, y_lower+0.5*size_cluster_i, str(i))
            
            y_lower = y_upper + 10
        
        ax1.set_title(f'Silhouette plot for {n_clusters} clusters.')
        ax1.set_ylabel('Silhouette coefficient values.')
        ax1.set_ylabel('Cluster label')
        ax1.axvline(x=silhouette_avg, color='red', linestyle='--')
        ax1.set_yticks([])
        ax1.set_xticks([-0.2,0,0.2,0.4,0.6,0.8,1])

        colours = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            three_PCs[:, 0], three_PCs[:, 1], marker='.', s=30, lw=0, alpha=0.3, c=colours, edgecolor='k'
        )
        centres = kmeans.cluster_centers_
        ax2.scatter(
            centres[:, 0]
            , centres[:, 1]
            , marker='o'
            , c='white'
            , alpha=1
            , s=200
            , edgecolor='k'
        )
        for i, c in enumerate(centres):
            ax2.scatter(c[0], c[1], marker='$%d$'%i, alpha=1, s=50, edgecolor='k')
        ax2.set_title('PC2 vs PC1 clusters')
        ax2.set_ylabel('PC2')
        ax2.set_xlabel('PC1')

        plt.suptitle(
            f"Silhouette analysis for KMeans on full wine data set with {n_clusters} clusters."
            , fontsize=14
            , fontweight='bold'
        )
        plt.savefig(f'plots/silhouette_k={n_clusters}.png')


if __name__ == '__main__':
    silhouette_analysis(range_n_clusters = [2,3,4,5])
