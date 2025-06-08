import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score, adjusted_rand_score


SEED = 4645251


# wine_quality = fetch_ucirepo(id=186) 
# X = wine_quality.data.features 
# y = wine_quality.data.targets

# The reason why we don't include the target in this analysis is that we would obtain
# biased clusters that are defined by the outcome, while here our goal is to discover 
# the underlying chemical profiles that lead to high or low quality ratings.

# You could ask why don't we just take the rows in the data set where the quality is
# excellent or poor and average out the profiles like we do in this programme.
# Clustering allows us to single out different chemical profiles and study how those
# outliers are assigned to these profiles.

# So let's say we're data scientist that were hired by businessmen looking to acquire
# a wine domain and market their wine as very high quality.
# If we showed them our results after part 1, they'd go "Well that's great, you were
# able to single out 

# What we take away from the Kmeans graph is that our data has a very complex 
# structure. There is no single cluster that contains exclusively excellent or poor
# wines, so GMM would be a good fit here.

# We can try and look at the extreme quality scores (3 and 9) but there's only a few
# samples with these scores so results would likely not be statistically significant.
# K-means does hard cluster assignment, so GMM provides a more nuanced view.
# Therefore GMM is an ideal choice of method to study these extreme wines.
# With GMM we can look at extreme quality scores and their responsibilities.


# Now I have seen that there seem to be sub-clusters of excellent and terrible wines.
# rerun the target viz Kmeans plot with the colour scale being the wine grade.
# Now that I KNOW what these sub clusters look like, how can I single them out?
# i know they exist based on their chemical features, now i'm just trying to
# find their Gaussian distributions. Should I run GMM with the target?

# only clustering on data points with extreme quality scores is not a good idea 
# because it doesn't say anything about the probability that a new wine with a 
# chemical profile similar to one of these clusters is actually excellent or poor. 
# Let's say that we exclude all medium wines and find 2 clusters of excellent wines. 
# One might indeed be an isolated island with a high percentage of excellent wines, 
# but the other might in reality be an area of the PC space with very high density, 
# and this second cluster of excellent one could be nested within a much more dense 
# cluster that also contains a large amount of medium wines, and so the probability 
# of finding an excellent wine in that chemical profile is very low.


data = pd.read_csv('data/wines.csv')
data['wine_grade'] = data['quality'].apply(
    lambda x: 2 if x>=8 else (0 if x<=4 else 1)
)
X = data.drop(columns=['quality', 'colour', 'wine_grade'])
y = data['quality']
wine_grade = data['wine_grade']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_X = PCA()
pca_X.fit(X_scaled)
scores = pca_X.transform(X_scaled)

n_PC = 3
range_n_clusters = [2,3,4,5]
k = 3


def run_KMeans(PC_scores):
    wcss = []
    pve = []
    tss = np.sum(np.square(PC_scores))
    k_vals = range(2,11)
    for k in k_vals:
        kmeans = KMeans(n_clusters=k, random_state=SEED, n_init=50).fit(PC_scores)
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


def silhouette_analysis(range_n_clusters, PC_scores):

    for n_clusters in range_n_clusters:

        fig, (ax1, ax2) = plt.subplots(1,2)
        fig.set_size_inches(18, 7)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=50)
        cluster_labels = kmeans.fit_predict(PC_scores)

        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            "For n_clusters ="
            , n_clusters
            , "The average silhouette_score is:"
            , silhouette_avg
        )

        sample_silhouette_values = silhouette_samples(PC_scores, cluster_labels)
        ax1.set_xlim([min(sample_silhouette_values), 1])
        ax1.set_ylim([0, len(PC_scores) + (n_clusters + 1)*10])

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
            PC_scores[:, 0], PC_scores[:, 1], marker='.', s=30, lw=0, alpha=0.3, c=colours, edgecolor='k'
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


def kmeans_stability_analysis(X, k, n_rep):
    '''     
        sample x% from full data and keep labels for those rows
        obtain labels from new KMeans object results, and pass both of these lists to adjusted_rand_score
    '''

    full_set_labels = KMeans(n_clusters=k, random_state=SEED, n_init=50).fit_predict(X)
    results = []
    subset_sizes = np.arange(0.1, 0.9, 0.1)

    for size in subset_sizes:
        adj_rand_scores = np.zeros((n_rep,))
        for i in range(n_rep):
            idx = np.random.randint(len(X)-1, size=int(size*len(X)))
            X_sample = X[idx, :]
            sample_labels = KMeans(n_clusters=k, random_state=SEED, n_init=50).fit_predict(X_sample)
            adj_rand_scores[i] = adjusted_rand_score(full_set_labels[idx], sample_labels)
            
            # model_sample = KMeans(n_clusters=k, random_state=SEED, n_init=50).fit(X_sample)
            # labels_recluster = KMeans(n_clusters=k, init=model_sample.cluster_centers_).fit_predict(X)
            # adj_rand_scores[i] = adjusted_rand_score(full_set_labels, labels_recluster)
        results.append(
            [round(np.mean(adj_rand_scores), 2), round(np.std(adj_rand_scores), 2)]
        )
        print(f"Mean ARI for {100*size} sample:", round(np.mean(adj_rand_scores), 2))

    df = pd.DataFrame(results, columns=['Mean ARI', 'SD'], index=subset_sizes)
    print(df)


def plot_cluster_profiles(X, columns, cluster_labels, colours):
    X_df = pd.DataFrame(X, columns=columns)
    X_df['cluster'] = cluster_labels
    clusters = sorted(X_df['cluster'].unique())
    features = X_df.columns.drop('cluster')

    means = X_df.groupby('cluster')[features].mean()
    # stds = X_df.groupby('cluster')[features].std()

    fig, ax = plt.subplots(figsize=(10, 8))
    y_positions = np.arange(len(features))

    width = 0.2
    offsets = np.linspace(-width, width, len(clusters))

    for i, cluster in enumerate(clusters):
        ax.barh(
            y_positions + offsets[i],
            means.loc[cluster],
            height=width,
            # xerr=stds.loc[cluster],
            label=f'Cluster {cluster}',
            color=colours[i],
            alpha=0.7
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(features)
    ax.set_xlabel('Average value (normalised)')
    ax.set_title(f'Feature profiles by cluster (k={k})')
    ax.legend()
    plt.tight_layout()
    plt.show()
    # plt.savefig(f'plots/kmeans_feature_profiles_k={k}.png')


def plot_kmeans_with_target(X, y, cluster_labels, cluster_centroids):
    fig, ax = plt.subplots(figsize=(12,8))
    clusters = np.unique(cluster_labels)
    markers=('o', 'x', 's')
    big_markers = ('o', 'X', 's')

    extreme_idx = y[y!=1].index
    X_extr = X[extreme_idx]
    y_extr = y[extreme_idx]
    labels_extr = cluster_labels[extreme_idx]
    for i, cluster in enumerate(clusters):
        mask = (labels_extr  == cluster)
        ax.scatter(
            X_extr[mask, 0]
            , X_extr[mask, 1]
            , c=y_extr[mask]
            , cmap='coolwarm'
            , marker=markers[i]
            , label=f'Cluster {cluster}'
            , alpha=0.5
            , s=15
        )
    for i, c in enumerate(cluster_centroids):
        ax.scatter(c[0], c[1], marker=big_markers[i], c='white', alpha=1, s=300, edgecolor='k')
        ax.scatter(c[0], c[1], marker='$%d$'%i, alpha=1, s=50, edgecolor='k')
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Wine quality score')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('K-means wine quality score distribution')
    ax.legend()
    plt.show()
    # plt.savefig('plots/kmeans_target_viz.png')


def plot_kmeans_grade_hist(cluster_labels, wine_grade):
    '''
        here, k is the number of KMeans clusters that is defined at the top of this programme
        get indicies of wine_grade where grade==0 or 2
        get labels of the poor and excellent wines by using these indices, put them in 2 different lists
        plot hist of labels for both lists
    '''

    terrible_wines_idx = wine_grade[wine_grade==0].index
    terrible_wines_labels = cluster_labels[terrible_wines_idx]

    excellent_wines_idx = wine_grade[wine_grade==2].index
    excellent_wines_labels = cluster_labels[excellent_wines_idx]

    plt.figure(figsize=(8,8))
    ax = sns.histplot(terrible_wines_labels, binwidth=1, binrange=(-0.5, k-0.5), color='mediumslateblue')
    counts = []
    for i in np.unique(cluster_labels):
        count = len(np.where(terrible_wines_labels == i)[0])
        counts.append(count)
        pct = 100 * count / len(terrible_wines_labels)
        ax.text(i, count + 5, f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.xticks(np.arange(0, k))
    plt.ylim(0, max(counts) + 20)
    plt.title('Cluster frequencies for poor wines')
    plt.xlabel('Cluster')
    plt.show()
    # plt.savefig('plots/kmeans_terrible_hist.png')

    plt.figure(figsize=(8,8))
    ax = sns.histplot(excellent_wines_labels, binwidth=1, binrange=(-0.5, k-0.5), color='coral')
    counts = []
    for i in np.unique(cluster_labels):
        count = len(np.where(excellent_wines_labels == i)[0])
        counts.append(count)
        pct = 100 * count / len(excellent_wines_labels)
        ax.text(i, count + 5, f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.xticks(np.arange(0, k))
    plt.ylim(0, max(counts) + 20)
    plt.title('Cluster frequencies for excellent wines')
    plt.xlabel('Cluster')
    plt.show()
    # plt.savefig('plots/kmeans_excellent_hist.png')


''' GMM '''

def GMM(X, y, range_n_components):

    for i in range_n_components:
        gmm = GaussianMixture(n_components=i, covariance_type='full', init_params='kmeans', n_init=50, random_state=SEED)
        gmm.fit(X)
        probs = gmm.predict_proba(X)
        labels = gmm.predict(X)

        df = pd.DataFrame()
        df['PC1'] = X[:, 0]
        df['PC2'] = X[:, 1]
        df['wine_grade'] = y
        df['cluster'] = labels

        # PLOT:
        fig, ax = plt.subplots(figsize=(12,8))
        grades = ['Poor', 'Medium', 'Excellent']
        grade_colours = ['#3B4CC0', '#DDDDDD', '#B40426']
        for grade in [1,0,2]:
            mask = (y == grade)
            ax.scatter(
                X[mask, 0]
                , X[mask, 1]
                , c=grade_colours[grade]
                , marker='o'
                , label=grades[grade]
                , alpha=0.5
                , s=15
            )
        # TODO: calculate centroids
        for i, c in enumerate(cluster_centroids):
            ax.scatter(c[0], c[1], marker='o', c='white', alpha=1, s=200, edgecolor='k')
            ax.scatter(c[0], c[1], marker='$%d$'%i, alpha=1, s=50, edgecolor='k')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('')
        ax.legend()
        plt.show()



if __name__ == '__main__':
    # run_KMeans(PC_scores=scores[:, :n_PC])
    # silhouette_analysis(range_n_clusters=range_n_clusters, PC_scores=scores[:, :n_PC])
    # kmeans_stability_analysis(X=scores[:, :n_PC], k=k, n_rep=100)

    kmeans = KMeans(n_clusters=k, random_state=SEED, n_init=50).fit(scores[:, :n_PC])
    full_set_labels = kmeans.labels_
    cluster_centroids = kmeans.cluster_centers_
    colours = cm.nipy_spectral(np.unique(full_set_labels.astype(float)) / k)
    plot_cluster_profiles(X_scaled, X.columns, full_set_labels, colours)
    plot_kmeans_with_target(scores[:, :2], wine_grade, full_set_labels, cluster_centroids)
    plot_kmeans_grade_hist(full_set_labels, wine_grade)
