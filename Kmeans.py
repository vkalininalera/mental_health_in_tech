import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import seaborn as sns
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer

data = pd.read_csv('processed_data.csv')

model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,8),timings=False)
visualizer.fit(data)
visualizer.show()

# K-Mean
model = KMeans(n_clusters=3,random_state=0).fit(data)
lab = model.labels_
S = silhouette_score(data,lab)


visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
visualizer.fit(data)
visualizer.show()

# Config
N_RUNS = 4
best_score = -1
best_run_data = None

for seed in range(N_RUNS):
    print(f"\n==== RUN {seed} ====")
    rng = np.random.default_rng(seed)

    # Load and filter data
    data = pd.read_csv('processed_data.csv')
    selector = VarianceThreshold(threshold=0.071)
    data_array = selector.fit_transform(data)
    selected_columns = data.columns[selector.get_support()]
    data = pd.DataFrame(data_array, columns=selected_columns)

    # Permutation Feature Importance
    baseline_score = silhouette_score(data, KMeans(n_clusters=3, random_state=0).fit_predict(data))
    importance_scores = {}

    for col in data.columns:
        data_permuted = data.copy()
        data_permuted[col] = rng.permutation(data_permuted[col])
        permuted_score = silhouette_score(data_permuted, KMeans(n_clusters=3, random_state=0).fit_predict(data_permuted))
        drop = baseline_score - permuted_score
        importance_scores[col] = drop

    importance_df = pd.DataFrame.from_dict(importance_scores, orient='index', columns=['silhouette_drop'])
    importance_df = importance_df.sort_values(by='silhouette_drop', ascending=False)

    selected_features = importance_df[importance_df['silhouette_drop'] > 0].index
    data_selected = data[selected_features]

    # PCA and clustering
    pca = PCA(n_components=3)
    data_pca_3d = pca.fit_transform(data_selected)
    labels = KMeans(n_clusters=3, random_state=0).fit_predict(data_pca_3d)
    score = silhouette_score(data_pca_3d, labels)
    print(f"Silhouette Score after PCA (seed {seed}): {score:.4f}")

    if score > best_score:
        best_score = score
        best_run_data = {
            "labels": labels,
            "selected_features": selected_features,
            "importance_df": importance_df,
            "pca_data": data_pca_3d,
            "data_selected": data_selected,
            "seed": seed
        }

# Save and report best result
print(f"\n🎯 Best Silhouette Score: {best_score:.4f} from seed {best_run_data['seed']}")
# pd.DataFrame(best_run_data['selected_features']).to_csv('best_selected_features.csv', index=False)

# Get features from best run
best_data_selected = best_run_data['data_selected']

# Plot the HeatMap of feature correlations from the best seed
plt.figure(figsize=(18, 16))
corr = best_data_selected.corr()


sns.heatmap(
    corr,
    annot=False,
    cmap='coolwarm',
    linewidths=0.5,
    cbar_kws={"shrink": 0.5}
)

plt.title(f"Correlation Heatmap of Selected Features (Best Seed = {best_run_data['seed']})")
plt.xticks(rotation=90, ha='center', fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.show()

# 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(best_run_data['pca_data'][:, 0],
                best_run_data['pca_data'][:, 1],
                best_run_data['pca_data'][:, 2],
                c=best_run_data['labels'], cmap='viridis', s=40)
ax.set_title(f"Best Clustering (seed={best_run_data['seed']})")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.show()


# Cluster summary
clustered = best_run_data['data_selected'].copy()
clustered['cluster'] = best_run_data['labels']
summary = clustered.groupby('cluster').mean().T

binary_columns = [col for col in summary.index if summary.loc[col].max() <= 1.0 and col != 'What is your age?']
summary.loc[binary_columns] *= 100
# Save to CSV
summary.to_csv('best_cluster_summary.csv')
