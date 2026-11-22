"""
kmeans_analysis.py
Lab 05 — K-Means Clustering | Parts B & C
Domain: Online Gaming Player Profiles
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

# =====================================================
# PART B — FIND K & TRAIN
# =====================================================

# 1. Load dataset
df = pd.read_csv('my_dataset.csv')
print("=" * 55)
print("DATASET OVERVIEW")
print("=" * 55)
print(f"Shape: {df.shape}")
print(df.describe().round(2))

features = ['weekly_hours', 'kill_death_ratio', 'monthly_spend_usd', 'squad_sessions']
X = df[features].values

# 2. Feature scaling (mandatory before K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaling applied — all features now have μ=0, σ=1")

# 3. K sweep: inertia + silhouette for K=2..10
print("\n" + "=" * 55)
print("K SWEEP  (K = 2 to 10)")
print("=" * 55)
print(f"{'K':>4}  {'Inertia':>10}  {'Silhouette':>12}")
print("-" * 30)

K_range = range(2, 11)
inertias   = []
sil_scores = []

for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))
    print(f"{k:>4}  {km.inertia_:>10.2f}  {sil_scores[-1]:>12.4f}")

# ----------------------------------------------------------------
# VISUALIZATION 2: Elbow + Silhouette Dual Plot  (Part C – Viz 2)
# ----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0a0a0f')

for ax in axes:
    ax.set_facecolor('#12121a')
    ax.tick_params(colors='#8888aa', labelsize=9)
    for sp in ax.spines.values():
        sp.set_color('#2a2a3d')

# Elbow curve
axes[0].plot(list(K_range), inertias, 'o-', color='#14b8a6',
             linewidth=2.5, markersize=7, markerfacecolor='white',
             markeredgecolor='#14b8a6', markeredgewidth=1.5)
axes[0].axvline(x=3, color='#f59e0b', linestyle='--', linewidth=1.8,
                label='Optimal K = 3')
axes[0].annotate('← Elbow', xy=(3, inertias[1]),
                 xytext=(4.2, inertias[1] * 1.05),
                 color='#f59e0b', fontsize=10, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#f59e0b'))
axes[0].set_xlabel('Number of Clusters (K)', color='#8888aa', fontsize=11)
axes[0].set_ylabel('Inertia (WCSS)', color='#8888aa', fontsize=11)
axes[0].set_title('Elbow Method', color='#e8e6f0', fontsize=13, fontweight='bold')
axes[0].grid(alpha=0.15, color='#2a2a3d')
axes[0].legend(framealpha=0.2, labelcolor='#e8e6f0', fontsize=9)

# Silhouette scores
axes[1].plot(list(K_range), sil_scores, 'o-', color='#8b5cf6',
             linewidth=2.5, markersize=7, markerfacecolor='white',
             markeredgecolor='#8b5cf6', markeredgewidth=1.5)
axes[1].axvline(x=3, color='#f59e0b', linestyle='--', linewidth=1.8,
                label='Optimal K = 3')
axes[1].set_xlabel('Number of Clusters (K)', color='#8888aa', fontsize=11)
axes[1].set_ylabel('Silhouette Score', color='#8888aa', fontsize=11)
axes[1].set_title('Silhouette Analysis', color='#e8e6f0', fontsize=13, fontweight='bold')
axes[1].grid(alpha=0.15, color='#2a2a3d')
axes[1].legend(framealpha=0.2, labelcolor='#e8e6f0', fontsize=9)

plt.suptitle('Finding Optimal K — Elbow + Silhouette', color='#e8e6f0',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('elbow_silhouette.png', dpi=150, bbox_inches='tight',
            facecolor='#0a0a0f')
plt.close()
print("\n✅ Saved: elbow_silhouette.png")

# ----------------------------------------------------------------
# 4. FINAL MODEL — K = 3  (justified by both methods agreeing)
# ----------------------------------------------------------------
print("\n" + "=" * 55)
print("FINAL MODEL  (K = 3)")
print("=" * 55)

kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

final_sil = silhouette_score(X_scaled, kmeans.labels_)
print(f"Final Inertia      : {kmeans.inertia_:.2f}")
print(f"Silhouette Score   : {final_sil:.4f}")

# Cluster sizes
print("\nCluster sizes:")
print(df['cluster'].value_counts().sort_index().to_string())

# Centroids in original scale
centroids_orig = scaler.inverse_transform(kmeans.cluster_centers_)
centroid_df = pd.DataFrame(centroids_orig, columns=features)
centroid_df.index.name = 'Cluster'
print("\nCentroids (original scale):")
print(centroid_df.round(2).to_string())

# Name clusters from centroid characteristics
names = {
    kmeans.cluster_centers_[:, 0].argmax(): "The Hardcore Grinders",
    kmeans.cluster_centers_[:, 3].argmax(): "The Social Casuals",
}
remaining = set(range(3)) - set(names.keys())
names[remaining.pop()] = "The Weekend Warriors"

df['segment'] = df['cluster'].map(names)
print("\nCustomer Segments:")
print(df['segment'].value_counts().to_string())

# 5. Cluster statistics
print("\n" + "=" * 55)
print("CLUSTER PROFILES")
print("=" * 55)
cluster_stats = df.groupby('cluster')[features].agg(['mean', 'std']).round(2)
print(cluster_stats.to_string())

# 6. Predict 3 hand-crafted new data points
print("\n" + "=" * 55)
print("PREDICTING 3 NEW PLAYERS")
print("=" * 55)
new_players = pd.DataFrame({
    'weekly_hours':       [38, 12,  4],
    'kill_death_ratio':   [3.5, 1.0, 0.6],
    'monthly_spend_usd':  [130, 18,  3],
    'squad_sessions':     [7,  25,  10]
})
new_scaled   = scaler.transform(new_players)
new_clusters = kmeans.predict(new_scaled)
new_players['predicted_cluster']  = new_clusters
new_players['predicted_segment']  = new_players['predicted_cluster'].map(names)
print(new_players.to_string(index=False))

# =====================================================
# PART C — VISUALIZATION PORTFOLIO
# =====================================================

COLORS = ['#14b8a6', '#8b5cf6', '#f43f5e']

# ----------------------------------------------------------------
# VISUALIZATION 1: 2D Cluster Scatter Plot
#   Features chosen: weekly_hours vs monthly_spend_usd
#   (highest spread / most discriminative pair)
# ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor('#0a0a0f')
ax.set_facecolor('#12121a')
ax.tick_params(colors='#8888aa', labelsize=9)
for sp in ax.spines.values():
    sp.set_color('#2a2a3d')

seg_names = {0: names[0], 1: names[1], 2: names[2]}

for c in range(3):
    mask = df['cluster'] == c
    ax.scatter(df.loc[mask, 'weekly_hours'],
               df.loc[mask, 'monthly_spend_usd'],
               c=COLORS[c], label=names[c],
               alpha=0.6, s=45,
               edgecolor='white', linewidth=0.3)

# Centroids (inverse-transformed back to original scale)
ax.scatter(centroids_orig[:, 0], centroids_orig[:, 2],
           c='white', marker='*', s=450,
           edgecolor='black', linewidth=1.5,
           zorder=10, label='Centroids')

ax.set_xlabel('Weekly Hours Played', color='#8888aa', fontsize=12)
ax.set_ylabel('Monthly Spend (USD)', color='#8888aa', fontsize=12)
ax.set_title('K-Means Clustering — Gaming Player Segments (K=3)',
             color='#e8e6f0', fontsize=13, fontweight='bold')
ax.legend(framealpha=0.25, labelcolor='#e8e6f0', fontsize=9,
          facecolor='#12121a', edgecolor='#2a2a3d')
ax.grid(alpha=0.12, color='#2a2a3d')
plt.tight_layout()
plt.savefig('cluster_scatter.png', dpi=150, bbox_inches='tight',
            facecolor='#0a0a0f')
plt.close()
print("\n✅ Saved: cluster_scatter.png")

# ----------------------------------------------------------------
# VISUALIZATION 3: Silhouette Plot
# ----------------------------------------------------------------
sil_values = silhouette_samples(X_scaled, kmeans.labels_)
avg_score  = silhouette_score(X_scaled, kmeans.labels_)

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor('#0a0a0f')
ax.set_facecolor('#12121a')
ax.tick_params(colors='#8888aa', labelsize=9)
for sp in ax.spines.values():
    sp.set_color('#2a2a3d')

y_lower = 10
for c in range(3):
    cluster_sil = sil_values[kmeans.labels_ == c]
    cluster_sil.sort()
    size = len(cluster_sil)
    ax.fill_betweenx(np.arange(y_lower, y_lower + size),
                     0, cluster_sil,
                     alpha=0.75, color=COLORS[c])
    ax.text(-0.08, y_lower + size / 2, names[c],
            ha='right', va='center', fontsize=9,
            color=COLORS[c], fontweight='bold')
    y_lower += size + 12

ax.axvline(x=avg_score, color='#ff5c6c', linestyle='--', linewidth=2,
           label=f'Average Silhouette: {avg_score:.3f}')
ax.axvline(x=0, color='#555570', linestyle='-', linewidth=1)
ax.set_xlabel('Silhouette Coefficient', color='#8888aa', fontsize=11)
ax.set_ylabel('Data Points (grouped by cluster)', color='#8888aa', fontsize=11)
ax.set_title('Silhouette Plot — K-Means (K=3)', color='#e8e6f0',
             fontsize=13, fontweight='bold')
ax.legend(framealpha=0.25, labelcolor='#e8e6f0', fontsize=9,
          facecolor='#12121a', edgecolor='#2a2a3d')
ax.grid(alpha=0.12, color='#2a2a3d', axis='x')
plt.tight_layout()
plt.savefig('silhouette_plot.png', dpi=150, bbox_inches='tight',
            facecolor='#0a0a0f')
plt.close()
print("✅ Saved: silhouette_plot.png")

# ----------------------------------------------------------------
# VISUALIZATION 4: Feature Distribution by Cluster (Boxplots)
# ----------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('#0a0a0f')

feat_labels = {
    'weekly_hours':       'Weekly Hours Played',
    'kill_death_ratio':   'Kill / Death Ratio',
    'monthly_spend_usd':  'Monthly Spend (USD)',
    'squad_sessions':     'Squad Sessions / Month'
}

for ax, feat in zip(axes.flat, features):
    ax.set_facecolor('#12121a')
    ax.tick_params(colors='#8888aa', labelsize=9)
    for sp in ax.spines.values():
        sp.set_color('#2a2a3d')

    data_by_cluster = [df.loc[df['cluster'] == c, feat].values for c in range(3)]
    bp = ax.boxplot(data_by_cluster,
                    patch_artist=True,
                    medianprops=dict(color='white', linewidth=2),
                    whiskerprops=dict(color='#555570'),
                    capprops=dict(color='#555570'),
                    flierprops=dict(marker='o', color='#555570',
                                   markersize=3, alpha=0.5))
    for patch, color in zip(bp['boxes'], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    # Overall median reference line
    overall_median = df[feat].median()
    ax.axhline(y=overall_median, color='#f59e0b', linestyle='--',
               linewidth=1.2, alpha=0.6, label=f'Overall median: {overall_median:.1f}')

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([f'C{c}\n{names[c][:14]}…' if len(names[c]) > 14
                        else f'C{c}\n{names[c]}' for c in range(3)],
                       color='#8888aa', fontsize=8)
    ax.set_ylabel(feat_labels[feat], color='#8888aa', fontsize=10)
    ax.set_title(feat_labels[feat], color='#e8e6f0', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.12, color='#2a2a3d', axis='y')
    ax.legend(framealpha=0.2, labelcolor='#e8e6f0', fontsize=8,
              facecolor='#12121a', edgecolor='#2a2a3d')

plt.suptitle('Feature Distribution by Cluster — Gaming Player Segments',
             color='#e8e6f0', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('feature_distribution.png', dpi=150, bbox_inches='tight',
            facecolor='#0a0a0f')
plt.close()
print("✅ Saved: feature_distribution.png")

# ----------------------------------------------------------------
# UNSCALED COMPARISON (Part D question: what changes without scaling?)
# ----------------------------------------------------------------
km_unscaled = KMeans(n_clusters=3, n_init=10, random_state=42)
labels_unscaled = km_unscaled.fit_predict(X)   # raw, unscaled X

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.patch.set_facecolor('#0a0a0f')

for ax, (labels, title) in zip(axes,
    [(kmeans.labels_, 'With StandardScaler (Correct)'),
     (labels_unscaled, 'Without Scaling (Distorted)')]):
    ax.set_facecolor('#12121a')
    ax.tick_params(colors='#8888aa', labelsize=9)
    for sp in ax.spines.values():
        sp.set_color('#2a2a3d')

    for c in np.unique(labels):
        mask = labels == c
        ax.scatter(df.loc[mask, 'weekly_hours'],
                   df.loc[mask, 'monthly_spend_usd'],
                   c=COLORS[c % 3], alpha=0.6, s=40,
                   edgecolor='white', linewidth=0.3,
                   label=f'Cluster {c}')
    ax.set_xlabel('Weekly Hours Played', color='#8888aa', fontsize=11)
    ax.set_ylabel('Monthly Spend (USD)', color='#8888aa', fontsize=11)
    ax.set_title(title, color='#e8e6f0', fontsize=12, fontweight='bold')
    ax.legend(framealpha=0.25, labelcolor='#e8e6f0', fontsize=9,
              facecolor='#12121a', edgecolor='#2a2a3d')
    ax.grid(alpha=0.12, color='#2a2a3d')

plt.suptitle('Scaling Comparison — K-Means Results',
             color='#e8e6f0', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('scaling_comparison.png', dpi=150, bbox_inches='tight',
            facecolor='#0a0a0f')
plt.close()
print("✅ Saved: scaling_comparison.png")

sil_unscaled = silhouette_score(X, labels_unscaled)
print(f"\nSilhouette (scaled)  : {final_sil:.4f}")
print(f"Silhouette (unscaled): {sil_unscaled:.4f}")
print("\n✅ All outputs generated successfully.")
