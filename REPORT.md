# REPORT.md — Lab 05: K-Means Clustering
## Domain: Online Gaming Player Profiles

---

### Q1 — What domain did you choose and why? What makes your dataset interesting?

**Domain: Online Multiplayer Gaming — Player Behavioral Segmentation**

I chose gaming because behavioral data in this space is genuinely rich and multi-dimensional.
A player's engagement can be split across at least four orthogonal axes: *time investment*
(weekly hours), *skill* (kill/death ratio), *spending* (monthly USD), and *social behavior*
(squad sessions). These axes are largely independent — a player can be highly skilled but
rarely spend money, or be a heavy spender who only plays casually on weekends.
That independence creates naturally distinct clusters rather than a single correlated blob,
making the dataset an honest test case for K-Means.

---

### Q2 — How did you generate the data? How many natural clusters did you embed? How did you inject noise?

Three clusters of 100 records each were drawn from multivariate Gaussian distributions
using `numpy.random.normal` with `np.random.seed(42)` for full reproducibility.
Each cluster corresponds to a real-world player archetype with distinct means:

| Archetype | weekly_hours | KDR | monthly_spend | squad_sessions |
|---|---|---|---|---|
| Hardcore Grinders | 35 ± 6 | 3.2 ± 0.6 | 120 ± 25 | 8 ± 3 |
| Social Casuals | 14 ± 4 | 1.1 ± 0.4 | 20 ± 8 | 22 ± 5 |
| Weekend Warriors | 6 ± 2 | 0.7 ± 0.25 | 5 ± 3 | 12 ± 4 |

**Noise injection:** After assembling the full DataFrame, I added independent Gaussian
noise (σ=1.0 on `weekly_hours`, σ=0.15 on `kill_death_ratio`) to blur the cluster
boundaries. This ensures that clusters overlap at the edges and are not perfectly
separable, making the problem realistic. Values were clipped to valid physical bounds
(e.g., KDR ≥ 0.1, hours ≥ 1).

**Embedded clusters:** 3.

---

### Q3 — What was your optimal K? Did the Elbow Method and Silhouette Score agree? Include the plots.

**Optimal K = 3.** Both methods agreed clearly.

| Method | Evidence |
|---|---|
| Elbow (Inertia) | Inertia drops steeply from K=2 (319.70) to K=3 (183.28), then the curve flattens — the marginal gain from K=4 onwards is much smaller. |
| Silhouette Score | The score peaks at K=2 (0.6588) then remains relatively high at K=3 (0.5353) before declining sharply from K=4 (0.4603) onward. |

K=2 has the highest silhouette but merges distinct behavioral groups (Casuals and
Weekend Warriors are collapsed into one). K=3 captures the true structure embedded in
the data, and both plots point to K=3 as the natural elbow before the improvement curve
flattens. The two methods agree that going beyond K=3 yields diminishing returns.

**Plots:** `elbow_silhouette.png`

---

### Q4 — Name each cluster and describe what makes it unique.

Final K=3 centroids (original scale):

| Cluster | weekly_hours | KDR | monthly_spend | squad_sessions | Name |
|---|---|---|---|---|---|
| 1 | 34.42 | 3.20 | $121.62 | 8.32 | **The Hardcore Grinders** |
| 2 | 13.18 | 1.10 | $18.54 | 23.16 | **The Social Casuals** |
| 0 | 7.77 | 0.73 | $8.14 | 12.22 | **The Weekend Warriors** |

- **The Hardcore Grinders** — play ~34 hrs/week, dominate opponents (KDR 3.2×), and
  invest heavily in cosmetics/items (~$122/month). They play mostly solo or in small
  competitive squads (8 sessions/month). These are the platform's most monetizable users.

- **The Social Casuals** — moderate time investment (~13 hrs), average skill (KDR 1.1),
  low spend (~$19), but the highest squad engagement (23 sessions/month). They play
  for social connection, not competitive ranking. Retention strategy should focus on
  community features, not balance patches.

- **The Weekend Warriors** — lowest engagement across all dimensions: ~8 hrs/week,
  KDR below 1.0 (~0.73), almost no spending (~$8). They dip in casually, probably on
  weekends. Churn risk is highest here; re-engagement campaigns and onboarding improvements
  would address this group.

---

### Q5 — What happened when you removed feature scaling? Show a before/after comparison.

**Plot:** `scaling_comparison.png`

Without `StandardScaler`, `monthly_spend_usd` (range 0–216) completely dominated the
Euclidean distance calculation. The three clusters became horizontal bands sliced purely
along the spending axis, ignoring `weekly_hours`, `kill_death_ratio`, and `squad_sessions`
almost entirely.

Interestingly, the *unscaled* silhouette score (0.7191) is numerically higher than the
*scaled* score (0.5353). This is misleading: when one feature dominates, clusters become
extremely tight along that single axis, which appears "clean" to the silhouette metric.
But those clusters have no behavioral meaning — a weekend casual who happens to spend $50
on a birthday would be placed in the same cluster as a hardcore grinder. **The silhouette
score can be artificially inflated by scale imbalance.** Always scale before K-Means.

---

### Q6 — Were there any points with negative silhouette scores? What does this mean?

Yes. From the silhouette plot (`silhouette_plot.png`), a small number of points in each
cluster extend into negative silhouette territory (coefficient < 0). A negative silhouette
score means that the point is, on average, *closer* to a neighboring cluster's members
than to its own cluster's members. In practical terms, these are misclassified or
boundary-dwelling players: for example, a Social Casual who plays more hours than average
and ends up geometrically closer to the Hardcore Grinders' centroid. These are the points
where the clusters' Gaussian distributions overlap most. K-Means places every point in
exactly one cluster, so borderline points are forced into a category even when their
behavior is genuinely ambiguous.

---

### Q7 — What are the limitations of K-Means for your specific dataset? When might it fail?

1. **Assumes spherical clusters.** K-Means partitions space into Voronoi cells, implicitly
   assuming roughly equal-radius, blob-shaped clusters. If the true player archetypes
   were elongated or crescent-shaped (e.g., "plays many hours AND earns money OR plays
   many hours AND earns nothing"), K-Means would fail to capture that shape. DBSCAN or
   Gaussian Mixture Models would be more appropriate.

2. **Sensitive to outliers.** A professional e-sports player with KDR=8 and 60 hrs/week
   would pull the "Hardcore Grinders" centroid away from the true cluster center.
   K-Means has no mechanism to down-weight outliers.

3. **Fixed K.** K-Means requires K to be specified in advance. If a 4th archetype emerged
   in the future (e.g., "The Streamers" — moderate play, very high social, zero spend),
   the model would not discover it automatically.

4. **Random initialization variance.** Even with `n_init=10`, different runs can sometimes
   produce slightly different cluster membership for boundary points. The overall structure
   is stable here (silhouette 0.53 is solid) but would be less stable with noisier data.

---

### Q8 — If you could add a 4th feature, what would it be and why?

I would add **`avg_session_length_minutes`** — the average duration of a single gaming
session.

This feature would distinguish two currently collapsed sub-groups within the
Hardcore Grinders: players who achieve their ~35 hours through many short-burst sessions
(perhaps competitive rank-pushers who play 10 × 3.5-hour sessions) versus those who have
marathon sessions (e.g., 3 × 12-hour weekend runs). These are fundamentally different
behavioral profiles with different optimal monetization strategies (the former benefits
from daily login rewards; the latter from one-time major content drops). Adding this
feature would likely reveal a 4th cluster, and the Elbow Method would move its
recommendation from K=3 to K=4.

---

*Reproducibility:* All results generated with `np.random.seed(42)` and
`KMeans(n_init=10, random_state=42)`. Python 3.12, scikit-learn 1.x, matplotlib 3.x.
