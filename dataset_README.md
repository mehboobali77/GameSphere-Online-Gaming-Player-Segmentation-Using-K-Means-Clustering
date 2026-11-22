# Dataset README — Online Gaming Player Profiles

## Domain
**Online Multiplayer Gaming** — player behavioral data from a fictional battle-royale game.
The dataset captures how players engage with the game across four behavioral dimensions.

## Why This Domain?
Gaming platforms collect rich behavioral telemetry per player.
Identifying distinct player archetypes helps game designers tailor matchmaking,
monetization, and retention strategies to each segment.

## Features (4 numerical, intentionally different scales)

| Feature | Description | Approx Range |
|---|---|---|
| `weekly_hours` | Hours spent in-game per week | 1 – 60 |
| `kill_death_ratio` | Kills divided by deaths (skill proxy) | 0.1 – 8.0 |
| `monthly_spend_usd` | USD spent on in-game items per month | 0 – 300 |
| `squad_sessions` | Number of squad/group play sessions per month | 0 – 40 |

The scales are deliberately mismatched (e.g., `monthly_spend_usd` ranges up to 300 while
`kill_death_ratio` stays below 5) to make feature scaling non-trivial and necessary.

## Generation Method

Three Gaussian clusters were drawn using `numpy.random.normal`, each with distinct
means and standard deviations that reflect a realistic player archetype:

| Cluster | Archetype | weekly_hours | KDR | spend_usd | squad_sessions |
|---|---|---|---|---|---|
| 1 | The Hardcore Grinders | μ=35, σ=6 | μ=3.2, σ=0.6 | μ=120, σ=25 | μ=8, σ=3 |
| 2 | The Social Casuals | μ=14, σ=4 | μ=1.1, σ=0.4 | μ=20, σ=8 | μ=22, σ=5 |
| 3 | The Weekend Warriors | μ=6, σ=2 | μ=0.7, σ=0.25 | μ=5, σ=3 | μ=12, σ=4 |

**Noise injection:** Additional Gaussian noise (`σ=1.0` on `weekly_hours`,
`σ=0.15` on `kill_death_ratio`) was added after generation to blur cluster boundaries
so they overlap slightly and are not trivially separable.

## Expected Clusters
**3 natural clusters** were embedded. The analysis pipeline (K=2..10 sweep) should
rediscover K=3 via the Elbow Method and Silhouette Score.

## Files
- `my_dataset.csv` — 300 records × 4 features (no labels, no cluster column)
- `generate_dataset.py` — reproducible generation script (`np.random.seed(42)`)
