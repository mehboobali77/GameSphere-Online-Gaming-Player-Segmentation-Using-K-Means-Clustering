"""
generate_dataset.py
Lab 05 — K-Means Clustering | Part A
Domain: Online Gaming Player Profiles
Generates 300 synthetic player records with 3 natural clusters.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
n = 300
per = n // 3  # 100 per cluster

# ============================================================
# Cluster 1: "The Hardcore Grinders"
#   — Many hours, high KDR, heavy spenders, few social ties
# ============================================================
c1_hours      = np.random.normal(35,  6,   per)   # hours/week (high)
c1_kdr        = np.random.normal(3.2, 0.6, per)   # kills-to-death ratio (high)
c1_spending   = np.random.normal(120, 25,  per)   # monthly USD spend (high)
c1_social     = np.random.normal(8,   3,   per)   # squad sessions/month (low)

# ============================================================
# Cluster 2: "The Social Casuals"
#   — Moderate hours, average KDR, low spend, heavy social
# ============================================================
c2_hours      = np.random.normal(14,  4,   per)
c2_kdr        = np.random.normal(1.1, 0.4, per)
c2_spending   = np.random.normal(20,  8,   per)
c2_social     = np.random.normal(22,  5,   per)

# ============================================================
# Cluster 3: "The Weekend Warriors"
#   — Low hours, low KDR, minimal spend, moderate social
# ============================================================
c3_hours      = np.random.normal(6,   2,   per)
c3_kdr        = np.random.normal(0.7, 0.25, per)
c3_spending   = np.random.normal(5,   3,   per)
c3_social     = np.random.normal(12,  4,   per)

df = pd.DataFrame({
    'weekly_hours':    np.concatenate([c1_hours,    c2_hours,    c3_hours]),
    'kill_death_ratio':np.concatenate([c1_kdr,      c2_kdr,      c3_kdr]),
    'monthly_spend_usd':np.concatenate([c1_spending, c2_spending, c3_spending]),
    'squad_sessions':  np.concatenate([c1_social,   c2_social,   c3_social])
})

# Clip to realistic bounds
df['weekly_hours']     = df['weekly_hours'].clip(1, 60)
df['kill_death_ratio'] = df['kill_death_ratio'].clip(0.1, 8)
df['monthly_spend_usd']= df['monthly_spend_usd'].clip(0, 300)
df['squad_sessions']   = df['squad_sessions'].clip(0, 40)

# Add extra noise to blur cluster boundaries (overlap)
df['weekly_hours']      += np.random.normal(0, 1.0, n)
df['kill_death_ratio']  += np.random.normal(0, 0.15, n)

df = df.round(2)

print(f"Shape: {df.shape}")
print(df.describe().round(2))

df.to_csv('my_dataset.csv', index=False)
print(f"\n✅ Generated {len(df)} records → my_dataset.csv")
