# 🏏 Cricket Scouting Intelligence System
### Data-Driven Talent Discovery for Rajasthan Royals | **Rank 4 / 7,599 Teams — SuperRR Selector Hackathon**

<p align="left">
  <img src="https://img.shields.io/badge/Rank-4%20%2F%207%2C599%20Teams-gold?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Organiser-Rajasthan%20Royals%20(IPL)-pink?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/ML-LightGBM-green?style=for-the-badge" />
</p>

---

## 🎯 Problem Statement

Rajasthan Royals and Team India selectors face a genuine data problem:

- **3,738 uncapped Indian cricketers** competing for limited national team spots
- Traditional scouting is **subjective and regionally biased** — Mumbai, Delhi, and Bangalore players get disproportionate attention
- Existing ranking systems use **raw merit scores that structurally favour batsmen** — pure merit ranking surfaces 38 batsmen and 2 bowlers in the Top 40, which doesn't reflect how real squads are built

**The goal:** Build a data-driven system to identify the Top 40 uncapped Indian cricketers most likely to receive a national cap — *before* selectors do.

---

## 🏗️ System Architecture

The pipeline follows a **3-Pillar Domain-First Framework**:

```
602,992 Ball-by-Ball Records
          │
          ▼
┌─────────────────────────┐
│  Feature Engineering    │  ← 84 cricket intelligence features
│  (Batting + Bowling +   │
│   Trajectory + WTO)     │
└────────────┬────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────┐
│           Cricket Likelihood Engine (CLE)              │
│                                                        │
│   80% Domain Intelligence  │  20% ML Validation        │
│   ─────────────────────────┼─────────────────────────  │
│   Phase-wise performance   │  LightGBM (AUC: 0.614)    │
│   Trajectory scoring       │  5-fold Stratified CV      │
│   Consistency metrics      │  Trained on 284 capped     │
│   Opportunity gap          │  players as ground truth   │
└────────────┬───────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────┐
│  9 Player Archetypes    │  ← K-Means clustering
│  (IPL-style roles)      │    Silhouette Score: 0.329
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Role-Normalized        │  ← Fixes structural bowler bias
│  Position-Based         │    25 Batsmen | 10 Bowlers | 5 ARs
│  Top 40 Selection       │
└────────────┬────────────┘
             │
             ▼
      Top 40 Final Rankings
      + 360° Player Cards
      + Capping Window Predictions
```

---

## 📊 Dataset

| Source | Records | Description |
|---|---|---|
| Ball-by-ball data | 602,992 deliveries | Every ball across domestic T20 matches |
| Player profiles | 3,738 uncapped players | Indian cricketers with nationality filter |
| Capped players (ground truth) | 284 players | Used for LightGBM training labels |
| Playing XI data | Multi-season | Match context and selection history |

---

## 🔧 Feature Engineering — 84 Cricket Intelligence Features

### Batting Features (40+)
- **Core stats:** Strike rate, average, boundary percentage
- **Phase-wise:** Powerplay SR, Middle-overs SR, Death SR
- **Consistency:** Runs standard deviation, innings-to-innings variance
- **Technical:** Shot variety score, scoring zone distribution
- **Matchup:** Performance vs pace vs spin

### Bowling Features (30+)
- **Core stats:** Economy rate, wicket-taking rate, dot ball percentage
- **WTO (Wicket-Taking Opportunities):** Wickets per over bowled — captures bowling efficiency independent of team context
- **Phase-wise:** Powerplay economy, middle-overs control, death bowling economy
- **Spatial control:** Wide percentage, yorker accuracy proxy

### Trajectory Features (5 per player type)
- **EWMA trend slope:** Exponentially weighted moving average of performance over time
- **Form index:** Recent 10 innings vs career average ratio
- **Consistency floor:** Minimum performance threshold
- **Peak ceiling:** Top 10% innings benchmark

### Opportunity Gap Feature
Identifies underutilized talent — players who consistently outperform their selection opportunities. Calculated as the delta between output percentile rank and opportunity (matches played) percentile rank.

---

## 🤖 Cricket Likelihood Engine (CLE)

The CLE is a **domain-first scoring system** that combines cricket expertise with ML validation.

### Role-Specific Scoring Formula

| Player Type | Batting Weight | Bowling Weight | Trajectory | Consistency |
|---|---|---|---|---|
| Batsman | 60% | — | 20% | 20% |
| Bowler | — | 60% | 20% | 20% |
| All-Rounder | 40% | 40% | 20% | — |

### Final Score = 80% CLE + 20% ML Probability

> **Why only 20% ML?** The training dataset has only 284 capped players — a small sample for a complex classification problem. Over-relying on ML would cause overfitting and overfit to historical biases (e.g., certain states or academies producing capped players). Domain expertise is more reliable here.

---

## 🧠 Key Innovation: Solving the Bowler Bias Problem

Traditional ranking systems favour batsmen because raw metrics don't compare fairly across roles:

| Issue | Impact |
|---|---|
| Strike rates (130-200) look higher than economies (6-9) | Batsmen inflate scores |
| Boundary count dominates over dot ball count | Batsmen ranked higher |
| **Result: Pure merit Top 40 = 38 batsmen, 2 bowlers** | Squad is unusable |

**Solution implemented:**

1. **Normalize CLE scores within each role** — batsmen compete with batsmen, bowlers with bowlers
2. **Position-based Top 40 selection** — 25 batsmen, 10 bowlers, 5 all-rounders, mirroring how actual T20I squads are structured

---

## 🎭 9 Player Archetypes — IPL-Style Role Identification

K-Means clustering (k=9, Silhouette Score: 0.329) on batting and bowling metrics:

| # | Archetype | Count | Key Characteristic |
|---|---|---|---|
| 1 | Powerplay Aggro Batter | 422 | High powerplay SR, boundary-heavy |
| 2 | Death Hitter / Finisher | 509 | Death-overs SR > 180 |
| 3 | Spin Basher | 299 | High SR vs spin-heavy attacks |
| 4 | Middle-Overs Anchor | 829 | Consistency, wicket preservation |
| 5 | Death Seam Specialist | 690 | Low death economy, yorker control |
| 6 | Powerplay Bowler | 108 | Early wicket-taker, movement |
| 7 | Spin Controller | 296 | Middle-overs economy, flight |
| 8 | All-Rounder (Pace) | 259 | Dual threat — bat + pace |
| 9 | All-Rounder (Spin) | 610 | Dual threat — bat + spin |

---


## ⏰ Capping Window Predictions

Players are assigned to actionable capping timelines:

| Window | Score Range | Action |
|---|---|---|
| 🟢 Ready Now (0–6 months) | 73.9 – 81.6 | Invite to T20I camp immediately |
| 🟡 Near Ready (6–12 months) | 69 – 73 | Track in India A tours |
| 🟠 Developing (1–2 years) | 65 – 69 | Monitor via domestic scouts |
| 🔴 Future Talent (2+ years) | < 65 | U-23 / State association pipeline |

---

## 📁 Repository Structure

```
├── RR_Feature_Engineering.ipynb   # 84 cricket features from ball-by-ball data
├── RR_Hackathon_Analysis.ipynb    # EDA, archetype analysis, validation
├── RR_ML_PIPELINE.py              # Full end-to-end pipeline script
├── Final_submission_ppt.pdf       # Competition submission presentation
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Data Processing | Python, Pandas, NumPy, SciPy |
| Machine Learning | LightGBM, Scikit-learn (KMeans, StandardScaler, StratifiedKFold) |
| Visualization | Matplotlib, Seaborn, Plotly |
| Statistical Analysis | EWMA trajectory modelling, Silhouette scoring, ROC-AUC validation |

---




> **Note:** The competition dataset (`player_features_complete.parquet`, `ball_by_ball_processed.parquet`) is proprietary to the Rajasthan Royals / Unstop hackathon and cannot be redistributed. The pipeline code and methodology are fully open.

---

## 📈 Validation

| Metric | Value | Context |
|---|---|---|
| LightGBM AUC | 0.614 ± 0.040 | 5-fold stratified CV on 284 capped players |
| Clustering Silhouette | 0.329 | Acceptable for real-world cricket data |
| Bowler bias gap fixed | 21 points | Before vs after role normalization |
| Competition rank | **4 / 7,599 teams** | SuperRR Selector Hackathon, Unstop |

---

## 💡 Key Insights for Rajasthan Royals

1. **Death bowlers are scarce** — only 4 death specialists in Top 40. Premium auction targets.
2. **Regional arbitrage** — Assam and Northeast players show comparable quality at significantly lower auction costs.
3. **Trajectory > current form** — Players with trajectory score > 90 are consistently capped within 6–12 months (Rinku Singh, Arshdeep Singh pattern).
4. **All-rounder shortage** — Only 5 Hardik Pandya-type players in Top 40. Highest squad-balance value.

---

## 👤 Author

**Bishal Roy**
Final Year B.E. — Artificial Intelligence & Data Science
Dr. D.Y. Patil Institute of Technology, Pune

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/bishal-roy-5410b5257/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/roybishal362)

---

*"Tomorrow's India stars are hidden in today's data. The question isn't IF they'll be capped — it's WHEN."*
