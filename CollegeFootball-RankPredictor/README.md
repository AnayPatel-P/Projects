# 🏈 Predicting College Football Final AP Poll Rankings (2013–2023)

This project uses historical NCAA college football statistics to **predict the final AP Poll ranking** for each team using machine learning models. It combines advanced data wrangling, exploratory analysis, and predictive modeling techniques to explore what drives top-ranked college football programs.

---

## 📦 Project Summary

- 📅 **Timeframe:** 2013–2023
- 🧠 **Model:** Random Forest Regressor (with and without RFE feature selection)
- 📊 **Target:** Final AP Poll Rank (1–25)
- 🏛 **Features:** 100+ team-level stats (offense, defense, turnovers, etc.)
- 🏆 **Goal:** Understand which features best predict end-of-season rankings

---

## 🚀 Highlights

- ✅ Cleaned and merged yearly college football data from 2013 to 2023
- 🏷 Matched team names with final AP poll rankings across years
- 🏛 Integrated conference affiliation for each season
- 📈 Conducted in-depth **EDA** (feature distributions, correlation heatmaps, time-series trends)
- 🤖 Trained Random Forest models to predict rankings
- ⚙️ Applied **Recursive Feature Elimination (RFE)** for feature selection
- 📉 Compared models with and without conference data to assess impact
- 📁 Exported predictions + RMSE scores

---

## 📁 File Structure

```
CFB-RankPredictor/
├── data/
│   ├── cfb20xx.csv (yearly stats)
│   ├── Poll 20xx.csv (final poll rankings)
│   ├── Conference 20xx.csv (team-conference mapping)
├── cfb_merged_with_conference.csv
├── predicted_vs_actual_rankings.csv
├── predicted_vs_actual_rankings_rfe.csv
├── model_notebook.ipynb
└── README.md
```

---

## 🔍 Exploratory Data Analysis (EDA)

- 📊 Distribution plots: ranked vs unranked teams
- 🔗 Top features correlated with final ranking
- 🗺 Average ranking per conference
- 📉 RMSE trend with and without conference features
- 📈 Time-series line plots for top programs (Alabama, Clemson, OSU, Georgia)

---

## 🧠 Modeling

### Model 1: Baseline Random Forest
- **RMSE:** ~5.62
- Input: All numerical + conference one-hot features
- Output: Predicted rank per year (1 = best)

### Model 2: RFE Feature-Selected Random Forest
- **RMSE:** ~5.15
- Reduced input to 30 most predictive features using Recursive Feature Elimination

---

## 🔍 Feature Importance

Top 15 features (RFE-selected) included:
- Total Points
- Avg Turnover Margin
- Yards Allowed
- Off Yards/Game
- Opponent 3rd % Allowed
- Redzone Points
- Sacks
- Touchdowns Allowed

---

## 🏛 Conference Analysis

- Slight performance gain (↓ RMSE) when including conference features
- Average prediction error plotted per conference
- Shows variance in ranking predictability across conferences

---

## 📈 Visualizations

- 📊 KDE and bar plots comparing ranked/unranked stats
- 🧯 Correlation heatmaps across 130+ features
- 📅 Year-by-year final rank line plots
- 🔁 Scatterplots of prediction vs actual rank (with y = x line)
- 📉 Bar charts showing RMSE with/without conference info

---

## 📤 Outputs

- `cfb_merged_with_conference.csv`: Cleaned master dataset
- `predicted_vs_actual_rankings.csv`: Baseline model predictions
- `predicted_vs_actual_rankings_rfe.csv`: RFE-optimized predictions

---

## 🧪 Evaluation Metric

**Root Mean Squared Error (RMSE)**  
Used to evaluate ranking prediction accuracy year-over-year.

---

## 📌 Next Steps

- [ ] Tune hyperparameters with GridSearchCV
- [ ] Try XGBoost or LightGBM for improved accuracy
- [ ] Build a Streamlit app or Power BI dashboard
- [ ] Introduce "momentum" features (e.g., win streak, strength of schedule)

---

## 👨‍💻 Built By

**Anay Patel**  
[LinkedIn](https://www.linkedin.com/in/anaypatel26) | [GitHub](https://github.com/AnayPatel-P)

---