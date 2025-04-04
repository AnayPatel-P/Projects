# College Football Rank Predictor
# =======================================
# Predicts final AP Poll rank using historical NCAA team stats (2013–2023)
# =======================================

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE

# ----------------------------
# 1. Load and Merge CFB Stats
# ----------------------------
data_dir = "data"
cfb_years = range(13, 24)
file_info = [(f"{data_dir}/cfb{year}.csv", year) for year in cfb_years]

common_cols, dfs = None, []
for path, year in file_info:
    df = pd.read_csv(path)
    df.columns = df.columns.str.replace('.', ' ', regex=False).str.strip()
    df['year'] = year
    common_cols = set(df.columns) if common_cols is None else common_cols & set(df.columns)
    dfs.append(df)

common_cols = list(common_cols | {'year'})
merged_df = pd.concat([df[common_cols].copy() for df in dfs], ignore_index=True)
merged_df = merged_df[['year', 'Team'] + [col for col in merged_df.columns if col not in ['year', 'Team']]]

# ----------------------------
# 2. Load Final AP Rankings
# ----------------------------
poll_info = [(f"{data_dir}/Poll {year}.csv", year) for year in cfb_years]
final_ranks = []

for file_path, year in poll_info:
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    week_col = next(col for col in df.columns if 'week' in col.lower())
    team_col = next(col for col in df.columns if 'school' in col.lower() or 'team' in col.lower())
    rank_col = next(col for col in df.columns if 'rank' in col.lower())
    final_df = df[df[week_col] == df[week_col].max()][[team_col, rank_col]].copy()
    final_df.columns = ['Team', 'Final Rank']
    final_df['year'] = year
    final_ranks.append(final_df)

rank_df = pd.concat(final_ranks).drop_duplicates(subset=['Team', 'year'])
rank_df['Team'] = rank_df['Team'].str.lower().str.strip()
merged_df['Team'] = merged_df['Team'].str.lower().str.replace(r"\s*\(.*?\)", "", regex=True)
merged_df = merged_df.merge(rank_df, on=['Team', 'year'], how='left')

# ----------------------------
# 3. Add Conference Labels
# ----------------------------
def clean_team(name):
    return re.sub(r"\s*\(.*?\)", "", str(name)).strip().lower()

team_corrections = {
    "arizona st.": "arizona state", "arkansas st.": "arkansas state",
    "army west point": "army", "ball st.": "ball state",
    "central mich.": "central michigan", "e. michigan": "eastern michigan",
    "fla. atlantic": "florida atlantic", "fla. intl.": "florida international",
    "florida st.": "florida state", "georgia st.": "georgia state",
    "kent st.": "kent state", "louisiana": "louisiana-lafayette",
    "massachusetts": "umass", "miami (oh)": "miami oh", "miami (fl)": "miami",
    "middle tenn.": "middle tennessee", "mississippi st.": "mississippi state",
    "nc state": "north carolina state", "n. illinois": "northern illinois",
    "oklahoma st.": "oklahoma state", "oregon st.": "oregon state",
    "penn st.": "penn state", "san diego st.": "san diego state",
    "san jose st.": "san jose state", "southern miss.": "southern miss",
    "uab": "alabama-birmingham", "ucf": "central florida", "uconn": "connecticut",
    "umass": "massachusetts", "unlv": "nevada-las vegas", "usc": "southern california",
    "utep": "texas-el paso", "utsa": "texas-san antonio",
    "western kentucky": "w. kentucky", "western mich.": "western michigan"
}

merged_df['Team'] = merged_df['Team'].replace(team_corrections)

conf_dfs = []
for year in range(2013, 2024):
    path = f"{data_dir}/Conference {year}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        df = df.rename(columns={'School': 'Team'})
        df['Team'] = df['Team'].apply(clean_team).replace(team_corrections)
        df['year'] = year
        conf_dfs.append(df)

conference_df = pd.concat(conf_dfs, ignore_index=True)
merged_df = pd.merge(merged_df, conference_df[['Team', 'year', 'Conference']], on=['Team', 'year'], how='left')

# ----------------------------
# 4. Train Model + Predict Ranks
# ----------------------------
ranked_df = merged_df[merged_df['Final Rank'].notna()].copy()
ranked_df['Final Rank'] = ranked_df['Final Rank'].astype(int)
ranked_df = pd.get_dummies(ranked_df, columns=['Conference'])
merged_df = pd.get_dummies(merged_df, columns=['Conference'])

conf_cols = [col for col in ranked_df.columns if col.startswith('Conference_')]
for col in conf_cols:
    if col not in merged_df.columns:
        merged_df[col] = 0

merged_df = merged_df.reindex(columns=ranked_df.columns, fill_value=0)

X = ranked_df.drop(columns=['Team', 'Final Rank', 'year'])
y = ranked_df['Final Rank']
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

features = merged_df.drop(columns=['Team', 'Final Rank', 'year'], errors='ignore')
features = features.apply(pd.to_numeric, errors='coerce').fillna(0)
merged_df['Predicted Rank Score'] = model.predict(features)
merged_df['Predicted Rank'] = merged_df.groupby('year')['Predicted Rank Score'].rank(method='first', ascending=True).astype(int)
merged_df['Actual Rank'] = merged_df['Final Rank'].astype(pd.Int64Dtype())

final_output = merged_df[['year', 'Team', 'Predicted Rank', 'Actual Rank']].sort_values(['year', 'Predicted Rank'])
final_output.to_csv("predicted_vs_actual_rankings.csv", index=False)

print("\n✅ Sample predictions:")
print(final_output.head(10))

# ----------------------------
# 5. RFE Feature Selection + Comparison
# ----------------------------
X_full = ranked_df.drop(columns=['Team', 'Final Rank', 'year'], errors='ignore').apply(pd.to_numeric, errors='coerce').fillna(0)
y_full = ranked_df['Final Rank']

rfe = RFE(RandomForestRegressor(n_estimators=100, random_state=42), n_features_to_select=30, step=10)
rfe.fit(X_full, y_full)
selected_features = X_full.columns[rfe.support_]

X_rfe = X_full[selected_features]
X_train_rfe, X_test_rfe, y_train_rfe, y_test_rfe = train_test_split(X_rfe, y_full, test_size=0.2, random_state=42)
rfe_model = RandomForestRegressor(n_estimators=100, random_state=42)
rfe_model.fit(X_train_rfe, y_train_rfe)

merged_df['Predicted Rank Score (RFE)'] = rfe_model.predict(features[selected_features])
merged_df['Predicted Rank (RFE)'] = merged_df.groupby('year')['Predicted Rank Score (RFE)'].rank(method='first', ascending=True).astype(int)

output_rfe = merged_df[['year', 'Team', 'Predicted Rank (RFE)', 'Actual Rank']].sort_values(['year', 'Predicted Rank (RFE)'])
output_rfe.to_csv("predicted_vs_actual_rankings_rfe.csv", index=False)

# ----------------------------
# 6. RMSE Evaluation & Visualization
# ----------------------------
y_pred = model.predict(X_test)
y_pred_rfe = rfe_model.predict(X_test_rfe)
rmse_baseline = mean_squared_error(y_test, y_pred, squared=False)
rmse_rfe = mean_squared_error(y_test_rfe, y_pred_rfe, squared=False)

print(f"\n📉 RMSE (Full Model): {rmse_baseline:.3f}")
print(f"📉 RMSE (RFE Model): {rmse_rfe:.3f}")

# Plot comparison
plt.figure(figsize=(6, 4))
plt.bar(['Full Model', 'RFE Model'], [rmse_baseline, rmse_rfe], color=['skyblue', 'orange'])
plt.ylabel('RMSE')
plt.title('Model Performance Comparison')
plt.tight_layout()
plt.show()
