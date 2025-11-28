import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import numpy as np
from statsmodels.formula.api import ols
from scipy.stats import kruskal
import seaborn as sns


# Show all rows in the console
pd.set_option('display.max_rows', None)

# Reads CSV file
df = pd.read_csv("brawlstars_top200.csv")


# --- 1. Best Brawler Bar Chart ---
plt.figure(figsize=(8,5))
brawler_counts = df['BestBrawler'].value_counts()
brawler_names = [b.title() for b in brawler_counts.index]
plt.bar(brawler_names, brawler_counts.values, color='lightcoral')
plt.title("Most Common Best Brawler")
plt.xlabel("Brawler")
plt.ylabel("Count")
plt.xticks(rotation=90, ha='right')
plt.tight_layout()
plt.savefig("chart_best_brawler.png", dpi=300)
plt.close()
print()

# --- 1.2 Best Brawler Type Bar Chart ---
plt.figure(figsize=(8,5))
df['BestBrawlerType'].value_counts().plot(kind='bar', color='lightcoral')
plt.title("Most Common Best Brawler Type")
plt.xlabel("Brawler Type")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("chart_best_brawler_type.png", dpi=300)
plt.close()
print()


# Average trophies by best brawler
avg_trophies_brawler = df.groupby('BestBrawler')['Trophies'].mean().sort_values(ascending=False)
# Average trophies by brawler type
avg_trophies_type = df.groupby('BestBrawlerType')['Trophies'].mean().sort_values(ascending=False)


# --- 1.3 Best Brawler Type Average Bar Chart ---
avg_trophies_type.plot(kind='bar', color='lightgreen', figsize=(10,5))
plt.title("Average Trophies by Best Brawler Type")
plt.xlabel("Brawler Type")
plt.ylabel("Average Trophies")
plt.ylim(100000, 130000)
plt.tight_layout()
plt.savefig("avg_trophies_best_brawler_type.png", dpi=300)
plt.close()
print()


# --- 1.4 Best Brawler Average Bar Chart ---
avg_trophies_brawler.plot(kind='bar', color='lightgreen', figsize=(10,5))
plt.title("Average Trophies by Best Brawler")
plt.xlabel("Brawler")
plt.ylabel("Average Trophies")
plt.ylim(100000, 130000)
plt.tight_layout()
plt.savefig("avg_trophies_best_brawler.png", dpi=300)
plt.close()
print()

print("ANOVA Models")
# --- 1.5 Best Brawler ANOVA ---
model = ols("Trophies ~ C(BestBrawler)", data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table["eta_sq"] = anova_table["sum_sq"] / anova_table["sum_sq"].sum()
print(anova_table)
print()


# --- 1.6 Best Brawler Type Bar ANOVA ---
model = ols("Trophies ~ C(BestBrawlerType)", data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table["eta_sq"] = anova_table["sum_sq"] / anova_table["sum_sq"].sum()
print(anova_table)
print()


# --- 1.7 Kruskal Wallis Test and Epsilion Squared ---
print("Kruskal Wallis Test:")
def epsilon_squared(H, N):
    return (H - len(groups) + 1) / (N - len(groups))
# --- Best Brawler ---
groups = [group['Trophies'].values for name, group in df.groupby('BestBrawler')]
H_brawler, p_brawler = kruskal(*groups)
N = len(df)
eta_sq_brawler = epsilon_squared(H_brawler, N)
print("Best Brawler:")
print("Kruskal-Wallis H:", H_brawler)
print("p-value:", p_brawler)
print("Epsilon squared:", eta_sq_brawler)
# --- Best Brawler Type ---
groups = [group['Trophies'].values for name, group in df.groupby('BestBrawlerType')]
H_type, p_type = kruskal(*groups)
N = len(df)
eta_sq_type = epsilon_squared(H_type, N)
print("\nBest Brawler Type:")
print("Kruskal-Wallis H:", H_type)
print("p-value:", p_type)
print("Epsilon squared:", eta_sq_type)
print()


# --- 1.8 Borda Count ---
# Total number of players
n_players = len(df)
# Calculate Borda points for each player
df['BordaPoints'] = n_players - df['Rank']
# Aggregate points by Best Brawler Type
type_points = df.groupby('BestBrawlerType')['BordaPoints'].sum().sort_values(ascending=False)
print("\nBorda points by Best Brawler Type:\n", type_points)
print()


# --- 2. 3v3 Victories vs Trophies Scatter Plot ---
plt.figure(figsize=(8,5))
plt.scatter(df['3v3Victories'], df['Trophies'], alpha=0.6, color='orange')
plt.title("3v3 vs Trophies")
plt.xlabel("3v3 Wins")
plt.ylabel("Trophies")
plt.tight_layout()
plt.savefig("chart_3v3_vs_trophies_scatter.png", dpi=300)
plt.close()


# --- 2.1 Linear Regression: 3v3 Victories vs 3v3 Trophies ---
X = df[['3v3Victories']].values  # must be 2D
y = df['Trophies'].values
lr = LinearRegression()
lr.fit(X, y)
print("Slope:", lr.coef_[0])
print("Intercept:", lr.intercept_)
print("R²:", lr.score(X, y))
plt.scatter(X, y, alpha=0.6, color='lightblue')
plt.plot(X, lr.predict(X), color='red', label=f'Best Fit Line (R²={lr.score(X,y):.2f})')
plt.xlabel("3v3 Wins")
plt.ylabel("Trophies")
plt.legend()
plt.title("3v3 Wins vs Trophies Regression Line")
plt.tight_layout()
plt.savefig("chart_3v3_vs_trophies_regression_line.png", dpi=300)
plt.close()


# --- 2.2 Duo Wins vs Trophies Scatter Plot ---
plt.figure(figsize=(8,5))
plt.scatter(df['DuoVictories'], df['Trophies'], alpha=0.6, color='orange')
plt.title("Duo Wins vs Trophies")
plt.xlabel("Duo Wins")
plt.ylabel("Trophies")
plt.tight_layout()
plt.savefig("chart_duo_vs_trophies_scatter.png", dpi=300)
plt.close()


# --- 2.3 Linear Regression: Duo Wins vs Trophies ---
X = df[['DuoVictories']].values  # must be 2D
y = df['Trophies'].values
lr = LinearRegression()
lr.fit(X, y)
print("Slope:", lr.coef_[0])
print("Intercept:", lr.intercept_)
print("R²:", lr.score(X, y))
plt.scatter(X, y, alpha=0.6, color='lightblue')
plt.plot(X, lr.predict(X), color='red', label=f'Best Fit Line (R²={lr.score(X,y):.2f})')
plt.xlabel("Duo Wins")
plt.ylabel("Trophies")
plt.legend()
plt.title("Duo Wins vs Trophies Regression Line")
plt.tight_layout()
plt.savefig("chart_duo_vs_trophies_regression_line.png", dpi=300)
plt.close()


# --- 2.4 Solo Wins vs Trophies Scatter Plot ---
plt.figure(figsize=(8,5))
plt.scatter(df['SoloVictories'], df['Trophies'], alpha=0.6, color='orange')
plt.title("Solo Wins vs Trophies")
plt.xlabel("Solo Wins")
plt.ylabel("Trophies")
plt.tight_layout()
plt.savefig("chart_solo_vs_trophies_scatter.png", dpi=300)
plt.close()


# --- 2.5 Linear Regression: Solo Wins vs Trophies ---
X = df[['SoloVictories']].values  # must be 2D
y = df['Trophies'].values
lr = LinearRegression()
lr.fit(X, y)
print("Slope:", lr.coef_[0])
print("Intercept:", lr.intercept_)
print("R²:", lr.score(X, y))
plt.scatter(X, y, alpha=0.6, color='lightblue')
plt.plot(X, lr.predict(X), color='red', label=f'Best Fit Line (R²={lr.score(X,y):.2f})')
plt.xlabel("Solo Wins")
plt.ylabel("Trophies")
plt.legend()
plt.title("Solo Wins vs Trophies Regression Line")
plt.tight_layout()
plt.savefig("chart_solo_vs_trophies_regression_line.png", dpi=300)
plt.close()


# --- 2.6 OLS (3v3, Duos, Solo) ---
# X = predictors, y = dependent (OTS)
X = df[['3v3Victories', 'SoloVictories', 'DuoVictories']]
y = df['Trophies']
# Add constant for intercept
X = sm.add_constant(X)
# Fit OLS model
model = sm.OLS(y, X).fit()
print(model.summary())
print()

# --- 2.7 Permutation Importance (3v3, Duos, Solo) ---
print("Permutation Importance:")
# Load the dataset (Permutations)
df = pd.read_csv("brawlstars_top200.csv")
# Features (predictors)
X = df[['3v3Victories', 'SoloVictories', 'DuoVictories']]
# Target (outcome)
y = df['Trophies']
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train model
model = LinearRegression()
model.fit(X_train, y_train)
result = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std
}).sort_values(by="importance_mean", ascending=False)
print(importance_df)
print()


# --- 3 Stratification ---
df["TotalMatches"] = df["SoloVictories"] + df["DuoVictories"] + df["3v3Victories"]
# Create trophy bins (strata)
bins = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000]  # adjust ranges if needed
labels = ['0–5k', '5k -10k', '10k-15k', '15k-20k', '20k-25k', '25k-30k', '30k-35k', '35k-40k', '40k-45k', '45k-50k', '50k-55k', '55k-60k', '60k-65k', '65k-70k', '70k-75k', '75k-80k', '80k-85k', '85k-90k', '90k-95k', '95k-100k']
df["MatchGroup"] = pd.cut(df["TotalMatches"], bins=bins, labels=labels, include_lowest=True)
# Count players per group
group_counts = df["MatchGroup"].value_counts().sort_index()
# Plot
plt.figure(figsize=(10, 5))
group_counts.plot(kind="bar")
plt.title("Number of Players by Total Matches Won")
plt.xlabel("Total Matches Won (Range)")
plt.ylabel("Number of Players")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("match_groups_distribution.png", dpi=300)
plt.close()


# --- 3.1 Calculate ratios for all players ---
df['TotalMatches'] = df['SoloVictories'] + df['DuoVictories'] + df['3v3Victories']
df = df[df['TotalMatches'] > 0]  # remove any with zero total matches
df['solo_ratio'] = df['SoloVictories'] / df['TotalMatches']
df['duo_ratio'] = df['DuoVictories'] / df['TotalMatches']
df['3v3_ratio'] = df['3v3Victories'] / df['TotalMatches']
df['solo_3v3_ratio'] = df['SoloVictories'] / df['3v3Victories']


# --- 3.2 Filter players by 5k won matches intervals ---
group_df = df[(df["TotalMatches"] >= 20000) & (df["TotalMatches"] <= 25000)]
firstgroup_df = df[(df["TotalMatches"] >= 25000) & (df["TotalMatches"] <= 30000)]
secondgroup_df = df[(df["TotalMatches"] >= 30000) & (df["TotalMatches"] <= 35000)]
thirdgroup_df = df[(df["TotalMatches"] >= 35000) & (df["TotalMatches"] <= 40000)]
fourthgroup_df = df[(df["TotalMatches"] >= 40000) & (df["TotalMatches"] <= 45000)]
fifthgroup_df = df[(df["TotalMatches"] >= 45000) & (df["TotalMatches"] <= 50000)]


# --- 3.3 Linear Regression: Solo to 3v3 Wins Ratio vs Trophies ---
X = group_df[['solo_3v3_ratio']].values
y = group_df['Trophies'].values
lr = LinearRegression()
lr.fit(X, y)
plt.figure(figsize=(8,5))
plt.scatter(X, y, alpha=0.6, color='green')
plt.plot(X, lr.predict(X), color='red', label=f'Best Fit Line (R²={lr.score(X,y):.2f})')
plt.xlabel("Solo to 3v3 Wins Ratio")
plt.ylabel("Trophies")
plt.legend()
plt.title("Solo to 3v3 Wins Ratio vs Total Wins Regression Line (20k-25k Wins)")
plt.tight_layout()
plt.savefig("chart_0stratified_3v3_ratio_vs_trophies_regression_line.png", dpi=300)
plt.close()
print("Slope:", lr.coef_[0])
print("Intercept:", lr.intercept_)
print("R²:", lr.score(X, y))
print()

X = firstgroup_df[['solo_3v3_ratio']].values
y = firstgroup_df['Trophies'].values
lr = LinearRegression()
lr.fit(X, y)
plt.figure(figsize=(8,5))
plt.scatter(X, y, alpha=0.6, color='green')
plt.plot(X, lr.predict(X), color='red', label=f'Best Fit Line (R²={lr.score(X,y):.2f})')
plt.xlabel("Solo to 3v3 Wins Ratio")
plt.ylabel("Trophies")
plt.legend()
plt.title("Solo to 3v3 Wins Ratio vs Total Wins Regression Line (25k-30k Wins)")
plt.tight_layout()
plt.savefig("chart_1stratified_3v3_ratio_vs_trophies_regression_line.png", dpi=300)
plt.close()
print("Slope:", lr.coef_[0])
print("Intercept:", lr.intercept_)
print("R²:", lr.score(X, y))
print()

X = secondgroup_df[['solo_3v3_ratio']].values
y = secondgroup_df['Trophies'].values
lr = LinearRegression()
lr.fit(X, y)
plt.figure(figsize=(8,5))
plt.scatter(X, y, alpha=0.6, color='green')
plt.plot(X, lr.predict(X), color='red', label=f'Best Fit Line (R²={lr.score(X,y):.2f})')
plt.xlabel("Solo to 3v3 Wins Ratio")
plt.ylabel("Trophies")
plt.legend()
plt.title("Solo to 3v3 Wins Ratio vs Total Wins Regression Line (30k-35k Wins)")
plt.tight_layout()
plt.savefig("chart_2stratified_3v3_ratio_vs_trophies_regression_line.png", dpi=300)
plt.close()
print("Slope:", lr.coef_[0])
print("Intercept:", lr.intercept_)
print("R²:", lr.score(X, y))
print()

X = thirdgroup_df[['solo_3v3_ratio']].values
y = thirdgroup_df['Trophies'].values
lr = LinearRegression()
lr.fit(X, y)
plt.figure(figsize=(8,5))
plt.scatter(X, y, alpha=0.6, color='green')
plt.plot(X, lr.predict(X), color='red', label=f'Best Fit Line (R²={lr.score(X,y):.2f})')
plt.xlabel("Solo to 3v3 Wins Ratio")
plt.ylabel("Trophies")
plt.legend()
plt.title("Solo to 3v3 Wins Ratio vs Total Wins Regression Line (35k-40k Wins)")
plt.tight_layout()
plt.savefig("chart_3stratified_3v3_ratio_vs_trophies_regression_line.png", dpi=300)
plt.close()
print("Slope:", lr.coef_[0])
print("Intercept:", lr.intercept_)
print("R²:", lr.score(X, y))
print()

X = fourthgroup_df[['solo_3v3_ratio']].values
y = fourthgroup_df['Trophies'].values
lr = LinearRegression()
lr.fit(X, y)
plt.figure(figsize=(8,5))
plt.scatter(X, y, alpha=0.6, color='green')
plt.plot(X, lr.predict(X), color='red', label=f'Best Fit Line (R²={lr.score(X,y):.2f})')
plt.xlabel("Solo to 3v3 Wins Ratio")
plt.ylabel("Trophies")
plt.legend()
plt.title("Solo to 3v3 Wins Ratio vs Total Wins Regression Line (40k-45k Wins)")
plt.tight_layout()
plt.savefig("chart_4stratified_3v3_ratio_vs_trophies_regression_line.png", dpi=300)
plt.close()
print("Slope:", lr.coef_[0])
print("Intercept:", lr.intercept_)
print("R²:", lr.score(X, y))
print()

X = fifthgroup_df[['solo_3v3_ratio']].values
y = fifthgroup_df['Trophies'].values
lr = LinearRegression()
lr.fit(X, y)
plt.figure(figsize=(8,5))
plt.scatter(X, y, alpha=0.6, color='green')
plt.plot(X, lr.predict(X), color='red', label=f'Best Fit Line (R²={lr.score(X,y):.2f})')
plt.xlabel("Solo to 3v3 Wins Ratio")
plt.ylabel("Trophies")
plt.legend()
plt.title("Solo to 3v3 Wins Ratio vs Total Wins Regression Line (45k-50k Wins)")
plt.tight_layout()
plt.savefig("chart_5stratified_3v3_ratio_vs_trophies_regression_line.png", dpi=300)
plt.close()
print("Slope:", lr.coef_[0])
print("Intercept:", lr.intercept_)
print("R²:", lr.score(X, y))
print()


# --- 3.4 Linear Regression: Total Won Matches vs Trophies ---
X = df[['TotalMatches']].values  # predictor
y = df['Trophies'].values        # outcome
# Fit linear regression
lr = LinearRegression()
lr.fit(X, y)
# Print results
print("Slope:", lr.coef_[0])
print("Intercept:", lr.intercept_)
print("R²:", lr.score(X, y))
# Plot
plt.figure(figsize=(8,5))
plt.scatter(X, y, alpha=0.6, color='blue')
plt.plot(X, lr.predict(X), color='red', label=f'Best Fit Line (R²={lr.score(X,y):.2f})')
plt.xlabel("Total Won Matches")
plt.ylabel("Trophies")
plt.title("Total Won Matches vs Trophies")
plt.legend()
plt.tight_layout()
plt.savefig("total_won_matches_vs_trophies_regression.png", dpi=300)


print("✅ All charts created successfully!")

