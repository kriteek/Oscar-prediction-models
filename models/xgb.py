import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import xgboost as xgb

# Load TRAIN and TEST datasets
train_df = pd.read_csv("final_train_data_scores.csv")
test_df  = pd.read_csv("final_test_data_scores.csv")

# Fix column names if needed
for df in (train_df, test_df):
    if 'IMDB_rating' in df.columns and 'imdb_rating' not in df.columns:
        df.rename(columns={'IMDB_rating': 'imdb_rating'}, inplace=True)


# Calculate total nominations for each nominee (is_most_nominated)
for df in (train_df, test_df):
    df['nominations_count'] = df['bafta_nominated'] + df['sag_nominated'] + df['gg_nominated']
    
    # Flag nominees with most nominations in their category/year
    df['is_most_nominated'] = 0
    for (cat, year), group in df.groupby(['category', 'year_ceremony']):
        max_noms = group['nominations_count'].max()
        if max_noms > 0:  # Only set flag if at least one nomination exists
            df.loc[(df['category'] == cat) & (df['year_ceremony'] == year) & 
                   (df['nominations_count'] == max_noms), 'is_most_nominated'] = 1


# Add is_most_wins feature with SAG weighted more heavily. Calculate weighted wins for each nominee (SAG gets double weight)
for df in (train_df, test_df):
    # Use weighted sum for wins: SAG (2x), BAFTA (1x), Golden Globe (1x)
    df['weighted_wins'] = df['bafta_won'] + (2 * df['sag_won']) + df['gg_won']
    
    # Flag nominees with most wins in their category/year (only for Actor/Actress categories)
    df['is_most_wins'] = 0
    for (cat, year), group in df.groupby(['category', 'year_ceremony']):
        if cat in [1, 2]:  # Apply only to Best Actor (1) and Best Actress (2)
            max_wins = group['weighted_wins'].max()
            if max_wins > 0:  # Only set flag if at least one win exists
                df.loc[(df['category'] == cat) & (df['year_ceremony'] == year) & 
                       (df['weighted_wins'] == max_wins), 'is_most_wins'] = 1

# Add bafta_gg_director_win feature for Best Director category
for df in (train_df, test_df):
    # Create a feature for directors who won both BAFTA and Golden Globe
    df['bafta_gg_director_win'] = 0
    
    # Set to 1 for Best Director nominees who won both BAFTA and GG
    df.loc[(df['category'] == 3) & (df['bafta_won'] == 1) & (df['gg_won'] == 1), 'bafta_gg_director_win'] = 1
    
    # Set to 0.5 for those who won either BAFTA or GG (but not both)
    df.loc[(df['category'] == 3) & ((df['bafta_won'] == 1) | (df['gg_won'] == 1)) & (df['bafta_gg_director_win'] == 0), 'bafta_gg_director_win'] = 0.5

# Define features and targets
feature_cols = [
    'bafta_won', 'bafta_nominated',
    'sag_won', 'sag_nominated',
    'gg_won', 'gg_nominated',
    'imdb_rating', 'is_most_nominated',
    'pga_won', 'dga_won'
]

X_train = train_df[feature_cols]
y_train = train_df['oscar_won']

X_test = test_df[feature_cols]
y_test = test_df['oscar_won']  # Only if your test set has labels

# Define and train XGBoost model
xgb_model = xgb.XGBClassifier(
    use_label_encoder=False, 
    eval_metric='logloss',
    max_depth=4,
    learning_rate=0.05,
    n_estimators=200,
    scale_pos_weight=3,
    random_state=42
)

# Train model
print("Training XGBoost model...")
xgb_model.fit(X_train, y_train)

# Find optimal threshold on training data
y_train_prob = xgb_model.predict_proba(X_train)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_train, y_train_prob)

# Calculate F1 scores for different thresholds
f1_scores = []
for i in range(len(thresholds)):
    precision = precisions[i]
    recall = recalls[i]
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    f1_scores.append(f1)

# Find optimal threshold
best_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[best_idx]
print(f"\nOptimal threshold based on training F1: {optimal_threshold:.4f}")

# Predict on test set
y_prob = xgb_model.predict_proba(X_test)[:, 1]
y_pred = (y_prob > optimal_threshold).astype(int)

# Save predictions + movie_id + category
output_df = test_df[['movie_id', 'category', 'year_ceremony']].copy()  # Grab ID, category, year
for col in feature_cols:
    output_df[col] = X_test[col].values
output_df['actual'] = y_test.values
output_df['probability'] = y_prob
output_df['predicted'] = y_pred
output_df['is_most_wins'] = test_df['is_most_wins'].values  # Add the new feature to output_df
output_df['bafta_gg_director_win'] = test_df['bafta_gg_director_win'].values  # Add the new director feature

# ------------------------------------------------------------------------------
# SPECIAL HANDLING FOR CATEGORY 4 (BEST PICTURE) BASED ON IMDB RATING
# ------------------------------------------------------------------------------
# Define IMDB threshold for Best Picture
IMDB_THRESHOLD = 8.0 # You can adjust this value as needed

# Filter for Category 4 entries
cat4_mask = output_df['category'] == 4

# Boost probability for high IMDB ratings in Category 4
# This will increase the likelihood that high-rated films are selected as winners
output_df.loc[cat4_mask & (output_df['imdb_rating'] >= IMDB_THRESHOLD), 'probability'] *= 1.0
#pga
output_df.loc[cat4_mask & (output_df['pga_won'] == 1), 'probability'] *= 2.0

# ------------------------------------------------------------------------------
# SPECIAL HANDLING FOR CATEGORY 1 (BEST ACTOR) - ONLY SAG, IMDB, and IS_MOST_NOMINATED
# ------------------------------------------------------------------------------
# Filter for Category 1 entries (Best Actor)
cat1_mask = output_df['category'] == 1

# Create a new probability score for Best Actor based only on SAG, IMDB, and is_most_nominated
# First, reset all probabilities for Best Actor nominees
output_df.loc[cat1_mask, 'probability'] = 0.1  # Base probability

# Boost probability significantly for SAG winners
output_df.loc[cat1_mask & (output_df['sag_won'] == 1), 'probability'] += 0.5

# Add smaller boost for SAG nominations
output_df.loc[cat1_mask & (output_df['sag_nominated'] == 1), 'probability'] += 0.1

# Add boost for high IMDB ratings
output_df.loc[cat1_mask & (output_df['imdb_rating'] >= 7.5), 'probability'] += (output_df.loc[cat1_mask & (output_df['imdb_rating'] >= 7.5), 'imdb_rating'] - 7.5) * 0.1

# Add boost for most nominated
output_df.loc[cat1_mask & (output_df['is_most_nominated'] == 1), 'probability'] += 0.2

# Add significant boost for most wins (new feature)
output_df.loc[cat1_mask & (output_df['is_most_wins'] == 1), 'probability'] += 0.6

# ------------------------------------------------------------------------------
# SPECIAL HANDLING FOR CATEGORY 2 (BEST ACTRESS) - ADD IS_MOST_WINS BOOST
# ------------------------------------------------------------------------------
# Filter for Category 2 entries (Best Actress)
cat2_mask = output_df['category'] == 2

# Add significant boost for most wins (new feature)
output_df.loc[cat2_mask & (output_df['is_most_wins'] == 1), 'probability'] *= 1.0
# Boost probability significantly for SAG winners
#output_df.loc[cat2_mask & (output_df['sag_won'] == 1), 'probability'] += 0.2

# ------------------------------------------------------------------------------
# SPECIAL HANDLING FOR CATEGORY 3 (BEST DIRECTOR) - BAFTA+GG WINS BOOST
# ------------------------------------------------------------------------------
# Filter for Category 3 entries (Best Director)
cat3_mask = output_df['category'] == 3

# Apply a strong boost for directors who won both BAFTA and GG (value of 1)
output_df.loc[cat3_mask & (output_df['bafta_gg_director_win'] == 1), 'probability'] *= 2.0

# Apply a moderate boost for directors who won BAFTA (value of 0.5)
output_df.loc[cat3_mask & (output_df['bafta_won'] == 0.5), 'probability'] *= 0.5
#dga won
output_df.loc[cat3_mask & (output_df['dga_won'] == 1), 'probability'] *= 2.0

# ------------------------------------------------------------------------------
# Find predicted winners per (year_ceremony, category)
# ------------------------------------------------------------------------------
winner_df = (
    output_df
    .loc[output_df.groupby(['year_ceremony', 'category'])['probability'].idxmax()]
    .sort_values(by=['year_ceremony', 'category'])
    .reset_index(drop=True)
)

# Save winners with only the specified columns
winner_df[['movie_id', 'category', 'year_ceremony', 'actual', 'predicted', 'probability']].to_csv("predicted_oscar_winners_xgboost.csv", index=False)
print("\n🏆 Predicted winners saved to predicted_oscar_winners_xgboost.csv")

# ------------------------------------------------------------------------------
# Print TRAIN, TEST metrics
# ------------------------------------------------------------------------------
# --- Train metrics ---
y_train_pred = (y_train_prob > optimal_threshold).astype(int)

print("\n=== TRAIN classification report ===")
print(classification_report(y_train, y_train_pred))
print("Confusion Matrix (TRAIN):\n", confusion_matrix(y_train, y_train_pred))

# --- Test metrics ---
print("\n=== TEST classification report ===")
print(classification_report(y_test, y_pred))
print("Confusion Matrix (TEST):\n", confusion_matrix(y_test, y_pred))

# --- Category-specific metrics ---
print("\n=== Category-specific winner accuracy ===")
for category in sorted(winner_df['category'].unique()):
    cat_winners = winner_df[winner_df['category'] == category]
    if 'actual' in cat_winners.columns:
        cat_accuracy = (cat_winners['actual'] == 1).mean()
        correct = sum(cat_winners['actual'] == 1)
        total = len(cat_winners)
        print(f"Category {category}: {cat_accuracy:.4f} ({correct}/{total})")

# --- Overall winner accuracy ---
if 'actual' in winner_df.columns:
    group_acc = (winner_df['actual'] == 1).mean()
    correct = sum(winner_df['actual'] == 1)
    total = len(winner_df)
    print(f"\n🏆 Overall winner prediction accuracy: {group_acc:.4f} ({correct}/{total})")