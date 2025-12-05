"""
K-League Pass Prediction - Data Preprocessing and Preparation
==============================================================
This script prepares the engineered features for modeling by:
1. Handling missing values
2. Scaling/normalizing features
3. Creating train/validation splits
4. Preparing sequence data for modeling
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("K-LEAGUE PASS PREDICTION - DATA PREPROCESSING")
print("="*80)

# ============================================================================
# 1. LOAD ENGINEERED FEATURES
# ============================================================================
print("\n[1] Loading engineered features...")
df = pd.read_csv('./processed_data/train_with_features.csv')
print(f"[OK] Loaded {len(df):,} events with {len(df.columns)} features")

# ============================================================================
# 2. DEFINE TARGET AND FEATURE COLUMNS
# ============================================================================
print("\n[2] Defining target and feature columns...")

# Target variables (what we want to predict)
target_cols = ['end_x', 'end_y']

# Columns to exclude from features
exclude_cols = [
    'game_id', 'period_id', 'episode_id', 'team_id', 'player_id', 'action_id',
    'type_name', 'result_name', 'game_episode',
    'end_x', 'end_y',  # Target variables
    'prev_end_x', 'prev_end_y'  # These are derived from target
]

# Feature columns
feature_cols = [col for col in df.columns if col not in exclude_cols]
print(f"[OK] Target columns: {target_cols}")
print(f"[OK] Feature columns: {len(feature_cols)}")

# ============================================================================
# 3. HANDLE MISSING VALUES
# ============================================================================
print("\n[3] Handling missing values...")

# Check for missing values
missing_counts = df[feature_cols].isnull().sum()
missing_features = missing_counts[missing_counts > 0]

if len(missing_features) > 0:
    print(f"[WARNING] Found missing values in {len(missing_features)} features:")
    for feat, count in missing_features.items():
        print(f"  {feat}: {count} ({count/len(df)*100:.2f}%)")
    
    # Fill missing values with appropriate strategies
    for col in missing_features.index:
        if 'std' in col:
            # Standard deviation features: fill with 0
            df[col].fillna(0, inplace=True)
        elif 'count' in col:
            # Count features: fill with 0
            df[col].fillna(0, inplace=True)
        else:
            # Other features: fill with median
            df[col].fillna(df[col].median(), inplace=True)
    
    print("[OK] Missing values handled")
else:
    print("[OK] No missing values found")

# ============================================================================
# 4. CREATE EPISODE-LEVEL DATASET
# ============================================================================
print("\n[4] Creating episode-level dataset...")

# Group by episode and get the last event (which contains the final pass target)
episode_targets = df.groupby('game_episode').tail(1)[['game_episode', 'end_x', 'end_y']].copy()
episode_targets.columns = ['game_episode', 'target_x', 'target_y']

print(f"[OK] Created targets for {len(episode_targets):,} episodes")

# ============================================================================
# 5. CREATE SEQUENCE FEATURES
# ============================================================================
print("\n[5] Creating sequence-level aggregated features...")

# For each episode, create aggregated features from the sequence
agg_features = df.groupby('game_episode').agg({
    # Spatial aggregations
    'pass_distance': ['mean', 'std', 'max', 'min', 'sum'],
    'pass_angle': ['mean', 'std'],
    'forward_progress': ['mean', 'std', 'sum'],
    'lateral_movement': ['mean', 'sum'],
    'start_dist_to_goal': ['mean', 'min'],
    'start_x': ['mean', 'std', 'first', 'last'],
    'start_y': ['mean', 'std', 'first', 'last'],
    
    # Temporal aggregations
    'seq_total_length': 'first',
    'time_diff': ['mean', 'std'],
    'time_since_start': 'max',
    
    # Contextual aggregations
    'same_team_as_prev': 'mean',
    'same_player_as_prev': 'mean',
    'action_type_encoded': lambda x: x.mode()[0] if len(x) > 0 else 0,
    
    # Zone aggregations
    'start_zone': lambda x: x.mode()[0] if len(x) > 0 else 0,
    'end_zone': lambda x: x.mode()[0] if len(x) > 0 else 0,
}).reset_index()

# Flatten column names
agg_features.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                        for col in agg_features.columns.values]

print(f"[OK] Created {len(agg_features.columns)-1} aggregated features")

# Merge with targets
episode_data = agg_features.merge(episode_targets, on='game_episode', how='left')
print(f"[OK] Episode dataset shape: {episode_data.shape}")

# ============================================================================
# 6. TRAIN/VALIDATION SPLIT
# ============================================================================
print("\n[6] Creating train/validation split...")

# Split by game_id to ensure no data leakage
# Extract game_id from game_episode
episode_data['game_id'] = episode_data['game_episode'].str.split('_').str[0].astype(int)

# Get unique games
unique_games = episode_data['game_id'].unique()
print(f"[INFO] Total unique games: {len(unique_games)}")

# Split games into train and validation (80/20)
train_games, val_games = train_test_split(
    unique_games, 
    test_size=0.2, 
    random_state=42
)

# Create train and validation sets
train_data = episode_data[episode_data['game_id'].isin(train_games)].copy()
val_data = episode_data[episode_data['game_id'].isin(val_games)].copy()

print(f"[OK] Train set: {len(train_data):,} episodes from {len(train_games)} games")
print(f"[OK] Validation set: {len(val_data):,} episodes from {len(val_games)} games")

# ============================================================================
# 7. FEATURE SCALING
# ============================================================================
print("\n[7] Scaling features...")

# Columns to exclude from scaling
non_feature_cols = ['game_episode', 'game_id', 'target_x', 'target_y']
scale_cols = [col for col in episode_data.columns if col not in non_feature_cols]

# Use RobustScaler (less sensitive to outliers)
scaler = RobustScaler()

# Fit on training data only
train_data[scale_cols] = scaler.fit_transform(train_data[scale_cols])
val_data[scale_cols] = scaler.transform(val_data[scale_cols])

# Fill any remaining NaN values (can occur with std=0 features)
train_data[scale_cols] = train_data[scale_cols].fillna(0)
val_data[scale_cols] = val_data[scale_cols].fillna(0)

print(f"[OK] Scaled {len(scale_cols)} features using RobustScaler")

# ============================================================================
# 8. SAVE PROCESSED DATA
# ============================================================================
print("\n[8] Saving processed data...")

OUTPUT_DIR = Path('./processed_data')
OUTPUT_DIR.mkdir(exist_ok=True)

# Save train and validation sets
train_data.to_csv(OUTPUT_DIR / 'train_processed.csv', index=False)
val_data.to_csv(OUTPUT_DIR / 'val_processed.csv', index=False)
print(f"[OK] Saved: {OUTPUT_DIR / 'train_processed.csv'}")
print(f"[OK] Saved: {OUTPUT_DIR / 'val_processed.csv'}")

# Save scaler
with open(OUTPUT_DIR / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(f"[OK] Saved: {OUTPUT_DIR / 'scaler.pkl'}")

# Save feature column names
feature_info = {
    'scale_cols': scale_cols,
    'target_cols': target_cols,
    'non_feature_cols': non_feature_cols
}
with open(OUTPUT_DIR / 'feature_info.pkl', 'wb') as f:
    pickle.dump(feature_info, f)
print(f"[OK] Saved: {OUTPUT_DIR / 'feature_info.pkl'}")

# ============================================================================
# 9. SAVE FULL SEQUENCE DATA (FOR ADVANCED MODELS)
# ============================================================================
print("\n[9] Saving full sequence data...")

# Prepare sequence data (event-level, not aggregated)
sequence_data = df.copy()

# Add split indicator
sequence_data['split'] = 'train'
sequence_data.loc[sequence_data['game_episode'].str.split('_').str[0].astype(int).isin(val_games), 'split'] = 'val'

# Save
sequence_data.to_csv(OUTPUT_DIR / 'sequence_data_full.csv', index=False)
print(f"[OK] Saved: {OUTPUT_DIR / 'sequence_data_full.csv'}")
print(f"    Shape: {sequence_data.shape}")

# ============================================================================
# 10. SUMMARY STATISTICS
# ============================================================================
print("\n[10] Summary statistics...")

print("\nDataset Statistics:")
print(f"  Total episodes: {len(episode_data):,}")
print(f"  Training episodes: {len(train_data):,} ({len(train_data)/len(episode_data)*100:.1f}%)")
print(f"  Validation episodes: {len(val_data):,} ({len(val_data)/len(episode_data)*100:.1f}%)")

print("\nTarget Statistics (Training Set):")
print(f"  Target X - Mean: {train_data['target_x'].mean():.2f}, Std: {train_data['target_x'].std():.2f}")
print(f"  Target Y - Mean: {train_data['target_y'].mean():.2f}, Std: {train_data['target_y'].std():.2f}")

print("\nFeature Statistics:")
print(f"  Total features: {len(scale_cols)}")
print(f"  Feature types: aggregated sequence features")

# ============================================================================
# 11. DATA QUALITY CHECKS
# ============================================================================
print("\n[11] Data quality checks...")

# Check for NaN values
train_nan = train_data[scale_cols].isnull().sum().sum()
val_nan = val_data[scale_cols].isnull().sum().sum()

if train_nan > 0 or val_nan > 0:
    print(f"[WARNING] Found NaN values - Train: {train_nan}, Val: {val_nan}")
else:
    print("[OK] No NaN values in processed data")

# Check for infinite values
train_inf = np.isinf(train_data[scale_cols]).sum().sum()
val_inf = np.isinf(val_data[scale_cols]).sum().sum()

if train_inf > 0 or val_inf > 0:
    print(f"[WARNING] Found infinite values - Train: {train_inf}, Val: {val_inf}")
else:
    print("[OK] No infinite values in processed data")

# Check target ranges
print("\nTarget Range Checks:")
print(f"  Train X range: [{train_data['target_x'].min():.2f}, {train_data['target_x'].max():.2f}]")
print(f"  Train Y range: [{train_data['target_y'].min():.2f}, {train_data['target_y'].max():.2f}]")
print(f"  Val X range: [{val_data['target_x'].min():.2f}, {val_data['target_x'].max():.2f}]")
print(f"  Val Y range: [{val_data['target_y'].min():.2f}, {val_data['target_y'].max():.2f}]")

print("\n" + "="*80)
print("DATA PREPROCESSING COMPLETE!")
print(f"All outputs saved to: {OUTPUT_DIR.absolute()}")
print("\nReady for modeling!")
print("="*80)
