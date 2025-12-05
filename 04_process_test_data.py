"""
K-League Pass Prediction - Test Data Processing
================================================
This script processes test data using the same pipeline as training data.
It loads test episodes, applies feature engineering, and prepares for prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("K-LEAGUE PASS PREDICTION - TEST DATA PROCESSING")
print("="*80)

# Field dimensions
FIELD_LENGTH = 105
FIELD_WIDTH = 68

# ============================================================================
# 1. LOAD TEST METADATA
# ============================================================================
print("\n[1] Loading test metadata...")
df_test = pd.read_csv('test.csv')
print(f"[OK] Found {len(df_test):,} test episodes")

# ============================================================================
# 2. LOAD SCALER AND FEATURE INFO
# ============================================================================
print("\n[2] Loading preprocessing artifacts...")
with open('./processed_data/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("[OK] Loaded scaler")

with open('./processed_data/feature_info.pkl', 'rb') as f:
    feature_info = pickle.load(f)
print("[OK] Loaded feature info")

# ============================================================================
# 3. DEFINE FEATURE ENGINEERING FUNCTIONS
# ============================================================================
print("\n[3] Defining feature engineering functions...")

def engineer_features(df):
    """Apply all feature engineering transformations"""
    
    # Spatial features
    df['pass_distance'] = np.sqrt(
        (df['end_x'] - df['start_x'])**2 + 
        (df['end_y'] - df['start_y'])**2
    )
    
    df['pass_angle'] = np.arctan2(
        df['end_y'] - df['start_y'],
        df['end_x'] - df['start_x']
    ) * 180 / np.pi
    
    df['start_dist_to_goal'] = np.sqrt(
        (FIELD_LENGTH - df['start_x'])**2 + 
        (FIELD_WIDTH/2 - df['start_y'])**2
    )
    
    df['end_dist_to_goal'] = np.sqrt(
        (FIELD_LENGTH - df['end_x'])**2 + 
        (FIELD_WIDTH/2 - df['end_y'])**2
    )
    
    df['start_dist_to_center_y'] = np.abs(df['start_y'] - FIELD_WIDTH/2)
    df['end_dist_to_center_y'] = np.abs(df['end_y'] - FIELD_WIDTH/2)
    
    # Zone features
    df['start_zone_x'] = pd.cut(df['start_x'], bins=[0, FIELD_LENGTH/3, 2*FIELD_LENGTH/3, FIELD_LENGTH], 
                                 labels=[0, 1, 2], include_lowest=True).astype(int)
    df['start_zone_y'] = pd.cut(df['start_y'], bins=[0, FIELD_WIDTH/3, 2*FIELD_WIDTH/3, FIELD_WIDTH], 
                                 labels=[0, 1, 2], include_lowest=True).astype(int)
    df['end_zone_x'] = pd.cut(df['end_x'], bins=[0, FIELD_LENGTH/3, 2*FIELD_LENGTH/3, FIELD_LENGTH], 
                               labels=[0, 1, 2], include_lowest=True).astype(int)
    df['end_zone_y'] = pd.cut(df['end_y'], bins=[0, FIELD_WIDTH/3, 2*FIELD_WIDTH/3, FIELD_WIDTH], 
                               labels=[0, 1, 2], include_lowest=True).astype(int)
    
    df['start_zone'] = df['start_zone_x'] * 3 + df['start_zone_y']
    df['end_zone'] = df['end_zone_x'] * 3 + df['end_zone_y']
    
    # Normalized coordinates
    df['start_x_norm'] = df['start_x'] / FIELD_LENGTH
    df['start_y_norm'] = df['start_y'] / FIELD_WIDTH
    df['end_x_norm'] = df['end_x'] / FIELD_LENGTH
    df['end_y_norm'] = df['end_y'] / FIELD_WIDTH
    
    # Polar coordinates
    center_x, center_y = FIELD_LENGTH / 2, FIELD_WIDTH / 2
    df['start_radius'] = np.sqrt((df['start_x'] - center_x)**2 + (df['start_y'] - center_y)**2)
    df['start_theta'] = np.arctan2(df['start_y'] - center_y, df['start_x'] - center_x)
    df['end_radius'] = np.sqrt((df['end_x'] - center_x)**2 + (df['end_y'] - center_y)**2)
    df['end_theta'] = np.arctan2(df['end_y'] - center_y, df['end_x'] - center_x)
    
    # Movement features
    df['delta_x'] = df['end_x'] - df['start_x']
    df['delta_y'] = df['end_y'] - df['start_y']
    df['forward_progress'] = df['delta_x']
    df['lateral_movement'] = np.abs(df['delta_y'])
    
    # Temporal features
    df = df.sort_values(['game_episode', 'time_seconds']).reset_index(drop=True)
    df['seq_position'] = df.groupby('game_episode').cumcount()
    df['seq_total_length'] = df.groupby('game_episode')['game_episode'].transform('count')
    df['seq_position_pct'] = df['seq_position'] / df['seq_total_length']
    
    df['time_since_start'] = df.groupby('game_episode')['time_seconds'].transform(
        lambda x: x - x.iloc[0]
    )
    df['time_diff'] = df.groupby('game_episode')['time_seconds'].diff().fillna(0)
    
    df['cumsum_distance'] = df.groupby('game_episode')['pass_distance'].cumsum()
    df['cumsum_forward'] = df.groupby('game_episode')['forward_progress'].cumsum()
    
    # Rolling features
    for window in [3, 5]:
        df[f'rolling_distance_mean_{window}'] = df.groupby('game_episode')['pass_distance'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f'rolling_forward_mean_{window}'] = df.groupby('game_episode')['forward_progress'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
    
    # Contextual features
    action_type_map = {action: idx for idx, action in enumerate(df['type_name'].unique())}
    df['action_type_encoded'] = df['type_name'].map(action_type_map)
    
    df['result_encoded'] = df['result_name'].fillna('Unknown')
    result_map = {result: idx for idx, result in enumerate(df['result_encoded'].unique())}
    df['result_encoded'] = df['result_encoded'].map(result_map)
    
    df['prev_action_type'] = df.groupby('game_episode')['action_type_encoded'].shift(1).fillna(-1)
    df['prev_pass_distance'] = df.groupby('game_episode')['pass_distance'].shift(1).fillna(0)
    df['prev_end_x'] = df.groupby('game_episode')['end_x'].shift(1).fillna(df['start_x'])
    df['prev_end_y'] = df.groupby('game_episode')['end_y'].shift(1).fillna(df['start_y'])
    
    df['dist_from_prev_end'] = np.sqrt(
        (df['start_x'] - df['prev_end_x'])**2 + 
        (df['start_y'] - df['prev_end_y'])**2
    )
    
    df['same_team_as_prev'] = (
        df.groupby('game_episode')['team_id'].shift(1) == df['team_id']
    ).astype(int).fillna(0)
    
    df['same_player_as_prev'] = (
        df.groupby('game_episode')['player_id'].shift(1) == df['player_id']
    ).astype(int).fillna(0)
    
    # Team and player features (use training statistics)
    # Load training data for statistics
    train_full = pd.read_csv('./processed_data/train_with_features.csv')
    
    team_stats = train_full.groupby('team_id').agg({
        'pass_distance': ['mean', 'std'],
        'forward_progress': ['mean', 'std'],
        'end_x': 'mean',
        'end_y': 'mean'
    }).reset_index()
    team_stats.columns = ['team_id', 'team_avg_pass_dist', 'team_std_pass_dist',
                          'team_avg_forward', 'team_std_forward',
                          'team_avg_end_x', 'team_avg_end_y']
    df = df.merge(team_stats, on='team_id', how='left')
    
    player_stats = train_full.groupby('player_id').agg({
        'pass_distance': ['mean', 'std', 'count'],
        'forward_progress': 'mean',
        'end_x': 'mean',
        'end_y': 'mean'
    }).reset_index()
    player_stats.columns = ['player_id', 'player_avg_pass_dist', 'player_std_pass_dist',
                            'player_event_count', 'player_avg_forward',
                            'player_avg_end_x', 'player_avg_end_y']
    df = df.merge(player_stats, on='player_id', how='left')
    
    # Fill missing player/team stats with global means
    for col in ['team_avg_pass_dist', 'team_std_pass_dist', 'team_avg_forward', 
                'team_std_forward', 'team_avg_end_x', 'team_avg_end_y',
                'player_avg_pass_dist', 'player_std_pass_dist', 'player_event_count',
                'player_avg_forward', 'player_avg_end_x', 'player_avg_end_y']:
        if col in df.columns:
            df[col].fillna(df[col].mean(), inplace=True)
    
    df['pass_dist_vs_player_avg'] = df['pass_distance'] - df['player_avg_pass_dist']
    df['forward_vs_player_avg'] = df['forward_progress'] - df['player_avg_forward']
    
    # Interaction features
    df['zone_change_x'] = df['end_zone_x'] - df['start_zone_x']
    df['zone_change_y'] = df['end_zone_y'] - df['start_zone_y']
    df['start_x_action'] = df['start_x_norm'] * df['action_type_encoded']
    df['start_y_action'] = df['start_y_norm'] * df['action_type_encoded']
    df['distance_x_zone'] = df['pass_distance'] * df['start_zone']
    df['time_x_position'] = df['seq_position_pct'] * df['start_x_norm']
    
    # Advanced spatial features
    df['dist_to_left_boundary'] = df['start_y']
    df['dist_to_right_boundary'] = FIELD_WIDTH - df['start_y']
    df['dist_to_back_boundary'] = df['start_x']
    df['dist_to_front_boundary'] = FIELD_LENGTH - df['start_x']
    df['min_boundary_dist'] = df[['dist_to_left_boundary', 'dist_to_right_boundary',
                                   'dist_to_back_boundary', 'dist_to_front_boundary']].min(axis=1)
    
    PENALTY_BOX_DEPTH = 16.5
    PENALTY_BOX_WIDTH = 40.3
    
    def in_penalty_box(x, y):
        return (x >= FIELD_LENGTH - PENALTY_BOX_DEPTH) and \
               (y >= (FIELD_WIDTH - PENALTY_BOX_WIDTH) / 2) and \
               (y <= (FIELD_WIDTH + PENALTY_BOX_WIDTH) / 2)
    
    df['start_in_penalty_box'] = df.apply(lambda row: in_penalty_box(row['start_x'], row['start_y']), axis=1).astype(int)
    df['end_in_penalty_box'] = df.apply(lambda row: in_penalty_box(row['end_x'], row['end_y']), axis=1).astype(int)
    
    return df

def create_aggregated_features(df):
    """Create episode-level aggregated features"""
    
    agg_features = df.groupby('game_episode').agg({
        'pass_distance': ['mean', 'std', 'max', 'min', 'sum'],
        'pass_angle': ['mean', 'std'],
        'forward_progress': ['mean', 'std', 'sum'],
        'lateral_movement': ['mean', 'sum'],
        'start_dist_to_goal': ['mean', 'min'],
        'start_x': ['mean', 'std', 'first', 'last'],
        'start_y': ['mean', 'std', 'first', 'last'],
        'seq_total_length': 'first',
        'time_diff': ['mean', 'std'],
        'time_since_start': 'max',
        'same_team_as_prev': 'mean',
        'same_player_as_prev': 'mean',
        'action_type_encoded': lambda x: x.mode()[0] if len(x) > 0 else 0,
        'start_zone': lambda x: x.mode()[0] if len(x) > 0 else 0,
        'end_zone': lambda x: x.mode()[0] if len(x) > 0 else 0,
    }).reset_index()
    
    agg_features.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                            for col in agg_features.columns.values]
    
    return agg_features

print("[OK] Feature engineering functions defined")

# ============================================================================
# 4. PROCESS TEST EPISODES
# ============================================================================
print("\n[4] Processing test episodes...")

all_test_data = []

for idx, row in df_test.iterrows():
    if (idx + 1) % 100 == 0:
        print(f"  Processing episode {idx + 1}/{len(df_test)}...")
    
    # Load episode data
    episode_path = row['path']
    episode_df = pd.read_csv(episode_path)
    episode_df['game_episode'] = row['game_episode']
    
    all_test_data.append(episode_df)

# Combine all episodes
test_full = pd.concat(all_test_data, ignore_index=True)
print(f"[OK] Loaded {len(test_full):,} events from {len(df_test):,} episodes")

# ============================================================================
# 5. APPLY FEATURE ENGINEERING
# ============================================================================
print("\n[5] Applying feature engineering...")
test_full = engineer_features(test_full)
print(f"[OK] Created features, shape: {test_full.shape}")

# ============================================================================
# 6. CREATE AGGREGATED FEATURES
# ============================================================================
print("\n[6] Creating aggregated features...")
test_agg = create_aggregated_features(test_full)
print(f"[OK] Aggregated features shape: {test_agg.shape}")

# ============================================================================
# 7. APPLY SCALING
# ============================================================================
print("\n[7] Applying scaling...")

# Get scale columns (same as training)
scale_cols = feature_info['scale_cols']

# Ensure all required columns exist
missing_cols = set(scale_cols) - set(test_agg.columns)
if missing_cols:
    print(f"[WARNING] Missing columns: {missing_cols}")
    for col in missing_cols:
        test_agg[col] = 0

# Apply scaler
test_agg[scale_cols] = scaler.transform(test_agg[scale_cols])
test_agg[scale_cols] = test_agg[scale_cols].fillna(0)

print(f"[OK] Scaled {len(scale_cols)} features")

# ============================================================================
# 8. SAVE PROCESSED TEST DATA
# ============================================================================
print("\n[8] Saving processed test data...")

OUTPUT_DIR = Path('./processed_data')
test_agg.to_csv(OUTPUT_DIR / 'test_processed.csv', index=False)
print(f"[OK] Saved: {OUTPUT_DIR / 'test_processed.csv'}")

# Also save full sequence data
test_full.to_csv(OUTPUT_DIR / 'test_sequence_full.csv', index=False)
print(f"[OK] Saved: {OUTPUT_DIR / 'test_sequence_full.csv'}")

print("\n" + "="*80)
print("TEST DATA PROCESSING COMPLETE!")
print(f"Processed {len(test_agg):,} test episodes")
print(f"Ready for prediction!")
print("="*80)
