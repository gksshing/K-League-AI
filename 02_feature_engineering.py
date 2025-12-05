"""
K-League Pass Prediction - Feature Engineering
===============================================
This script creates advanced features from the raw K-League dataset
for pass prediction modeling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("K-LEAGUE PASS PREDICTION - FEATURE ENGINEERING")
print("="*80)

# Field dimensions (FIFA standard)
FIELD_LENGTH = 105
FIELD_WIDTH = 68

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1] Loading training data...")
df = pd.read_csv('train.csv')
print(f"[OK] Loaded {len(df):,} events")

# ============================================================================
# 2. SPATIAL FEATURES
# ============================================================================
print("\n[2] Creating spatial features...")

# 2.1 Basic distance and direction
df['pass_distance'] = np.sqrt(
    (df['end_x'] - df['start_x'])**2 + 
    (df['end_y'] - df['start_y'])**2
)

df['pass_angle'] = np.arctan2(
    df['end_y'] - df['start_y'],
    df['end_x'] - df['start_x']
) * 180 / np.pi  # Convert to degrees

# 2.2 Distance from goal (assuming attacking towards x=105)
df['start_dist_to_goal'] = np.sqrt(
    (FIELD_LENGTH - df['start_x'])**2 + 
    (FIELD_WIDTH/2 - df['start_y'])**2
)

df['end_dist_to_goal'] = np.sqrt(
    (FIELD_LENGTH - df['end_x'])**2 + 
    (FIELD_WIDTH/2 - df['end_y'])**2
)

# 2.3 Distance from center line
df['start_dist_to_center_y'] = np.abs(df['start_y'] - FIELD_WIDTH/2)
df['end_dist_to_center_y'] = np.abs(df['end_y'] - FIELD_WIDTH/2)

# 2.4 Field zones (divide field into 6 zones: defensive/middle/attacking x left/center/right)
def get_zone_x(x):
    """Get horizontal zone: 0=defensive, 1=middle, 2=attacking"""
    if x < FIELD_LENGTH / 3:
        return 0  # Defensive third
    elif x < 2 * FIELD_LENGTH / 3:
        return 1  # Middle third
    else:
        return 2  # Attacking third

def get_zone_y(y):
    """Get vertical zone: 0=left, 1=center, 2=right"""
    if y < FIELD_WIDTH / 3:
        return 0  # Left
    elif y < 2 * FIELD_WIDTH / 3:
        return 1  # Center
    else:
        return 2  # Right

df['start_zone_x'] = df['start_x'].apply(get_zone_x)
df['start_zone_y'] = df['start_y'].apply(get_zone_y)
df['end_zone_x'] = df['end_x'].apply(get_zone_x)
df['end_zone_y'] = df['end_y'].apply(get_zone_y)

# Combined zone (0-8)
df['start_zone'] = df['start_zone_x'] * 3 + df['start_zone_y']
df['end_zone'] = df['end_zone_x'] * 3 + df['end_zone_y']

# 2.5 Normalized coordinates (0-1 range)
df['start_x_norm'] = df['start_x'] / FIELD_LENGTH
df['start_y_norm'] = df['start_y'] / FIELD_WIDTH
df['end_x_norm'] = df['end_x'] / FIELD_LENGTH
df['end_y_norm'] = df['end_y'] / FIELD_WIDTH

# 2.6 Polar coordinates (from center of field)
center_x, center_y = FIELD_LENGTH / 2, FIELD_WIDTH / 2
df['start_radius'] = np.sqrt((df['start_x'] - center_x)**2 + (df['start_y'] - center_y)**2)
df['start_theta'] = np.arctan2(df['start_y'] - center_y, df['start_x'] - center_x)
df['end_radius'] = np.sqrt((df['end_x'] - center_x)**2 + (df['end_y'] - center_y)**2)
df['end_theta'] = np.arctan2(df['end_y'] - center_y, df['end_x'] - center_x)

# 2.7 Movement direction
df['delta_x'] = df['end_x'] - df['start_x']
df['delta_y'] = df['end_y'] - df['start_y']
df['forward_progress'] = df['delta_x']  # Positive = towards opponent goal
df['lateral_movement'] = np.abs(df['delta_y'])

print(f"[OK] Created {30} spatial features")

# ============================================================================
# 3. TEMPORAL FEATURES (SEQUENCE-BASED)
# ============================================================================
print("\n[3] Creating temporal features...")

# Sort by game_episode and time
df = df.sort_values(['game_episode', 'time_seconds']).reset_index(drop=True)

# 3.1 Position in sequence
df['seq_position'] = df.groupby('game_episode').cumcount()
df['seq_total_length'] = df.groupby('game_episode')['game_episode'].transform('count')
df['seq_position_pct'] = df['seq_position'] / df['seq_total_length']

# 3.2 Time differences
df['time_since_start'] = df.groupby('game_episode')['time_seconds'].transform(
    lambda x: x - x.iloc[0]
)
df['time_diff'] = df.groupby('game_episode')['time_seconds'].diff().fillna(0)

# 3.3 Cumulative statistics within episode
df['cumsum_distance'] = df.groupby('game_episode')['pass_distance'].cumsum()
df['cumsum_forward'] = df.groupby('game_episode')['forward_progress'].cumsum()

# 3.4 Rolling statistics (last 3 events)
for window in [3, 5]:
    df[f'rolling_distance_mean_{window}'] = df.groupby('game_episode')['pass_distance'].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    df[f'rolling_forward_mean_{window}'] = df.groupby('game_episode')['forward_progress'].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )

print(f"[OK] Created temporal features")

# ============================================================================
# 4. CONTEXTUAL FEATURES
# ============================================================================
print("\n[4] Creating contextual features...")

# 4.1 Action type encoding
action_type_map = {action: idx for idx, action in enumerate(df['type_name'].unique())}
df['action_type_encoded'] = df['type_name'].map(action_type_map)

# 4.2 Result encoding (handle missing values)
df['result_encoded'] = df['result_name'].fillna('Unknown')
result_map = {result: idx for idx, result in enumerate(df['result_encoded'].unique())}
df['result_encoded'] = df['result_encoded'].map(result_map)

# 4.3 Previous action features
df['prev_action_type'] = df.groupby('game_episode')['action_type_encoded'].shift(1).fillna(-1)
df['prev_pass_distance'] = df.groupby('game_episode')['pass_distance'].shift(1).fillna(0)
df['prev_end_x'] = df.groupby('game_episode')['end_x'].shift(1).fillna(df['start_x'])
df['prev_end_y'] = df.groupby('game_episode')['end_y'].shift(1).fillna(df['start_y'])

# 4.4 Distance from previous event end position
df['dist_from_prev_end'] = np.sqrt(
    (df['start_x'] - df['prev_end_x'])**2 + 
    (df['start_y'] - df['prev_end_y'])**2
)

# 4.5 Team continuity (same team as previous action)
df['same_team_as_prev'] = (
    df.groupby('game_episode')['team_id'].shift(1) == df['team_id']
).astype(int).fillna(0)

# 4.6 Player continuity
df['same_player_as_prev'] = (
    df.groupby('game_episode')['player_id'].shift(1) == df['player_id']
).astype(int).fillna(0)

print(f"[OK] Created contextual features")

# ============================================================================
# 5. TEAM AND PLAYER FEATURES
# ============================================================================
print("\n[5] Creating team and player features...")

# 5.1 Team statistics (aggregated)
team_stats = df.groupby('team_id').agg({
    'pass_distance': ['mean', 'std'],
    'forward_progress': ['mean', 'std'],
    'end_x': 'mean',
    'end_y': 'mean'
}).reset_index()
team_stats.columns = ['team_id', 'team_avg_pass_dist', 'team_std_pass_dist',
                      'team_avg_forward', 'team_std_forward',
                      'team_avg_end_x', 'team_avg_end_y']
df = df.merge(team_stats, on='team_id', how='left')

# 5.2 Player statistics (aggregated)
player_stats = df.groupby('player_id').agg({
    'pass_distance': ['mean', 'std', 'count'],
    'forward_progress': 'mean',
    'end_x': 'mean',
    'end_y': 'mean'
}).reset_index()
player_stats.columns = ['player_id', 'player_avg_pass_dist', 'player_std_pass_dist',
                        'player_event_count', 'player_avg_forward',
                        'player_avg_end_x', 'player_avg_end_y']
df = df.merge(player_stats, on='player_id', how='left')

# 5.3 Deviation from player average
df['pass_dist_vs_player_avg'] = df['pass_distance'] - df['player_avg_pass_dist']
df['forward_vs_player_avg'] = df['forward_progress'] - df['player_avg_forward']

print(f"[OK] Created team and player features")

# ============================================================================
# 6. INTERACTION FEATURES
# ============================================================================
print("\n[6] Creating interaction features...")

# 6.1 Zone transitions
df['zone_change_x'] = df['end_zone_x'] - df['start_zone_x']
df['zone_change_y'] = df['end_zone_y'] - df['start_zone_y']

# 6.2 Position x Action type interactions
df['start_x_action'] = df['start_x_norm'] * df['action_type_encoded']
df['start_y_action'] = df['start_y_norm'] * df['action_type_encoded']

# 6.3 Distance x Zone interactions
df['distance_x_zone'] = df['pass_distance'] * df['start_zone']

# 6.4 Time x Position interactions
df['time_x_position'] = df['seq_position_pct'] * df['start_x_norm']

print(f"[OK] Created interaction features")

# ============================================================================
# 7. ADVANCED SPATIAL FEATURES
# ============================================================================
print("\n[7] Creating advanced spatial features...")

# 7.1 Voronoi-inspired features (space control estimation)
# Simplified: distance to nearest boundary
df['dist_to_left_boundary'] = df['start_y']
df['dist_to_right_boundary'] = FIELD_WIDTH - df['start_y']
df['dist_to_back_boundary'] = df['start_x']
df['dist_to_front_boundary'] = FIELD_LENGTH - df['start_x']
df['min_boundary_dist'] = df[['dist_to_left_boundary', 'dist_to_right_boundary',
                               'dist_to_back_boundary', 'dist_to_front_boundary']].min(axis=1)

# 7.2 Penalty box features
# Penalty box dimensions (approximate): 16.5m from goal line, 40.3m wide
PENALTY_BOX_DEPTH = 16.5
PENALTY_BOX_WIDTH = 40.3

def in_penalty_box(x, y):
    """Check if position is in attacking penalty box"""
    return (x >= FIELD_LENGTH - PENALTY_BOX_DEPTH) and \
           (y >= (FIELD_WIDTH - PENALTY_BOX_WIDTH) / 2) and \
           (y <= (FIELD_WIDTH + PENALTY_BOX_WIDTH) / 2)

df['start_in_penalty_box'] = df.apply(lambda row: in_penalty_box(row['start_x'], row['start_y']), axis=1).astype(int)
df['end_in_penalty_box'] = df.apply(lambda row: in_penalty_box(row['end_x'], row['end_y']), axis=1).astype(int)

print(f"[OK] Created advanced spatial features")

# ============================================================================
# 8. SAVE ENGINEERED FEATURES
# ============================================================================
print("\n[8] Saving engineered features...")

# Create output directory
OUTPUT_DIR = Path('./processed_data')
OUTPUT_DIR.mkdir(exist_ok=True)

# Save full feature set
output_file = OUTPUT_DIR / 'train_with_features.csv'
df.to_csv(output_file, index=False)
print(f"[OK] Saved: {output_file}")
print(f"    Total features: {len(df.columns)}")
print(f"    Total rows: {len(df):,}")

# Save feature list
feature_cols = [col for col in df.columns if col not in [
    'game_id', 'period_id', 'episode_id', 'team_id', 'player_id', 'action_id',
    'type_name', 'result_name', 'game_episode'
]]

feature_info = pd.DataFrame({
    'feature_name': feature_cols,
    'dtype': [str(df[col].dtype) for col in feature_cols]
})
feature_info.to_csv(OUTPUT_DIR / 'feature_list.csv', index=False)
print(f"[OK] Saved feature list: {OUTPUT_DIR / 'feature_list.csv'}")

# ============================================================================
# 9. FEATURE STATISTICS
# ============================================================================
print("\n[9] Feature statistics summary...")

print("\nSpatial Features:")
spatial_features = [col for col in df.columns if any(x in col for x in ['distance', 'angle', 'zone', 'radius', 'delta'])]
print(f"  Count: {len(spatial_features)}")

print("\nTemporal Features:")
temporal_features = [col for col in df.columns if any(x in col for x in ['seq_', 'time_', 'cumsum', 'rolling'])]
print(f"  Count: {len(temporal_features)}")

print("\nContextual Features:")
contextual_features = [col for col in df.columns if any(x in col for x in ['prev_', 'same_', 'encoded'])]
print(f"  Count: {len(contextual_features)}")

print("\nTeam/Player Features:")
team_player_features = [col for col in df.columns if any(x in col for x in ['team_', 'player_'])]
print(f"  Count: {len(team_player_features)}")

print("\n" + "="*80)
print("FEATURE ENGINEERING COMPLETE!")
print(f"Total features created: {len(df.columns)}")
print(f"Output saved to: {OUTPUT_DIR.absolute()}")
print("="*80)
