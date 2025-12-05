"""
K-League Pass Prediction - Data Exploration Summary
====================================================
Quick reference for key statistics and insights
"""

# ============================================================================
# DATASET OVERVIEW
# ============================================================================
DATASET_STATS = {
    'training_events': 356_721,
    'unique_episodes': 15_435,
    'unique_games': 198,
    'unique_teams': 12,
    'unique_players': 446,
    'action_types': 26,
    'test_episodes': 2_414
}

# ============================================================================
# EPISODE STATISTICS
# ============================================================================
EPISODE_STATS = {
    'mean_length': 23.11,
    'median_length': 16,
    'min_length': 1,
    'max_length': 270,
    'std_length': 21.99
}

# ============================================================================
# FIELD DIMENSIONS
# ============================================================================
FIELD_LENGTH = 105  # FIFA standard
FIELD_WIDTH = 68

# ============================================================================
# COORDINATE RANGES
# ============================================================================
COORDINATE_RANGES = {
    'start_x': (0.0, 105.0),
    'start_y': (0.0, 68.0),
    'end_x': (0.0, 105.0),
    'end_y': (0.0, 68.0)
}

# ============================================================================
# PASS STATISTICS
# ============================================================================
PASS_STATS = {
    'mean_distance': 12.55,
    'median_distance': 10.15,
    'min_distance': 0.0,
    'max_distance': 101.08,
    'std_distance': 11.40
}

# ============================================================================
# ACTION TYPE DISTRIBUTION (TOP 10)
# ============================================================================
TOP_ACTIONS = {
    'Pass': 178_582,
    'Carry': 82_046,
    'Recovery': 27_352,
    'Interception': 11_088,
    'Duel': 8_734,
    'Tackle': 8_138,
    'Throw-In': 6_801,
    'Clearance': 6_563,
    'Intervention': 6_038,
    'Block': 3_983
}

# ============================================================================
# FEATURES CREATED
# ============================================================================
FEATURE_COUNTS = {
    'event_level_features': 86,
    'aggregated_features': 31,
    'spatial_features': 19,
    'temporal_features': 13,
    'contextual_features': 9,
    'team_player_features': 18,
    'interaction_features': 4
}

# ============================================================================
# TRAIN/VALIDATION SPLIT
# ============================================================================
SPLIT_INFO = {
    'train_episodes': 12_389,
    'train_games': 158,
    'train_percentage': 80.3,
    'val_episodes': 3_046,
    'val_games': 40,
    'val_percentage': 19.7
}

# ============================================================================
# TARGET STATISTICS (TRAINING SET)
# ============================================================================
TARGET_STATS = {
    'target_x_mean': 68.47,
    'target_x_std': 23.80,
    'target_x_range': (0.0, 105.0),
    'target_y_mean': 33.56,
    'target_y_std': 24.34,
    'target_y_range': (0.0, 68.0)
}

# ============================================================================
# MISSING DATA
# ============================================================================
MISSING_DATA = {
    'result_name': 140_254  # 39.32% of data
}

# ============================================================================
# KEY INSIGHTS
# ============================================================================
KEY_INSIGHTS = [
    "Episodes have high variability in length (1-270 events)",
    "Pass is the most common action (50.1% of all events)",
    "Average pass distance is 12.55 units on 105x68 field",
    "Passes tend to be forward-oriented (towards opponent goal)",
    "Team and player styles show significant statistical variation",
    "39% of result_name values are missing (handled in preprocessing)",
    "Home and away teams have similar event distributions",
    "Sequence position matters (early vs late events differ)"
]

# ============================================================================
# MODELING RECOMMENDATIONS
# ============================================================================
MODELING_RECOMMENDATIONS = [
    "Start with aggregated features (31) for baseline models",
    "Use Random Forest or Gradient Boosting for quick baseline",
    "Try LSTM/GRU on full sequences for temporal patterns",
    "Consider Transformer models for attention mechanisms",
    "Ensemble multiple approaches for best performance",
    "Use Euclidean distance as evaluation metric",
    "Spatial features likely most predictive",
    "Include team/player features for personalization"
]

# ============================================================================
# FILES GENERATED
# ============================================================================
OUTPUT_FILES = {
    'scripts': [
        '01_data_exploration.py',
        '02_feature_engineering.py',
        '03_data_preprocessing.py',
        '04_process_test_data.py'
    ],
    'processed_data': [
        'train_with_features.csv',  # 356,721 rows x 86 features
        'train_processed.csv',       # 12,389 episodes x 34 columns
        'val_processed.csv',         # 3,046 episodes x 34 columns
        'sequence_data_full.csv',    # 356,721 rows with split indicator
        'feature_list.csv',          # Feature metadata
        'scaler.pkl',                # Fitted RobustScaler
        'feature_info.pkl'           # Feature column info
    ],
    'visualizations': [
        'episode_lengths.png',
        'action_types.png',
        'coordinate_distributions.png',
        'position_heatmaps.png',
        'temporal_analysis.png',
        'pass_distances.png',
        'home_away_analysis.png',
        'result_names.png',
        'correlation_matrix.png',
        'sample_episode_trajectory.png',
        'summary_statistics.json'
    ]
}

if __name__ == '__main__':
    print("="*80)
    print("K-LEAGUE PASS PREDICTION - DATA SUMMARY")
    print("="*80)
    
    print("\n[DATASET]")
    for key, value in DATASET_STATS.items():
        print(f"  {key}: {value:,}")
    
    print("\n[EPISODES]")
    for key, value in EPISODE_STATS.items():
        print(f"  {key}: {value}")
    
    print("\n[FEATURES]")
    for key, value in FEATURE_COUNTS.items():
        print(f"  {key}: {value}")
    
    print("\n[SPLIT]")
    for key, value in SPLIT_INFO.items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")
    
    print("\n[KEY INSIGHTS]")
    for i, insight in enumerate(KEY_INSIGHTS, 1):
        print(f"  {i}. {insight}")
    
    print("\n" + "="*80)
    print("Ready for modeling!")
    print("="*80)
