"""
K-League Pass Prediction - Comprehensive Data Exploration
==========================================================
This script performs thorough exploratory data analysis on the K-League dataset
to understand patterns, distributions, and relationships before modeling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 8)
plt.rcParams['font.size'] = 10

# Create output directory for plots
OUTPUT_DIR = Path('./eda_outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("K-LEAGUE PASS PREDICTION - DATA EXPLORATION")
print("="*80)

# ============================================================================
# 1. LOAD DATASETS
# ============================================================================
print("\n[1] Loading datasets...")
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_match = pd.read_csv('match_info.csv')
df_sample = pd.read_csv('sample_submission.csv')

print(f"[OK] Train data: {df_train.shape[0]:,} rows x {df_train.shape[1]} columns")
print(f"[OK] Test data: {df_test.shape[0]:,} episodes")
print(f"[OK] Match info: {df_match.shape[0]:,} matches")
print(f"[OK] Sample submission: {df_sample.shape[0]:,} predictions needed")

# ============================================================================
# 2. BASIC DATA STRUCTURE ANALYSIS
# ============================================================================
print("\n[2] Analyzing data structure...")
print("\n--- Column Information ---")
print(df_train.dtypes)

print("\n--- Missing Values ---")
missing = df_train.isnull().sum()
missing_pct = (missing / len(df_train) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Percentage': missing_pct
})
print(missing_df[missing_df['Missing Count'] > 0])

print("\n--- Basic Statistics ---")
print(f"Unique games: {df_train['game_id'].nunique()}")
print(f"Unique episodes: {df_train['game_episode'].nunique()}")
print(f"Unique teams: {df_train['team_id'].nunique()}")
print(f"Unique players: {df_train['player_id'].nunique()}")
print(f"Unique action types: {df_train['type_name'].nunique()}")

# ============================================================================
# 3. EPISODE STRUCTURE ANALYSIS
# ============================================================================
print("\n[3] Analyzing episode structure...")

# Calculate sequence lengths per episode
episode_lengths = df_train.groupby('game_episode').size()
print(f"\nSequence Length Statistics:")
print(f"  Mean: {episode_lengths.mean():.2f}")
print(f"  Median: {episode_lengths.median():.0f}")
print(f"  Min: {episode_lengths.min()}")
print(f"  Max: {episode_lengths.max()}")
print(f"  Std: {episode_lengths.std():.2f}")

# Visualize sequence length distribution
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.hist(episode_lengths, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.title('Distribution of Episode Sequence Lengths')
plt.axvline(episode_lengths.mean(), color='red', linestyle='--', label=f'Mean: {episode_lengths.mean():.1f}')
plt.axvline(episode_lengths.median(), color='green', linestyle='--', label=f'Median: {episode_lengths.median():.0f}')
plt.legend()

plt.subplot(1, 2, 2)
plt.boxplot(episode_lengths, vert=True)
plt.ylabel('Sequence Length')
plt.title('Boxplot of Episode Sequence Lengths')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'episode_lengths.png', dpi=300, bbox_inches='tight')
print(f"[OK] Saved: {OUTPUT_DIR / 'episode_lengths.png'}")

# ============================================================================
# 4. ACTION TYPE ANALYSIS
# ============================================================================
print("\n[4] Analyzing action types...")

action_counts = df_train['type_name'].value_counts()
print("\nAction Type Distribution:")
print(action_counts)

# Visualize action types
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
action_counts.plot(kind='bar', color='steelblue', edgecolor='black')
plt.xlabel('Action Type')
plt.ylabel('Count')
plt.title('Action Type Distribution')
plt.xticks(rotation=45, ha='right')

plt.subplot(1, 2, 2)
plt.pie(action_counts, labels=action_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Action Type Proportions')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'action_types.png', dpi=300, bbox_inches='tight')
print(f"[OK] Saved: {OUTPUT_DIR / 'action_types.png'}")

# ============================================================================
# 5. COORDINATE ANALYSIS (FIELD: 105 x 68)
# ============================================================================
print("\n[5] Analyzing coordinate distributions...")

# Field dimensions
FIELD_LENGTH = 105
FIELD_WIDTH = 68

print(f"\nField dimensions: {FIELD_LENGTH} × {FIELD_WIDTH}")
print("\n--- Start Coordinates ---")
print(f"X range: [{df_train['start_x'].min():.2f}, {df_train['start_x'].max():.2f}]")
print(f"Y range: [{df_train['start_y'].min():.2f}, {df_train['start_y'].max():.2f}]")
print("\n--- End Coordinates (Target) ---")
print(f"X range: [{df_train['end_x'].min():.2f}, {df_train['end_x'].max():.2f}]")
print(f"Y range: [{df_train['end_y'].min():.2f}, {df_train['end_y'].max():.2f}]")

# Visualize coordinate distributions
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Start X distribution
axes[0, 0].hist(df_train['start_x'], bins=50, edgecolor='black', alpha=0.7, color='blue')
axes[0, 0].set_xlabel('Start X')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Start X Coordinate Distribution')
axes[0, 0].axvline(FIELD_LENGTH/2, color='red', linestyle='--', label='Midfield')
axes[0, 0].legend()

# Start Y distribution
axes[0, 1].hist(df_train['start_y'], bins=50, edgecolor='black', alpha=0.7, color='green')
axes[0, 1].set_xlabel('Start Y')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Start Y Coordinate Distribution')
axes[0, 1].axvline(FIELD_WIDTH/2, color='red', linestyle='--', label='Center')
axes[0, 1].legend()

# End X distribution (target)
axes[1, 0].hist(df_train['end_x'], bins=50, edgecolor='black', alpha=0.7, color='orange')
axes[1, 0].set_xlabel('End X (Target)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('End X Coordinate Distribution (Target)')
axes[1, 0].axvline(FIELD_LENGTH/2, color='red', linestyle='--', label='Midfield')
axes[1, 0].legend()

# End Y distribution (target)
axes[1, 1].hist(df_train['end_y'], bins=50, edgecolor='black', alpha=0.7, color='purple')
axes[1, 1].set_xlabel('End Y (Target)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('End Y Coordinate Distribution (Target)')
axes[1, 1].axvline(FIELD_WIDTH/2, color='red', linestyle='--', label='Center')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'coordinate_distributions.png', dpi=300, bbox_inches='tight')
print(f"[OK] Saved: {OUTPUT_DIR / 'coordinate_distributions.png'}")

# ============================================================================
# 6. SPATIAL HEATMAPS
# ============================================================================
print("\n[6] Creating spatial heatmaps...")

# Sample data for faster plotting
sample_size = min(50000, len(df_train))
df_sample_plot = df_train.sample(n=sample_size, random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Start position heatmap
axes[0].hexbin(df_sample_plot['start_x'], df_sample_plot['start_y'], 
               gridsize=30, cmap='YlOrRd', alpha=0.8)
axes[0].set_xlim(0, FIELD_LENGTH)
axes[0].set_ylim(0, FIELD_WIDTH)
axes[0].set_xlabel('X Coordinate')
axes[0].set_ylabel('Y Coordinate')
axes[0].set_title('Start Position Heatmap')
axes[0].axvline(FIELD_LENGTH/2, color='white', linestyle='--', alpha=0.5)
axes[0].axhline(FIELD_WIDTH/2, color='white', linestyle='--', alpha=0.5)

# End position heatmap (target)
axes[1].hexbin(df_sample_plot['end_x'], df_sample_plot['end_y'], 
               gridsize=30, cmap='YlGnBu', alpha=0.8)
axes[1].set_xlim(0, FIELD_LENGTH)
axes[1].set_ylim(0, FIELD_WIDTH)
axes[1].set_xlabel('X Coordinate')
axes[1].set_ylabel('Y Coordinate')
axes[1].set_title('End Position Heatmap (Target)')
axes[1].axvline(FIELD_LENGTH/2, color='white', linestyle='--', alpha=0.5)
axes[1].axhline(FIELD_WIDTH/2, color='white', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'position_heatmaps.png', dpi=300, bbox_inches='tight')
print(f"[OK] Saved: {OUTPUT_DIR / 'position_heatmaps.png'}")

# ============================================================================
# 7. TEMPORAL ANALYSIS
# ============================================================================
print("\n[7] Analyzing temporal patterns...")

# Time statistics
print("\nTime Statistics:")
print(f"  Min time: {df_train['time_seconds'].min():.2f}s")
print(f"  Max time: {df_train['time_seconds'].max():.2f}s")
print(f"  Mean time: {df_train['time_seconds'].mean():.2f}s")

# Period distribution
period_counts = df_train['period_id'].value_counts().sort_index()
print("\nPeriod Distribution:")
print(period_counts)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
period_counts.plot(kind='bar', color='teal', edgecolor='black')
plt.xlabel('Period ID')
plt.ylabel('Count')
plt.title('Distribution of Events by Period')
plt.xticks(rotation=0)

plt.subplot(1, 2, 2)
plt.hist(df_train['time_seconds'], bins=50, edgecolor='black', alpha=0.7, color='coral')
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency')
plt.title('Distribution of Event Times')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'temporal_analysis.png', dpi=300, bbox_inches='tight')
print(f"[OK] Saved: {OUTPUT_DIR / 'temporal_analysis.png'}")

# ============================================================================
# 8. PASS DISTANCE ANALYSIS
# ============================================================================
print("\n[8] Analyzing pass distances...")

# Calculate Euclidean distance for each pass
df_train['pass_distance'] = np.sqrt(
    (df_train['end_x'] - df_train['start_x'])**2 + 
    (df_train['end_y'] - df_train['start_y'])**2
)

print("\nPass Distance Statistics:")
print(f"  Mean: {df_train['pass_distance'].mean():.2f}")
print(f"  Median: {df_train['pass_distance'].median():.2f}")
print(f"  Min: {df_train['pass_distance'].min():.2f}")
print(f"  Max: {df_train['pass_distance'].max():.2f}")
print(f"  Std: {df_train['pass_distance'].std():.2f}")

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.hist(df_train['pass_distance'], bins=50, edgecolor='black', alpha=0.7, color='mediumseagreen')
plt.xlabel('Pass Distance')
plt.ylabel('Frequency')
plt.title('Distribution of Pass Distances')
plt.axvline(df_train['pass_distance'].mean(), color='red', linestyle='--', 
            label=f'Mean: {df_train["pass_distance"].mean():.2f}')
plt.legend()

plt.subplot(1, 2, 2)
plt.boxplot(df_train['pass_distance'], vert=True)
plt.ylabel('Pass Distance')
plt.title('Boxplot of Pass Distances')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'pass_distances.png', dpi=300, bbox_inches='tight')
print(f"[OK] Saved: {OUTPUT_DIR / 'pass_distances.png'}")

# ============================================================================
# 9. HOME VS AWAY ANALYSIS
# ============================================================================
print("\n[9] Analyzing home vs away patterns...")

home_away_counts = df_train['is_home'].value_counts()
print("\nHome vs Away Distribution:")
print(home_away_counts)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
home_away_counts.plot(kind='bar', color=['steelblue', 'coral'], edgecolor='black')
plt.xlabel('Is Home')
plt.ylabel('Count')
plt.title('Home vs Away Event Distribution')
plt.xticks(rotation=0)

# Compare pass distances
plt.subplot(1, 2, 2)
df_train.boxplot(column='pass_distance', by='is_home', ax=plt.gca())
plt.xlabel('Is Home')
plt.ylabel('Pass Distance')
plt.title('Pass Distance: Home vs Away')
plt.suptitle('')  # Remove default title
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'home_away_analysis.png', dpi=300, bbox_inches='tight')
print(f"[OK] Saved: {OUTPUT_DIR / 'home_away_analysis.png'}")

# ============================================================================
# 10. RESULT NAME ANALYSIS
# ============================================================================
print("\n[10] Analyzing result names...")

result_counts = df_train['result_name'].value_counts()
print("\nResult Name Distribution:")
print(result_counts)

plt.figure(figsize=(12, 6))
result_counts.plot(kind='bar', color='darkslateblue', edgecolor='black')
plt.xlabel('Result Name')
plt.ylabel('Count')
plt.title('Distribution of Result Names')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'result_names.png', dpi=300, bbox_inches='tight')
print(f"[OK] Saved: {OUTPUT_DIR / 'result_names.png'}")

# ============================================================================
# 11. CORRELATION ANALYSIS
# ============================================================================
print("\n[11] Analyzing correlations...")

# Select numeric columns for correlation
numeric_cols = ['start_x', 'start_y', 'end_x', 'end_y', 'time_seconds', 'pass_distance']
corr_matrix = df_train[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, linewidths=1)
plt.title('Correlation Matrix of Numeric Features')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
print(f"[OK] Saved: {OUTPUT_DIR / 'correlation_matrix.png'}")

# ============================================================================
# 12. SAMPLE EPISODE EXAMINATION
# ============================================================================
print("\n[12] Examining sample episodes...")

# Get a sample episode
sample_episode = df_train[df_train['game_episode'] == df_train['game_episode'].iloc[0]]
print(f"\nSample Episode: {sample_episode['game_episode'].iloc[0]}")
print(f"Number of events: {len(sample_episode)}")
print("\nSequence of events:")
print(sample_episode[['time_seconds', 'type_name', 'start_x', 'start_y', 'end_x', 'end_y']].head(10))

# Visualize episode trajectory
plt.figure(figsize=(12, 8))
plt.plot(sample_episode['start_x'], sample_episode['start_y'], 
         'o-', markersize=8, linewidth=2, label='Event Sequence')
plt.scatter(sample_episode['start_x'].iloc[0], sample_episode['start_y'].iloc[0], 
           s=200, c='green', marker='*', label='Start', zorder=5)
plt.scatter(sample_episode['end_x'].iloc[-1], sample_episode['end_y'].iloc[-1], 
           s=200, c='red', marker='X', label='Final Target', zorder=5)
plt.xlim(0, FIELD_LENGTH)
plt.ylim(0, FIELD_WIDTH)
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title(f'Sample Episode Trajectory: {sample_episode["game_episode"].iloc[0]}')
plt.axvline(FIELD_LENGTH/2, color='gray', linestyle='--', alpha=0.3)
plt.axhline(FIELD_WIDTH/2, color='gray', linestyle='--', alpha=0.3)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'sample_episode_trajectory.png', dpi=300, bbox_inches='tight')
print(f"[OK] Saved: {OUTPUT_DIR / 'sample_episode_trajectory.png'}")

# ============================================================================
# 13. SUMMARY STATISTICS EXPORT
# ============================================================================
print("\n[13] Exporting summary statistics...")

summary_stats = {
    'Dataset Overview': {
        'Total Events': len(df_train),
        'Unique Games': df_train['game_id'].nunique(),
        'Unique Episodes': df_train['game_episode'].nunique(),
        'Unique Teams': df_train['team_id'].nunique(),
        'Unique Players': df_train['player_id'].nunique(),
        'Test Episodes': len(df_test)
    },
    'Episode Statistics': {
        'Mean Sequence Length': episode_lengths.mean(),
        'Median Sequence Length': episode_lengths.median(),
        'Min Sequence Length': episode_lengths.min(),
        'Max Sequence Length': episode_lengths.max(),
        'Std Sequence Length': episode_lengths.std()
    },
    'Coordinate Statistics': {
        'Field Length': FIELD_LENGTH,
        'Field Width': FIELD_WIDTH,
        'Start X Range': f"[{df_train['start_x'].min():.2f}, {df_train['start_x'].max():.2f}]",
        'Start Y Range': f"[{df_train['start_y'].min():.2f}, {df_train['start_y'].max():.2f}]",
        'End X Range': f"[{df_train['end_x'].min():.2f}, {df_train['end_x'].max():.2f}]",
        'End Y Range': f"[{df_train['end_y'].min():.2f}, {df_train['end_y'].max():.2f}]"
    },
    'Pass Distance Statistics': {
        'Mean Distance': df_train['pass_distance'].mean(),
        'Median Distance': df_train['pass_distance'].median(),
        'Min Distance': df_train['pass_distance'].min(),
        'Max Distance': df_train['pass_distance'].max(),
        'Std Distance': df_train['pass_distance'].std()
    }
}

# Save to JSON (convert numpy types to Python types)
import json

def convert_to_python_type(obj):
    """Convert numpy types to Python native types"""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_type(item) for item in obj]
    return obj

summary_stats_converted = convert_to_python_type(summary_stats)

with open(OUTPUT_DIR / 'summary_statistics.json', 'w') as f:
    json.dump(summary_stats_converted, f, indent=2)
print(f"[OK] Saved: {OUTPUT_DIR / 'summary_statistics.json'}")

print("\n" + "="*80)
print("DATA EXPLORATION COMPLETE!")
print(f"All outputs saved to: {OUTPUT_DIR.absolute()}")
print("="*80)
