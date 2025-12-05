# K-League Pass Prediction - Quick Start Guide

## Overview
This project provides a complete data exploration, feature engineering, and preprocessing pipeline for the K-League pass prediction challenge.

## Files Created

### Analysis Scripts
1. **`01_data_exploration.py`** - Comprehensive EDA with 10+ visualizations
2. **`02_feature_engineering.py`** - Creates 86 features from raw data
3. **`03_data_preprocessing.py`** - Prepares data for modeling
4. **`04_process_test_data.py`** - Processes test data for predictions

### Output Directories
- **`eda_outputs/`** - EDA visualizations and statistics
- **`processed_data/`** - Processed datasets ready for modeling

## Quick Start

### 1. Run Data Exploration
```bash
python 01_data_exploration.py
```
**Output**: Visualizations in `eda_outputs/` directory

### 2. Engineer Features
```bash
python 02_feature_engineering.py
```
**Output**: `processed_data/train_with_features.csv` (86 features)

### 3. Preprocess Data
```bash
python 03_data_preprocessing.py
```
**Output**: 
- `processed_data/train_processed.csv` (12,389 episodes)
- `processed_data/val_processed.csv` (3,046 episodes)
- `processed_data/scaler.pkl`
- `processed_data/feature_info.pkl`

### 4. Process Test Data (when ready)
```bash
python 04_process_test_data.py
```
**Output**: `processed_data/test_processed.csv`

## Data Summary

### Training Data
- **Events**: 356,721
- **Episodes**: 15,435
- **Games**: 198
- **Features**: 86 (event-level) → 31 (aggregated)

### Train/Validation Split
- **Training**: 12,389 episodes (80.3%) from 158 games
- **Validation**: 3,046 episodes (19.7%) from 40 games

### Features Created

| Category | Count | Examples |
|----------|-------|----------|
| Spatial | 19 | pass_distance, zones, angles |
| Temporal | 13 | sequence position, rolling stats |
| Contextual | 9 | previous actions, continuity |
| Team/Player | 18 | averages, deviations |
| Interaction | 4 | zone transitions, combinations |

## Using the Processed Data

```python
import pandas as pd
import pickle

# Load processed data
train = pd.read_csv('./processed_data/train_processed.csv')
val = pd.read_csv('./processed_data/val_processed.csv')

# Load preprocessing artifacts
with open('./processed_data/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('./processed_data/feature_info.pkl', 'rb') as f:
    feature_info = pickle.load(f)

# Prepare for modeling
X_train = train[feature_info['scale_cols']]
y_train = train[['target_x', 'target_y']]

X_val = val[feature_info['scale_cols']]
y_val = val[['target_x', 'target_y']]

# Train your model
# model.fit(X_train, y_train)
# predictions = model.predict(X_val)
```

## Evaluation Metric

Euclidean distance between predicted and actual coordinates:
```python
distance = sqrt((pred_x - actual_x)^2 + (pred_y - actual_y)^2)
```

## Next Steps

1. **Baseline Model**: Start with Random Forest or XGBoost on aggregated features
2. **Feature Selection**: Identify most important features
3. **Advanced Models**: Try LSTM/Transformer on sequence data
4. **Ensemble**: Combine multiple approaches
5. **Submit**: Process test data and generate predictions

## Key Insights

- Episodes vary greatly in length (1-270 events)
- Passes are predominantly forward-oriented
- Team and player styles show significant variation
- Sequence position and spatial features are likely most important

## Competition Requirements

✅ All code uses relative paths  
✅ UTF-8 encoding  
✅ No external data used  
✅ No data leakage (split by game_id)  
✅ Reproducible with saved scaler and feature info  

## Files for Submission

When submitting:
- All 4 Python scripts
- `requirements.txt` (create with your environment)
- Trained model weights
- Inference script (to be created)
- Solution documentation

---

**Ready to start modeling!** 🚀
