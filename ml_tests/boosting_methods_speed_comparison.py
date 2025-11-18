import numpy as np
import pandas as pd
import time
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURATION ---
N_ROWS = 1000000  # 1 Million Data Points
N_COLS = 300        # 300 Features
TREES = 100         # Fair comparison: All models build exactly 100 trees

print(f"Generating synthetic data ({N_ROWS} rows, {N_COLS} features)...")
print("This might take a moment and uses ~2.5GB RAM.")


# Replace the data generation block with this:

# 1. Create Random Floats
X_num = np.random.rand(N_ROWS, N_COLS).astype(np.float32)

# 2. Create a High-Cardinality Categorical Column (e.g., "CityID")
# Random integers from 0 to 1000 (simulating 1000 different cities)
X_cat = np.random.randint(0, 1000, size=(N_ROWS, 1))

# 3. Combine them
X = np.hstack([X_num, X_cat])

# ⚠️ CRITICAL: Tell CatBoost which column is categorical
cat_features = [N_COLS] # The last column is the category






# # Generate random data (Float32 to save memory)
# X = np.random.rand(N_ROWS, N_COLS).astype(np.float32)
# Generate a random target
y = np.random.rand(N_ROWS).astype(np.float32)

# Take a single row for the "Single Inference" test
single_row = X[0:1]

print("Data generation complete. Starting Benchmark...\n")
print(f"{'Algorithm':<15} | {'Train Time':<12} | {'Batch (1M)':<12} | {'Single Row':<12}")
print("-" * 60)

# Define the models with identical tree counts
models = [
    ("XGBoost", xgb.XGBRegressor(n_estimators=TREES, n_jobs=-1, 
                                #  tree_method="hist"
                                 )),
    ("LightGBM", lgb.LGBMRegressor(n_estimators=TREES, n_jobs=-1)),
    ("CatBoost", CatBoostRegressor(iterations=TREES, verbose=0, allow_writing_files=False, thread_count=-1))
]

for name, model in models:
    # 1. TRAINING SPEED
    start = time.time()
    model.fit(X, y)
    train_time = time.time() - start
    
    # 2. BATCH INFERENCE SPEED (Predicting all 1 Million rows at once)
    start = time.time()
    model.predict(X)
    batch_time = time.time() - start
    
    # 3. SINGLE INFERENCE SPEED (Simulating Real-Time API)
    # We loop 1,000 times and average it to measure latency in microseconds
    # Note: This includes Python overhead, but it affects all models equally.
    loops = 1000
    start = time.time()
    for _ in range(loops):
        model.predict(single_row)
    end = time.time()
    
    # Calculate average time per single prediction in microseconds
    single_time_us = ((end - start) / loops) * 1_000_000
    
    # Print Result Row
    print(f"{name:<15} | {train_time:.2f}s {'':<5} | {batch_time:.4f}s {'':<5} | {single_time_us:.0f} µs")

print("-" * 60)
print("\nAnalysis:")
print("1. Train Time: LightGBM is usually fastest here.")
print("2. Batch (1M): CatBoost should win (due to SIMD/Vectorization).")
print("3. Single Row: CatBoost should be fastest (due to symmetric tree array lookup).")