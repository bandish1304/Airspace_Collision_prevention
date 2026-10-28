PHASE 1: RESEARCH IMPLEMENTATION STEPS (Weeks 1-15)
GNN Baseline vs. TFT Primary — Airspace Collision Risk Prediction

WEEK 1 - Setup
1. Initialize Git repository with folders: data/, src/, notebooks/, models/, experiments/
2. Install and run MLflow locally, confirm tracking UI works
3. Register for OpenSky Network API access
4. Write a script to pull sample ADS-B state vectors for Southern California airspace
5. Download NTSB accident database
6. Write a short data dictionary documenting available fields

WEEK 2 - Feature Engineering and Labeling
1. Write function to compute haversine distance between two aircraft
2. Write function to compute closing speed
3. Write function to compute bearing difference
4. Write function to compute vertical separation
5. Write function to compute time to closest point of approach (CPA)
6. Combine functions into a pipeline that runs on every aircraft pair within 50nm
7. Label each pair as collision risk / not using ICAO separation minima
8. Run exploratory data analysis: class distribution, feature distributions, correlations

WEEK 3 - Imbalance Study
1. Build a shared evaluation harness for the imbalance study (AUROC, precision, recall, F1, PR curve, false negative rate)
2. Train a logistic regression baseline without any imbalance handling
3. Train logistic regression baseline with SMOTE oversampling
4. Train logistic regression baseline with focal loss
5. Train logistic regression baseline with class weight scaling
6. Log all three runs and their precision-recall curves to MLflow
7. Compare results and select the best imbalance strategy to carry forward

WEEK 4 - Graph Construction
1. Write function to convert a time-window snapshot into a graph structure (aircraft = nodes, proximity pairs = edges)
2. Attach engineered kinematic features as edge attributes
3. Validate graph construction on a sample time window
4. Build a dataset class producing a sequence of graph snapshots

WEEK 5 - GNN Baseline Tuning and Evaluation
1. Implement a GNN baseline (GraphSAGE or GAT) for edge-level collision risk classification
2. Train and tune the GNN baseline using Optuna hyperparameter search
3. Evaluate using a shared harness computing AUROC, F1, precision, recall, and a cost-sensitive metric penalizing false negatives at five times the cost of false positives
4. Log all results to MLflow

WEEK 6 - TFT Study and Input Pipeline
1. Study the Temporal Fusion Transformer in depth, including variable selection networks, gated residual networks, and interpretable multi-head attention
2. Build sliding-window trajectory sequences from ADS-B data to serve as model inputs

WEEK 7 - TFT Input Pipeline
1. Build sliding-window trajectory sequences per aircraft pair
2. Construct a TimeSeriesDataSet (or custom dataset) formatted for TFT input
3. Split into train/validation/test sets with proper temporal separation

WEEK 8 - TFT Implementation
1. Implement/configure TFT model using pytorch-forecasting
2. Replace default output head with a binary classification head
3. Wire in the imbalance-handling strategy into the TFT loss function
4. Run a short training pass to confirm the pipeline works end-to-end
