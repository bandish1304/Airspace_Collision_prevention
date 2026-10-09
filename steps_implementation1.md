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

WEEK 9 - TFT Tuning
1. Set up Optuna study to tune TFT hyperparameters (attention heads, hidden size, dropout, learning rate)
2. Train tuned TFT with early stopping and learning rate scheduling
3. Log all tuning runs and final tuned model to MLflow

WEEK 10 - Main Comparison (GNN vs TFT)
1. Run tuned GNN on the held-out test set
2. Run tuned TFT on the same held-out test set
3. Compute all shared-harness metrics for both models
4. Run statistical significance tests on the metric differences
5. Produce a comparison table/figure for the paper

WEEK 11 - Interpretability
1. Extract TFT attention weights from the trained model
2. Extract TFT variable selection scores from the trained model
3. Build visualizations (heatmaps) of feature/time-step importance
4. Write up findings connecting patterns to known collision precursors

WEEK 12 - Ablation Studies
1. Create ablated TFT variant with variable selection network removed
2. Create ablated TFT variant with shortened input sequence length
3. Create ablated TFT variant with static covariates removed
4. Retrain and evaluate each variant using the shared harness
5. Compare results to the full TFT to quantify each component's contribution

WEEK 13 - Drift Analysis and Visualization
1. Split historical data into distinct time windows
2. Measure whether feature distributions shift between windows
3. Compare GNN vs TFT performance degradation across windows
4. Build a small static visualization plotting aircraft positions and predicted risk for a sample time window

WEEK 14 - Manuscript Drafting
1. Draft Introduction section
2. Draft Related Work section
3. Draft Methodology section
4. Draft Experimental Setup section
5. Draft Results section with figures/tables from Weeks 10-13
6. Draft Discussion and Conclusion sections

WEEK 15 - Finalize and Submit
1. Revise manuscript based on self-review
2. Clean up code repository
3. Write README and requirements file
4. Ensure MLflow experiment logs and trained model artifacts are included
5. Submit final paper