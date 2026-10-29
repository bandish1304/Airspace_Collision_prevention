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
2. Define the TFT input features, encoder length, prediction horizon, and prediction target for ADS-B trajectories

WEEK 7 - TFT Input Pipeline and Binary Classifier Implementation
1. Build sliding-window trajectory sequences per aircraft pair
2. Construct a TimeSeriesDataSet (or custom dataset) formatted for TFT input
3. Split into train/validation/test sets with proper temporal separation
4. Implement/configure the Temporal Fusion Transformer with PyTorch and pytorch-forecasting
5. Adapt the TFT output for binary collision-risk classification
6. Apply the Week 3 selected SMOTE strategy to training data only; leave validation and test data unchanged
7. Run a short training pass to confirm the input pipeline, classifier, and SMOTE integration work end-to-end

WEEK 8 - TFT Tuning, Evaluation, and Midterm Draft
1. Train and tune the Week 7 TFT using Optuna, searching attention heads, hidden size, dropout, and learning rate; select settings using validation data only
2. Prepare a midterm rough draft of the research paper summarizing the GNN baseline and initial TFT development

WEEK 9 - TFT Final Training
1. Train the selected TFT configuration to convergence with early stopping and learning-rate scheduling
2. Save the final tuned model and training configuration
3. Log Optuna trials, the final tuned model, and training details to MLflow

WEEK 10 - Main Comparison (GNN vs TFT)
1. Evaluate the tuned GNN and TFT once each on the same held-out test set
2. Compute shared-harness metrics for both models, including precision, recall, F1, AUROC, average precision, and false negative rate
3. Run statistical significance tests on the metric differences
4. Produce a comparison table and figure for the paper

WEEK 11 - Interpretability
1. Extract TFT attention weights from the trained model
2. Extract TFT variable selection scores from the trained model
3. Build visualizations (heatmaps) of feature and time-step importance
4. Interpret findings in relation to known collision precursors, noting limitations

WEEK 12 - Ablation Studies
1. Create a TFT variant without the variable selection network
2. Create a TFT variant with a shortened input sequence
3. Create a TFT variant without static covariates
4. Retrain and evaluate each variant using the shared evaluation harness
5. Compare each variant with the full TFT to estimate component contributions

WEEK 13 - Drift Analysis and Visualization
1. Split historical data into distinct time windows
2. Measure feature-distribution shifts between windows
3. Compare GNN and TFT performance across windows
4. Build a static visualization of aircraft positions and predicted risk for a sample time window

WEEK 14 - Manuscript Drafting
1. Draft the Introduction and Related Work sections
2. Draft the Methodology and Experimental Setup sections
3. Draft the Results section with figures and tables from Weeks 10-13
4. Draft the Discussion and Conclusion sections

WEEK 15 - Finalize and Submit
1. Revise the manuscript based on self-review
2. Clean up the code repository
3. Update the README and requirements file
4. Verify MLflow experiment logs and required model artifacts are available
5. Submit the final paper
