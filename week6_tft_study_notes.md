# Week 6, Step 1: Temporal Fusion Transformer study notes

Source: Lim, Arik, Loeff and Pfister, "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" (International Journal of Forecasting, 2021). These notes are my own summary for this project, checked against the arXiv version of the paper (1912.09363), Sections 4.1 to 4.6. Equation numbers below refer to that version. Re-check any wording before quoting it in the manuscript.

## 1. Why a TFT here

The GraphSAGE baseline sees one frame at a time. It scores each aircraft pair from that moment's features. A TFT reads a window of past frames for a pair, so it can use how separation and closing speed changed over the last few minutes.

The TFT was designed for forecasting with a mix of inputs, and each type maps onto our data:

| TFT input type | Meaning | In this project |
|---|---|---|
| Static covariates | Fixed for the whole series | Pair-level constants, for example the aircraft pair's median altitude band or region (to be decided) |
| Observed past inputs | Known only up to now | Lateral separation, vertical separation, closing speed, bearing difference, time to closest approach, altitude, ground speed |
| Known future inputs | Known ahead of time | Probably none, except time of day |

## 2. The three parts named in the plan

### Gated residual network (GRN), Section 4.1
- The basic building block used throughout the model.
- Steps (Eqs. 2-4): a linear layer and an ELU activation on the input (plus an optional context vector), a second linear layer, a gated linear unit (GLU), the result added to the original input, then layer normalization. Dropout is applied before the gate during training.
- The GLU is a sigmoid gate multiplied by a linear transform (Eq. 5). When the gate outputs are near 0, the nonlinear part is suppressed and the layer is effectively skipped.
- The paper's motivation is that simpler processing can be better when data is small or noisy. Value for us: our data is small, so this fits.

### Variable selection networks (VSN), Section 4.2
- Continuous inputs are each turned into a vector by a linear transformation (categoricals use embeddings). Static, past and future inputs each have their own VSN.
- At each time step the flattened inputs, plus a static context vector, go through a GRN and a softmax (Eq. 6). The result is one weight per variable, summing to 1, so the weights can change over time and between samples.
- Each variable also passes through its own GRN, shared across time steps (Eq. 7), and the outputs are summed using the weights (Eq. 8).
- Value for us: the weights show which features the model relied on. Section 7.1 of the paper does this by looking at the 10th, 50th and 90th percentiles of the weights over the test set. This feeds the Week 11 interpretability step.

### Interpretable multi-head attention, Section 4.4
- Standard multi-head attention uses different value projections per head, so a head's attention weights are hard to compare or read.
- The TFT uses one value projection shared by all heads and averages the heads' attention matrices (Eqs. 13-16). Each head can still learn a different temporal pattern, but the average can be read as a single attention pattern over time steps.
- Section 4.5.3 applies decoder masking, so a time step only attends to earlier ones.
- Value for us: we can see which earlier frames, such as the moment two aircraft began converging, mattered for a prediction.

## 3. The rest of the pipeline, briefly

1. Variable selection and static covariate encoders prepare the inputs. The encoders (Section 4.3) are GRNs that produce four context vectors for variable selection, the LSTM initial states, and static enrichment.
2. An LSTM encoder reads the past window and an LSTM decoder handles the forecast horizon (Section 4.5.1). This replaces positional encoding. A gated skip connection goes over it.
3. A static enrichment GRN, then the masked interpretable attention layer with a gate, then a position-wise GRN with a gated skip over the whole block (Sections 4.5.2 to 4.5.4).
4. The output layer predicts quantiles (10th, 50th, 90th) trained with quantile loss (Section 4.6, Eqs. 23-25). We will replace it with a binary classification head and a classification loss in Week 8. Our windows have no known future inputs, so the decoder side is mostly unused.

## 4. Design questions to settle before building (Weeks 7-8)

1. **What does the model predict?** The current label is a direct function of lateral and vertical separation. If the TFT only classifies the current frame, it can copy the rule, as the earlier experiments showed. A more meaningful task is to predict whether a pair will meet the rule some number of seconds ahead. We have not chosen a horizon yet, and this should be discussed with the professor, together with the unresolved label definition.
2. **Window length.** For example, 30 frames at 10 seconds is 5 minutes of history. Observed track durations are median about 16 minutes (complete segments), so long windows will drop many pairs.
3. **Splits.** Windows from the same encounter overlap heavily. Splits must keep whole sessions together, as in the current split file.
4. **Imbalance.** Positives are about 0.6% of pair-observations. The imbalance method will be decided after the systematic comparison the professor asked for. SMOTE is the working choice for now.
5. **Library.** pytorch-forecasting is named in the plan. It is built around regression and forecasting, so a custom classification head and loss need checking against its API.

## 5. Study checklist

- [x] Read the paper's Sections 4.1 to 4.6 (GRN, variable selection, static encoders, attention, decoder), done by the assistant on 2026-10-09; the user has not read it yet.
- [x] Explain each of the three parts (see Section 2).
- [x] Sketch observed and static inputs (see Section 1). All six model features are observed past inputs; static covariates are optional.
- [x] Decide the prediction target and horizon: risk within the next 60 s (6 frames), 30-frame windows, with `onset_only` as the harder variant. Implemented in `src/features/sequences.py`. The professor has not confirmed this yet.
