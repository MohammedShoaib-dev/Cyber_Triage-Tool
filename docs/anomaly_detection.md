# Anomaly Detection — Issue #2

## Algorithm: Isolation Forest

### How it works

Normal points require many random cuts to isolate → long path length → low anomaly score
Anomalous points are isolated quickly → short path length → high anomaly score

### Why Isolation Forest?

- Works on unlabeled data (unsupervised) — perfect for forensics where attacks are unknown
- Handles large datasets efficiently — tested on 2.5M records
- No assumption about data distribution
- Built-in anomaly scoring, not just binary classification

## Model Configuration

| Parameter     | Value | Reason                                           |
| ------------- | ----- | ------------------------------------------------ |
| n_estimators  | 100   | 100 isolation trees, balances speed and accuracy |
| contamination | 0.1   | Expects ~10% anomalous records in dataset        |
| random_state  | 42    | Reproducible results                             |
| n_jobs        | -1    | Uses all CPU cores for speed                     |

## Test Results

| Metric             | Value                               |
| ------------------ | ----------------------------------- |
| Total records      | 2,520,590                           |
| Anomalies detected | 252,059                             |
| Normal records     | 2,268,531                           |
| Anomaly rate       | 10% (matches contamination setting) |

## Output

- predictions → array of 1 (normal) or -1 (anomaly) per record
- scores → raw decision function values, negative = more anomalous
- Saved model → models/isolation_forest.pkl

## Methods

- train(X) → fits model on scaled feature data
- predict(X) → returns predictions and raw scores
- get_anomaly_score_normalized(score) → converts raw score to 0-100 scale
- save_model(path) → saves trained model as .pkl file
- load_model(path) → loads saved model for reuse

## References

Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008).
Isolation Forest. IEEE International Conference on Data Mining.
