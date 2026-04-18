# Preprocessing Pipeline — Issue #1

## What it does

Loads raw CICIDS2017 network flow data and prepares it for the ML model.

## Steps

1. Load CSV → strips whitespace from column names
2. Clean → removes 161 null/infinity/duplicate rows from 2,520,751 records
3. Extract → selects 11 relevant network flow features from 53 columns
4. Scale → StandardScaler normalizes all values to mean=0, std=1

## Features Used

- Flow Duration
- Total Fwd Packets
- Total Length of Fwd Packets
- Fwd Packet Length Max
- Fwd Packet Length Min
- Fwd Packet Length Mean
- Bwd Packet Length Max
- Bwd Packet Length Min
- Flow Bytes/s
- Flow Packets/s
- Packet Length Mean

## Output
- df_scaled → (2,520,590 x 11) normalized, ready for ML model
- df_raw    → original unmodified dataframe, for audit trail
- df_clean  → cleaned dataframe, used for risk scoring and reports

## Dataset

CICIDS2017 — cicids2017_cleaned.csv — 2,520,751 records, 53 columns

## Next Step
Output feeds into: [Anomaly Detection](anomaly_detection.md)
