# Adversarial Review Report

## Challenge Summary

**Overall risk assessment**: MEDIUM

While the pipeline is functional and correct for the standard CSI 300 price dataset, several structural assumptions could lead to failures if the data format changes or under resource constraints.

## Challenges

### [High] Challenge 1: Data Loader Hardcoding & Pipeline Failures
- **Assumption challenged**: The file name checks and column expectations in `pipeline_manager.py` (line 23) assume that only CSI 300 data requires macro fields, and that the path contains `"csi300_price.csv"`.
- **Attack scenario**: If a user runs the pipeline on a newly exported price dataset (e.g. `csi300_price_2026.csv`), the filename check fails. The loader bypasses merging, and factor calculations immediately crash with `KeyError: 'pe'` or `KeyError: 'yield_10y'`.
- **Blast radius**: Complete pipeline crash during the generation phase for any price file not named exactly `csi300_price.csv`.
- **Mitigation**: Update the condition to check if the file belongs to an index index/group or check if macro CSVs are present in the folder, rather than matching a specific string pattern.

### [Medium] Challenge 2: Division by Zero in Factor Formulas
- **Assumption challenged**: The factor `ERP` calculates `100.0 / df['pe']` and `PE_Gap_120` calculates `df['pe'] / pe_mean - 1.0`.
- **Attack scenario**: If `pe` is zero (for example, in mock data or for companies with zero earnings in a multi-asset dataset) or if `pe_mean` is zero, these operations will trigger division-by-zero errors.
- **Blast radius**: Propagates `inf` or `NaN` values, or raises runtime warnings/exceptions that can disrupt statistical scoring and XGBoost training.
- **Mitigation**: Add a small epsilon to the denominator or filter out non-positive PE values before computing the ratios: e.g., `100.0 / (df['pe'] + 1e-8)`.

### [Medium] Challenge 3: Rolling Window Length vs. Data Length
- **Assumption challenged**: The rolling normalization window for a single asset is fixed at 252 days (`series.rolling(252)`).
- **Attack scenario**: If the price dataset has fewer than 252 rows (e.g., a newly listed stock or a short mock dataset for testing), the rolling mean and standard deviation will be entirely `NaN`.
- **Blast radius**: The single-asset normalization will result in a DataFrame filled entirely with NaNs. This will fail the initial screening phase (since mean IC and t-stat cannot be computed), aborting the pipeline execution.
- **Mitigation**: Dynamically adjust the rolling window size based on the dataset length (e.g., `min(len(df), 252)`) or enforce a minimum length requirement before running the pipeline.

### [Low] Challenge 4: Zero Volatility / Constant Factors
- **Assumption challenged**: Spearman correlation in the monthly Rank IC calculation assumes the factor and target have non-zero variance.
- **Attack scenario**: If the price has been suspended or is constant (e.g. mock data or price limit), or if a factor is constant during a calendar month, `corr` returns `NaN`.
- **Blast radius**: Monthly IC becomes `NaN`, reducing the number of valid months. If the number of valid months falls below 10, the factor is excluded.
- **Mitigation**: Check for zero variance before computing correlation and log a warning.

## Stress Test Results

- **Run pipeline on CSI300 price file** → Runs successfully and outputs 7 promoted factors → **PASS**
- **Run pipeline with fewer than 252 rows** → Z-scores become NaN, screener fails to find factors → **FAIL** (as predicted)
- **Run pipeline with renamed price file (e.g., `csi300_price_v2.csv`)** → Skips merging, crashes on factor calculation → **FAIL** (as predicted)

## Unchallenged Areas

- **XGBoost Hyperparameters**: The learning rate and estimator parameters are assumed to be optimal and stable. No hyperparameter search or cross-validation was tested.
