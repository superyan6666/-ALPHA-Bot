# Handoff Report — Milestone 3 Factor Implementation

## 1. Observation
- **Codebase Files**: We observed the three target files for modifications at the root directory of the workspace:
  - `pipeline_manager.py` (containing `PipelineManager.load_data` and execution flow).
  - `factor_library.py` (containing factor decorators, `apply_factor` normalization function, and factor calculation registries).
  - `factor_screener.py` (containing `InitialScreener._calc_single_ic` and `InitialScreener.screen`).
- **Data Files**: We observed macroeconomic and index CSV files under `research/data/`:
  - `csi300_price.csv` (contains columns `Date`, `close`)
  - `csi300_pe.csv` (contains columns `Date`, `pe`)
  - `cn_10y_yield.csv` (contains columns `Date`, `yield_10y`)
  - `vix_data.csv` (contains columns `Date`, `vix`, `vix3m`)
- **Execution Run**: Running `python pipeline_manager.py` produced the following log:
  ```
  2026-06-27 09:46:19,957 - INFO - ALPHA FACTORY PIPELINE STARTED
  ...
  2026-06-27 09:46:21,663 - INFO - PIPELINE COMPLETE. 8 factors successfully promoted to production.
  ```
- **Promoted Factors output**: Checking the generated file `promoted_factors.json` confirmed that 8 factors were successfully written with status `"ACTIVE"` and history events:
  - `F_PE_Zscore_252`
  - `F_PE_Yield_Ratio`
  - `F_MACD_DIFF`
  - `F_VIX_TS`
  - `F_VIX_Momentum_10`
  - `F_VOL_5`
  - `F_MOM_10`
  - `F_BIAS_5`

## 2. Logic Chain
1. **Data Ingestion and Merging**:
   - *Observation*: The raw price CSV file `csi300_price.csv` has only daily `Date` and `close` price and does not have the macro indicators, which would cause macro factors to fail. It also lacks columns like `code`, `open`, and `volume` which are expected by the calculation functions.
   - *Action*: In `pipeline_manager.py`, we implemented a merge on the date index with `csi300_pe.csv`, `cn_10y_yield.csv`, and `vix_data.csv`, followed by forward filling. We renamed the index column `Date` to `date` and injected the mock columns `code = '000300.SH'`, `open = close`, and `volume = 1e6`.
2. **Normalization & Single-Asset Case**:
   - *Observation*: Cross-sectional normalization (MAD / Z-score grouped by date) yields NaN or 0 when there is only one asset.
   - *Action*: We modified `apply_factor` in `factor_library.py` to check `df['code'].nunique()`. If it is `> 1`, it performs cross-sectional normalization. Otherwise (single-asset case), it skips cross-sectional normalization and applies a rolling 252-day z-score normalization: `(series - series.rolling(252).mean()) / (series.rolling(252).std() + 1e-8)`.
3. **Macro/Valuation Factor Implementation**:
   - *Observation*: We need to calculate 8 new macro/valuation factors.
   - *Action*: We implemented the 8 factors decorated with `@registry.register` in `factor_library.py`. For the rolling/shifting operations, we grouped by `code` first (e.g. using `df.groupby('code')['pe'].transform(...)`) to keep them robust for both single-asset and multi-asset cases.
4. **Time-Series Monthly IC Calculation**:
   - *Observation*: The standard Rank IC is cross-sectional and fails when there is only one asset because daily correlation is undefined.
   - *Action*: In `factor_screener.py`, we updated `InitialScreener.screen` to set `target` to `fwd_ret` without subtracting a cross-sectional market return when `df['code'].nunique() == 1`. In `_calc_single_ic`, we checked `is_single_asset` and calculated the monthly time-series IC (Spearman correlation between the factor and the target for each month, and took the mean and t-statistic of these monthly correlations).

## 3. Caveats
- **Rolling Window Warm-up**: The rolling z-score normalization uses a 252-day window, meaning that the first 251 days of factor values will be `NaN`. Similarly, factors with rolling or shift windows (e.g. `PE_Gap_120`, `Yield_Momentum_20`, `VIX_Momentum_10`) will have leading `NaN` values. The screeners drop rows with `NaN` in target/factors, reducing the initial dataset size from 1,279 rows to 757 rows. This is expected and correct behavior.

## 4. Conclusion
- The pipeline loading, merging, normalization, calculation, and screening logic for both single-asset and multi-asset cases is successfully implemented.
- Running the pipeline successfully executes the screening, passes 17 factors in Stage 2, selects 8 non-collinear factors in Stage 3 using XGBoost gain importance, and registers them in `promoted_factors.json`.

## 5. Verification Method
- **Command to Run**: `python -m pytest tests/test_factor_pipeline.py`
  - This test runs the complete data merging, factor generation, and initial screening processes to ensure correctness.
- **Files to Inspect**:
  - `promoted_factors.json` (to verify that the 8 promoted factors are correctly written).
  - `pipeline_manager.py` (load_data method).
  - `factor_library.py` (apply_factor method and the 8 new registered factors).
  - `factor_screener.py` (_calc_single_ic method and screen method).
