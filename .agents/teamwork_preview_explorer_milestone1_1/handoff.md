# Handoff Report — Milestone 1 Exploration

## 1. Observation
- **Data Files**: We inspected the files in `research/data/` and found they only contain daily macroeconomic/index-level columns rather than multi-asset panel data:
  - `research/data/csi300_price.csv` contains columns: `Date,close`
  - `research/data/csi300_pe.csv` contains columns: `Date,pe`
  - `research/data/cn_10y_yield.csv` contains columns: `Date,yield_10y`
  - `research/data/vix_data.csv` contains columns: `Date,vix,vix3m`
- **Verbatim Error**: Running `pipeline_manager.py` using `python pipeline_manager.py` failed with:
  ```
  Traceback (most recent call last):
    File "D:\Antigravity\A-Bot\pipeline_manager.py", line 98, in <module>
      manager.run_pipeline()
    File "D:\Antigravity\A-Bot\pipeline_manager.py", line 41, in run_pipeline
      df = calculate_factors(df)
    File "D:\Antigravity\A-Bot\factor_library.py", line 111, in calculate_factors
      df = df.sort_values(['code', 'date']).reset_index(drop=True)
    KeyError: 'code'
  ```
- **Code Structure**:
  - `factor_library.py` (lines 111-117) uses `df = df.sort_values(['code', 'date'])` and iterates through factors, applying `mad_outlier` and `zscore_standardize` grouped by `'date'`.
  - `factor_screener.py` (line 36) shifts `'open'` to calculate returns: `df['next_open'] = df.groupby('code')['open'].shift(-1)`.
  - `factor_screener.py` (line 17) calculates Spearman correlation per cross-section: `df.groupby('date').apply(...)`.
- **Prototype Script Output**: Running the prototype script `test_pipeline.py` (which implements data merging, mock columns, time-series standardization, and monthly time-series IC) resulted in successful screening and promotion of 8 factors:
  - Promoted list: `['F_PE_Zscore_252', 'F_PE_Yield_Ratio', 'F_MACD_DIFF', 'F_VIX_TS', 'F_VIX_Momentum_10', 'F_VOL_5', 'F_MOM_10', 'F_BIAS_5']`.

## 2. Logic Chain
1. Since the dataset `csi300_price.csv` lacks the `'code'` column, sorting by `['code', 'date']` in `factor_library.py` raises `KeyError: 'code'` (supported by the verbatim traceback observation).
2. Because the dataset lacks `'open'` and `'volume'` columns, any factor relying on volume (like `VP_REV_w`) or screening step shifting `'open'` raises `KeyError` (supported by code observation in `factor_screener.py` and `factor_library.py`).
3. Because there is only a single asset in the input data (group size of 1 per date), standardizing factor values cross-sectionally grouped by `'date'` results in `NaN` standard deviation, causing all factor values to become `NaN` and drop out during cleanup (supported by standard deviation behavior of size-1 groups).
4. If we rename `'Date'` to `'date'`, merge PE, Yield, and VIX datasets on date, and add mock columns (`code = '000300.SH'`, `open = close`, `volume = 1e6`), we avoid all initial `KeyError` exceptions (supported by `test_pipeline.py` successful loading logs).
5. If we replace cross-sectional z-scoring with time-series z-scoring (rolling 252-day z-score) and replace cross-sectional Rank IC with monthly time-series IC, the pipeline can successfully screen and rank factors without encountering `NaN` errors (supported by successful execution of `test_pipeline.py` and outputting promoted factors).

## 3. Caveats
- Volume-based factors (like `VP_REV_w`) will return flat/zero values because volume is mocked as a constant `1e6`.
- The forward return calculation uses `open = close` as a fallback, which approximates close-to-close return rather than open-to-close return.
- Pearson correlation threshold in FineScreener is set to `0.7` as specified in the plan (default in codebase was `0.8`).

## 4. Conclusion
The codebase is structured for cross-sectional stock screening, but the available dataset is a single-asset index/macro time series. To run the pipeline successfully on the available dataset:
- `pipeline_manager.py` must load and merge the available CSV files (`csi300_price.csv`, `csi300_pe.csv`, `cn_10y_yield.csv`, `vix_data.csv`).
- Mock/fallback columns (`code`, `open`, `volume`) must be injected.
- The factor z-score normalization and Rank IC calculation must be adapted from cross-sectional grouping to time-series calculations.

## 5. Verification Method
- **Command to run**:
  Execute the prototype test script in the agent folder:
  `python .agents/teamwork_preview_explorer_milestone1_1/test_pipeline.py`
- **Expected results**:
  The script loads the datasets, generates 23 factors (including 8 new macro/valuation factors), calculates monthly time-series IC, trains an XGBoost Regressor, applies collinearity filtering, and outputs the final promoted factors list.
