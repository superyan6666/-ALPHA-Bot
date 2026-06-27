## 2026-06-27T01:45:17Z

You are Milestone 3 Factor Implementer. Your working directory is d:\Antigravity\A-Bot\.agents\teamwork_preview_worker_milestone3_1.
Your task is:
1. Review the exploration report at `d:\Antigravity\A-Bot\.agents\teamwork_preview_explorer_milestone1_1\analysis.md` and the prototype code in `d:\Antigravity\A-Bot\.agents\teamwork_preview_explorer_milestone1_1\test_pipeline.py`.
2. Implement data merging and mock column injection in `pipeline_manager.py` (load and merge `csi300_price.csv`, `csi300_pe.csv`, `cn_10y_yield.csv`, and `vix_data.csv` on the date index, forward fill, rename `Date` to `date`, and inject mock columns `code = '000300.SH'`, `open = close`, `volume = 1e6`).
3. Modify `factor_library.py` to:
   - Handle single-asset case in `apply_factor` (if `df['code'].nunique() > 1`, do cross-sectional normalization; otherwise, skip cross-sectional normalization and apply rolling 252-day z-score normalization: `(series - series.rolling(252).mean()) / (series.rolling(252).std() + 1e-8)`).
   - Implement the 8 new macroeconomic/valuation factors decorated with `@registry.register`:
     - `ERP`: `(100.0 / df['pe']) - df['yield_10y']`
     - `VIX_TS`: `df['vix'] / df['vix3m'] - 1.0`
     - `PE_Zscore_252`: `(df['pe'] - df['pe'].rolling(252).mean()) / (df['pe'].rolling(252).std() + 1e-8)`
     - `Yield_Momentum_20`: `df['yield_10y'] - df['yield_10y'].shift(20)`
     - `PE_Yield_Ratio`: `df['pe'] * df['yield_10y']`
     - `VIX_Momentum_10`: `df['vix'] - df['vix'].shift(10)`
     - `VIX_Volatility_20`: `df['vix'].rolling(20).std()`
     - `PE_Gap_120`: `df['pe'] / df['pe'].rolling(120).mean() - 1.0`
4. Modify `factor_screener.py` to:
   - Handle single-asset case in `InitialScreener.screen` and `_calc_single_ic`: if `df['code'].nunique() == 1`, calculate **monthly time-series IC** (Spearman correlation between factor and target for each month, and take the mean and t-statistic of these monthly correlations). Otherwise, do standard cross-sectional Rank IC.
5. Run the local testing pipeline `python pipeline_manager.py` to verify the factors compile, execute without errors, and are successfully promoted. Check that at least one factor gets written to `promoted_factors.json`.
6. Write a comprehensive report `handoff.md` summarizing the changes, the command executed, the execution logs, and the list of promoted factors.
7. Send a message back to the orchestrator (conversation ID: 7b35a5ef-6ad5-43e6-93b0-a3f99cdb3f36).

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.
