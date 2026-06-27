# Project: A-Bot Factor Implementation

## Architecture
- **factor_library.py**: Contains factor definitions and the registry framework. Factors are registered via `@registry.register("FACTOR_NAME")` and compute series-based metrics from a Pandas DataFrame containing price/volume data.
- **factor_screener.py**: Initial Screener (Rank IC filter) and Fine Screener (XGBoost importance & collinearity filter) to select valid alpha factors.
- **pipeline_manager.py**: Loads base stock data, generates factors using `factor_library.py`, runs them through screening stages, and writes promoted factors to `promoted_factors.json`.

## Code Layout
- `factor_library.py`: Target file for factor implementation.
- `pipeline_manager.py`: Testing pipeline script.
- `promoted_factors.json`: Output registry file for promoted factors.
- `research/data/`: Input dataset directory.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Codebase Exploration | Analyze registration, input data format, and pipeline requirements | None | DONE |
| 2 | Factor Design | Design 5-10 novel quantitative factors based on price/volume data | M1 | DONE |
| 3 | Factor Implementation | Write the new factors into `factor_library.py` decorated with `@registry.register` | M2 | DONE |
| 4 | Pipeline Execution & Tuning | Execute `pipeline_manager.py` and iterate on factor formulas to ensure promotion | M3 | DONE |
| 5 | Verification & Audit | Validate correctness, avoid integrity issues, and run formal review | M4 | DONE |
| 6 | Victory Report | Document output and report victory to the Sentinel | M5 | DONE |

## Interface Contracts
### factor_library.py ↔ pipeline_manager.py
- Each factor function receives a Pandas DataFrame with stock price/volume metrics (open, close, high, low, volume, amount, etc.).
- Each factor returns a Pandas Series aligned with the DataFrame's index.
- No manual outlier clipping (MAD) or standardization (Z-score) should be applied within factor functions; the registry applies them automatically.
