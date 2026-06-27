# Original User Request

## Initial Request — 2026-06-27T01:42:15Z

# Teamwork Project Prompt

[Project description: Explore and implement a broad set (5-10) of novel quantitative factors into the A-Bot Alpha Factory using the new `@registry.register` framework, and automatically run them through the local evaluation pipeline to see which ones survive.]

Working directory: `d:\Antigravity\A-Bot`
Integrity mode: demo (Team is allowed to copy/adapt core logic from open-source quant libraries like WorldQuant Alpha 101).

## Requirements

### R1. Factor Breadth (广度优先)
Implement 5 to 10 distinct, lightweight quantitative factors into `factor_library.py`. You may draw inspiration from open-source libraries (e.g., Alpha 101). Focus on pure price/volume data available in the input DataFrame (open, close, high, low, volume, amount).

### R2. Factor Registration
Every new factor must be decorated with `@registry.register("YOUR_FACTOR_NAME")` and must return a Pandas Series aligned with the DataFrame's index. Do NOT apply MAD or Z-score normalization manually; the registry handles this automatically.

### R3. Local Pipeline Execution
Once the factors are written to `factor_library.py`, you must execute the local testing pipeline by running:
`python pipeline_manager.py`
The execution environment is the local Windows machine where the codebase resides.

## Acceptance Criteria

### Execution & Survival
- [ ] The team successfully writes 5-10 new factors into `factor_library.py` without syntax errors.
- [ ] The command `python pipeline_manager.py` is executed and finishes without Python traceback errors.
- [ ] At least ONE newly implemented factor successfully passes both the Initial Screen (Rank IC) and Fine Screen (XGBoost) and gets automatically written into `promoted_factors.json` by the pipeline. If 0 factors pass on the first run, the team must analyze the logs, adjust the factor logic, and retry until at least one factor survives.
