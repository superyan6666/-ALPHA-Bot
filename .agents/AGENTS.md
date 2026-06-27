# A-Bot Project AI Guidelines

This file establishes the workspace-level customizations and ironclad rules for ANY AI agent operating within the A-Bot quantitative trading project. You must strictly adhere to these guidelines to prevent architectural decay, context window bloat, and system crashes.

## 1. Architectural Integrity & Context Management
- **Monolith Preservation (`main.py`)**: The `main.py` file is explicitly kept as a unified, large orchestrator (~2500 lines). DO NOT attempt to "refactor" or "modularize" it into separate files (like `data.py`, `signals.py`, `features.py`) unless the user explicitly overrides this rule. Keeping it monolithic ensures you (the AI) can see the entire data ingestion, feature generation, and execution pipeline in one context window without context fragmentation.
- **Physical Zones**:
  - `/scripts/archive/`: Contains old experimental, patch, and test scripts. Do not use these as reference for production logic.
  - `/research/`: Contains ML training (`train_*.py`), strategy backtesting (`wfo_*.py`), and data pipelines (`build_*.py`).
  - `/logs/` & `/outputs/`: Temporary dynamically generated files.

## 2. Dynamic Data Isolation & Anti-OOM (Wind Control)
- **Immediate Ignore Rule**: Heavy outputs (such as `*.log`, `*.png`, `.quantbot_data/` models, DB files, and `pushed_state.json`) must ALWAYS be routed to `/logs/` or `/outputs/`. 
- **No Global Scans**: Before running any file system search (like `grep` or directory trees), ensure you are ignoring the `/logs/`, `/outputs/`, and `.quantbot_data/` folders to prevent severe OOM crashes or context window blowouts.

## 3. Engineering Robustness & Magic Numbers
- **Config Externalization**: Do not hardcode new strategy thresholds, filter boundaries (e.g., PE > 0, Market Cap < 500e8), or technical logic weights inside the calculation functions. All magic numbers MUST be declared in the `Config` class at the top of `main.py`.
- **Global State Encapsulation**: Concurrency locks, semaphores, and consecutive failure states are NOT allowed in the global module scope. They must be encapsulated within singleton instances or specific manager classes like `DataProxy` to ensure multi-threading safety during backtests.
- **Fine-Grained Exceptions**: Ban the use of naked `except Exception: pass` around network calls. Trap explicit exceptions (e.g., `requests.exceptions.RequestException`) to ensure fallback telemetry isn't silenced.
