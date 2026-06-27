import pytest
import os
import pandas as pd
import numpy as np
from pipeline_manager import PipelineManager
from factor_library import calculate_factors, registry, apply_factor
from factor_screener import InitialScreener, FineScreener

def test_data_merging_and_injection():
    # Verify load_data works on the csi300_price.csv dataset
    manager = PipelineManager(data_path="research/data/csi300_price.csv")
    df = manager.load_data()
    
    assert "date" in df.columns
    assert "close" in df.columns
    assert "pe" in df.columns
    assert "yield_10y" in df.columns
    assert "vix" in df.columns
    assert "vix3m" in df.columns
    assert (df["code"] == "000300.SH").all()
    assert (df["open"] == df["close"]).all()
    assert (df["volume"] == 1e6).all()
    assert not df["close"].isna().any()

def test_factor_calculations():
    # Load data
    manager = PipelineManager(data_path="research/data/csi300_price.csv")
    df = manager.load_data()
    
    # Calculate factors
    df_factors = calculate_factors(df)
    
    # Check that 8 new factors exist in the resulting DataFrame
    new_factors = ["ERP", "VIX_TS", "PE_Zscore_252", "Yield_Momentum_20", 
                   "PE_Yield_Ratio", "VIX_Momentum_10", "VIX_Volatility_20", "PE_Gap_120"]
    
    for f in new_factors:
        col = f"F_{f}"
        assert col in df_factors.columns
        # Since it is single asset, they should be z-score standardized and not all NaN (except first few due to rolling)
        valid_count = df_factors[col].notna().sum()
        assert valid_count > 0, f"Factor {col} is all NaN"

def test_single_asset_screener():
    # Load data and calculate factors
    manager = PipelineManager(data_path="research/data/csi300_price.csv")
    df = manager.load_data()
    df_factors = calculate_factors(df)
    
    factor_cols = [c for c in df_factors.columns if c.startswith('F_') and c != 'F_float_cap']
    
    # Run initial screen
    screener = InitialScreener(target_ret_window=20, max_workers=2)
    screened_df, passed_factors = screener.screen(df_factors, factor_cols)
    
    assert "target" in screened_df.columns
    assert not screened_df["target"].isna().any()
    assert len(passed_factors) > 0
