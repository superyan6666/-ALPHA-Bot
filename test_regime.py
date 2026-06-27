import os
import pytest
import numpy as np
import pandas as pd
from regime_detector import RegimeDetector

@pytest.fixture
def temp_dir(tmpdir):
    data_dir = str(tmpdir.mkdir(".quantbot_data"))
    return data_dir

def test_extract_features_and_regime(temp_dir):
    detector = RegimeDetector(model_dir=temp_dir)
    
    # Generate 150 days of fake market data
    dates = pd.date_range("2020-01-01", periods=150, freq="B")
    
    # 2 stocks
    df1 = pd.DataFrame({
        'date': dates,
        'code': '000001',
        'close': np.exp(np.cumsum(np.random.normal(0.001, 0.01, 150))),
        'amount': np.random.uniform(50000000, 200000000, 150)
    })
    
    df2 = pd.DataFrame({
        'date': dates,
        'code': '000002',
        'close': np.exp(np.cumsum(np.random.normal(-0.001, 0.02, 150))),
        'amount': np.random.uniform(10000000, 80000000, 150)
    })
    
    df = pd.concat([df1, df2])
    
    # Extract features
    features = detector.extract_features(df)
    assert len(features) == 149 # diff drops the first row
    assert 'ret' in features.columns
    assert 'vol' in features.columns
    assert 'amt' in features.columns
    
    # Train
    success = detector.train(features)
    assert success is True
    assert os.path.exists(detector.model_path)
    
    # Predict
    regime = detector.get_current_regime(df)
    assert regime["state"] in [0, 1, 2]
    assert "期" in regime["label"]
    assert len(regime["probs"]) == 3
    assert np.isclose(sum(regime["probs"]), 1.0)
