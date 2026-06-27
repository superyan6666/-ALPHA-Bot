import os
import json
import pytest
import pandas as pd
import numpy as np
from gene_atlas import GeneAtlas

@pytest.fixture
def temp_dir(tmpdir):
    data_dir = str(tmpdir.mkdir(".quantbot_data"))
    return data_dir

def test_perfect_positive_correlation(temp_dir):
    atlas = GeneAtlas(data_dir=temp_dir)
    dates = pd.date_range("2020-01-01", periods=30, freq="ME")
    
    ic_df = pd.DataFrame({
        "F_A": np.linspace(0.1, 0.9, 30),
        "F_B": np.linspace(0.1, 0.9, 30), # Perfect positive correlation
    }, index=dates)
    
    assignments, n_clusters, stab = atlas._cluster(ic_df)
    
    # 既然两个完全正相关，距离为0，应该聚为同一簇
    assert assignments["F_A"] == assignments["F_B"]
    
def test_perfect_negative_correlation(temp_dir):
    atlas = GeneAtlas(data_dir=temp_dir)
    dates = pd.date_range("2020-01-01", periods=30, freq="ME")
    
    ic_df = pd.DataFrame({
        "F_A": np.linspace(0.1, 0.9, 30),
        "F_B": np.linspace(0.9, 0.1, 30), # Perfect negative correlation
    }, index=dates)
    
    assignments, n_clusters, stab = atlas._cluster(ic_df)
    
    # 完全负相关，距离大，应该不在同一簇
    assert assignments["F_A"] != assignments["F_B"]

def test_random_independent_factors(temp_dir):
    atlas = GeneAtlas(data_dir=temp_dir)
    dates = pd.date_range("2020-01-01", periods=30, freq="ME")
    
    np.random.seed(42)
    # Generate 5 random factors
    data = {f"F_{i}": np.random.randn(30) for i in range(5)}
    ic_df = pd.DataFrame(data, index=dates)
    
    assignments, n_clusters, stab = atlas._cluster(ic_df)
    
    # The conservative fallback for 5 factors is ceil(sqrt(5)) = 3
    # Our permutation test shouldn't find significant clusters in pure noise, or it might fall back.
    # At minimum, it shouldn't crash.
    assert len(set(assignments.values())) > 0

def test_enforce_diversity_empty(temp_dir):
    atlas = GeneAtlas(data_dir=temp_dir)
    assert atlas.enforce_diversity([]) == []

def test_enforce_diversity_same_gene(temp_dir):
    atlas = GeneAtlas(data_dir=temp_dir)
    atlas.atlas_data = {
        "factors": {
            "F_1": {"gene_id": 1, "gene_label": "G1"},
            "F_2": {"gene_id": 1, "gene_label": "G1"},
            "F_3": {"gene_id": 1, "gene_label": "G1"}
        }
    }
    
    res = atlas.enforce_diversity(["F_1", "F_2", "F_3"], max_per_gene=2)
    assert len(res) == 2
    assert res == ["F_1", "F_2"]

def test_enforce_diversity_all_different(temp_dir):
    atlas = GeneAtlas(data_dir=temp_dir)
    atlas.atlas_data = {
        "factors": {
            "F_1": {"gene_id": 1, "gene_label": "G1"},
            "F_2": {"gene_id": 2, "gene_label": "G2"},
            "F_3": {"gene_id": 3, "gene_label": "G3"}
        }
    }
    
    res = atlas.enforce_diversity(["F_1", "F_2", "F_3"], max_per_gene=1)
    assert len(res) == 3
    assert res == ["F_1", "F_2", "F_3"]

def test_load_missing_fields(temp_dir):
    atlas = GeneAtlas(data_dir=temp_dir)
    bad_data = {"some_key": 123}
    with open(atlas.atlas_path, "w") as f:
        json.dump(bad_data, f)
        
    loaded = atlas.load()
    # It should load the bad data but log a warning, not crash.
    assert loaded == bad_data
    
    # And it shouldn't crash when using enforce_diversity because it checks for "factors"
    assert atlas.enforce_diversity(["F_1"]) == ["F_1"]
