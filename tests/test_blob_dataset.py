import numpy as np
import pandas as pd
import pytest
import sys
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue
from filecmp import cmp

@pytest.fixture
def blobs():
    return pd.read_csv("./test_datasets/blob.csv")

def test_blobs_clustering(blobs):
    c = clue.clusterer(0.8,5,1.5)
    c.read_data(blobs)
    c.run_clue()
    c.to_csv('./','blobs_output.csv')

    assert cmp('./blobs_output.csv', './test_datasets/blobs_truth.csv')

if __name__ == "__main__":
    c = clue.clusterer(0.8,5,1.5)
    c.read_data('./test_datasets/blob.csv')
    c.run_clue()
    c.cluster_plotter()
    c.to_csv('./test_datasets/', 'blobs_truth.csv')