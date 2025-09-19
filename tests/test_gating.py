import pytest
import pandas as pd
import numpy as np
from utils import gating

def test_threshold_gate():
    df = pd.DataFrame({'A': np.linspace(0, 10, 100)})
    gate = gating.ThresholdGate('A', 5, '>=')
    indices = gate.apply(df)
    assert all(df.loc[indices]['A'] >= 5)

def test_rectangular_gate():
    df = pd.DataFrame({'X': np.linspace(0, 10, 100), 'Y': np.linspace(0, 10, 100)})
    gate = gating.RectangularGate('X', 'Y', 2, 8, 2, 8)
    indices = gate.apply(df)
    assert all((df.loc[indices]['X'] >= 2) & (df.loc[indices]['X'] <= 8))
