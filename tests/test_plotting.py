import pytest
import pandas as pd
import numpy as np
from utils import plotting

def test_create_histogram():
    df = pd.DataFrame({'A': np.random.rand(100)})
    plotter = plotting.PlottingUtils()
    fig = plotter.create_histogram(df, 'A')
    assert fig is not None

def test_create_scatter_plot():
    df = pd.DataFrame({'X': np.random.rand(100), 'Y': np.random.rand(100)})
    plotter = plotting.PlottingUtils()
    fig = plotter.create_scatter_plot(df, 'X', 'Y')
    assert fig is not None
