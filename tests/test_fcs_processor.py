import pytest
import pandas as pd
from utils import fcs_processor

def test_fcs_processor_initialization():
    processor = fcs_processor.FCSProcessor(pd.DataFrame())
    assert processor is not None

def test_export_data_empty():
    processor = fcs_processor.FCSProcessor(pd.DataFrame())
    csv = processor.export_data(pd.DataFrame(), data_type="data")
    assert isinstance(csv, str)
