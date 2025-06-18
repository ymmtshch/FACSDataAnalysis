"""
FCS file processing utilities using fcsparser
"""
import fcsparser
import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, Dict, Any, Optional
import io

class FCSProcessor:
    """FCS file processor using fcsparser library"""
    
    def __init__(self):
        self.data = None
        self.metadata = None
        self.channels = None
        
    def load_fcs_file(self, file_path_or_buffer) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load FCS file using fcsparser
        
        Args:
            file_path_or_buffer: File path or buffer object
            
        Returns:
            Tuple of (data_dataframe, metadata_dict)
        """
        try:
            # Parse FCS file
            if hasattr(file_path_or_buffer, 'read'):
                # Handle uploaded file buffer
                metadata, data = fcsparser.parse(file_path_or_buffer, 
                                               meta_data_only=False, 
                                               output_format='DataFrame')
            else:
                # Handle file path
                metadata, data = fcsparser.parse(file_path_or_buffer, 
                                               meta_data_only=False, 
                                               output_format='DataFrame')
            
            self.data = data
            self.metadata = metadata
            self.channels = list(data.columns)
            
            return data, metadata
            
        except Exception as e:
            st.error(f"FCSファイルの読み込みに失敗しました: {str(e)}")
            return None, None
    
    def get_channel_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed channel information from metadata
        
        Returns:
            Dictionary with channel information
        """
        if not self.metadata:
            return {}
            
        channel_info = {}
        
        # Extract channel information from metadata
        for key, value in self.metadata.items():
            if key.startswith('$P') and key.endswith('N'):
                # Channel name
                param_num = key[2:-1]  # Extract parameter number
                channel_name = value
                
                # Initialize channel info
                if channel_name not in channel_info:
                    channel_info[channel_name] = {
                        'name': channel_name,
                        'parameter': param_num,
                        'range': None,
                        'gain': None,
                        'voltage': None
                    }
                
                # Look for additional parameter information
                range_key = f'$P{param_num}R'
                gain_key = f'$P{param_num}G'
                voltage_key = f'$P{param_num}V'
                
                if range_key in self.metadata:
                    channel_info[channel_name]['range'] = self.metadata[range_key]
                if gain_key in self.metadata:
                    channel_info[channel_name]['gain'] = self.metadata[gain_key]
                if voltage_key in self.metadata:
                    channel_info[channel_name]['voltage'] = self.metadata[voltage_key]
        
        return channel_info
    
    def get_basic_stats(self, channel: Optional[str] = None) -> Dict[str, Any]:
        """
        Get basic statistics for data
        
        Args:
            channel: Specific channel name, if None returns stats for all channels
            
        Returns:
            Dictionary with statistical information
        """
        if self.data is None:
            return {}
            
        if channel:
            if channel in self.data.columns:
                data_subset = self.data[channel]
                return {
                    'count': len(data_subset),
                    'mean': float(data_subset.mean()),
                    'median': float(data_subset.median()),
                    'std': float(data_subset.std()),
                    'min': float(data_subset.min()),
                    'max': float(data_subset.max()),
                    'q25': float(data_subset.quantile(0.25)),
                    'q75': float(data_subset.quantile(0.75))
                }
            else:
                return {}
        else:
            # Return stats for all channels
            stats = {}
            for col in self.data.columns:
                stats[col] = {
                    'count': len(self.data[col]),
                    'mean': float(self.data[col].mean()),
                    'median': float(self.data[col].median()),
                    'std': float(self.data[col].std()),
                    'min': float(self.data[col].min()),
                    'max': float(self.data[col].max()),
                    'q25': float(self.data[col].quantile(0.25)),
                    'q75': float(self.data[col].quantile(0.75))
                }
            return stats
    
    def apply_transformation(self, data: pd.DataFrame, 
                           transformation: str, 
                           channels: list = None) -> pd.DataFrame:
        """
        Apply transformation to data
        
        Args:
            data: Input dataframe
            transformation: Type of transformation ('log', 'asinh', 'biexp')
            channels: List of channels to transform, if None transform all
            
        Returns:
            Transformed dataframe
        """
        if channels is None:
            channels = data.columns.tolist()
            
        data_transformed = data.copy()
        
        for channel in channels:
            if channel in data.columns:
                if transformation == 'log':
                    # Log transformation (avoid log of negative values)
                    positive_data = np.maximum(data[channel], 1)
                    data_transformed[channel] = np.log10(positive_data)
                    
                elif transformation == 'asinh':
                    # Inverse hyperbolic sine transformation
                    data_transformed[channel] = np.arcsinh(data[channel] / 150)
                    
                elif transformation == 'biexp':
                    # Biexponential transformation (simplified version)
                    # This is a simplified implementation
                    pos_mask = data[channel] > 0
                    neg_mask = data[channel] <= 0
                    
                    data_transformed.loc[pos_mask, channel] = np.log10(
                        data.loc[pos_mask, channel])
                    data_transformed.loc[neg_mask, channel] = -np.log10(
                        -data.loc[neg_mask, channel] + 1)
        
        return data_transformed
    
    def subsample_data(self, data: pd.DataFrame, 
                          max_events: int = 10000) -> pd.DataFrame:
            """
            Subsample data for better performance
            
            Args:
                data: Input dataframe
                max_events: Maximum number of events
            
            Returns:
                Subsampled dataframe
            """
            if len(data) <= max_events:
                return data
            
            # Random sampling
            return data.sample(n=max_events, random_state=42).reset_index(drop=True)


    def load_and_process_fcs(uploaded_file, transformation='asinh', max_events=10000):
        """
        Load and process FCS file from uploaded file
    
        Args:
            uploaded_file: Streamlit uploaded file object
            transformation: Transformation to apply ('log', 'asinh', 'biexp', 'none')
            max_events: Maximum number of events to keep
        
        Returns:
            Tuple of (processed_data, metadata, processor_instance)
        """
        processor = FCSProcessor()
    
        # Load FCS file
        data, metadata = processor.load_fcs_file(uploaded_file)
    
        if data is None:
            return None, None, None
    
        # Apply transformation if requested
        if transformation != 'none':
            data = processor.apply_transformation(data, transformation)
    
        # Subsample if necessary
        if max_events and len(data) > max_events:
            data = processor.subsample_data(data, max_events)
    
        return data, metadata, processor

    def subsample_data(self, data: pd.DataFrame, 
                      max_events: int = 10000) -> pd.DataFrame:
        """
        Subsample data for better performance
        
        Args:
            data: Input dataframe
            max_events: Maximum number of events
            
        Returns:
            Subsampled dataframe
        """
        if len(data) <= max_events:
            return data
            
        # Random sampling
        return data.sample(n=max_events, random_state=42).reset_index(drop=True)


# クラス外の関数として定義
def load_and_process_fcs(uploaded_file, transformation='asinh', max_events=10000):
    """
    Load and process FCS file from uploaded file
    
    Args:
        uploaded_file: Streamlit uploaded file object
        transformation: Transformation to apply ('log', 'asinh', 'biexp', 'none')
        max_events: Maximum number of events to keep
        
    Returns:
        Tuple of (processed_data, metadata, processor_instance)
    """
    processor = FCSProcessor()
    
    try:
        # Load FCS file
        data, metadata = processor.load_fcs_file(uploaded_file)
        
        if data is None:
            return None, None, None
        
        # Apply transformation if requested
        if transformation != 'none':
            data = processor.apply_transformation(data, transformation)
        
        # Subsample if necessary
        if max_events and len(data) > max_events:
            data = processor.subsample_data(data, max_events)
        
        return data, metadata, processor
        
    except Exception as e:
        st.error(f"FCSファイルの処理中にエラーが発生しました: {str(e)}")
        return None, None, None
