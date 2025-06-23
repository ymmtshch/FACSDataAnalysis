"""
FCS file processing utilities with multi-library support
"""
import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, Dict, Any, Optional
import io

class FCSProcessor:
    """FCS file processor with automatic library selection"""
    
    def __init__(self):
        self.data = None
        self.metadata = None
        self.channels = None
        self.used_library = None
        
    def _try_flowio(self, file_path_or_buffer):
        """Try loading with flowio library"""
        try:
            import flowio
            
            # Reset buffer position if it's a buffer
            if hasattr(file_path_or_buffer, 'seek'):
                file_path_or_buffer.seek(0)
                
            fcs = flowio.FlowData(file_path_or_buffer)
            
            # Convert to pandas DataFrame
            data = pd.DataFrame(fcs.events, columns=fcs.channels['PnN'])
            metadata = fcs.text
            
            self.used_library = 'flowio'
            return data, metadata
            
        except ImportError:
            return None, None
        except Exception as e:
            st.debug(f"flowio loading failed: {str(e)}")
            return None, None
    
    def _try_flowkit(self, file_path_or_buffer):
        """Try loading with flowkit library"""
        try:
            import flowkit
            
            # Reset buffer position if it's a buffer
            if hasattr(file_path_or_buffer, 'seek'):
                file_path_or_buffer.seek(0)
                
            sample = flowkit.Sample(file_path_or_buffer)
            
            # Get data and metadata
            data = sample.as_dataframe()
            metadata = sample.metadata
            
            self.used_library = 'flowkit'
            return data, metadata
            
        except ImportError:
            return None, None
        except Exception as e:
            st.debug(f"flowkit loading failed: {str(e)}")
            return None, None
    
    def _try_fcsparser(self, file_path_or_buffer):
        """Try loading with fcsparser library"""
        try:
            import fcsparser
            
            # Reset buffer position if it's a buffer
            if hasattr(file_path_or_buffer, 'seek'):
                file_path_or_buffer.seek(0)
                
            metadata, data = fcsparser.parse(file_path_or_buffer, 
                                           meta_data_only=False, 
                                           output_format='DataFrame')
            
            self.used_library = 'fcsparser'
            return data, metadata
            
        except ImportError:
            return None, None
        except Exception as e:
            st.debug(f"fcsparser loading failed: {str(e)}")
            return None, None
        
    def load_fcs_file(self, file_path_or_buffer) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load FCS file using automatic library selection
        Priority: flowio → flowkit → fcsparser
        
        Args:
            file_path_or_buffer: File path or buffer object
            
        Returns:
            Tuple of (data_dataframe, metadata_dict)
        """
        # Try libraries in order of preference
        loaders = [
            ('flowio', self._try_flowio),
            ('flowkit', self._try_flowkit),
            ('fcsparser', self._try_fcsparser)
        ]
        
        for lib_name, loader_func in loaders:
            try:
                data, metadata = loader_func(file_path_or_buffer)
                if data is not None and metadata is not None:
                    self.data = data
                    self.metadata = metadata
                    self.channels = list(data.columns)
                    st.sidebar.success(f"使用ライブラリ: {lib_name}")
                    return data, metadata
            except Exception as e:
                st.sidebar.warning(f"{lib_name} での読み込みに失敗: {str(e)}")
                continue
        
        # All libraries failed
        st.error("すべてのFCS読み込みライブラリで失敗しました。flowio、flowkit、fcsparserのいずれかをインストールしてください。")
        return None, None
    
    def get_file_info(self) -> Dict[str, Any]:
        """
        Get file information from metadata
        
        Returns:
            Dictionary with file information
        """
        if not self.metadata:
            return {}
        
        info = {}
        
        # Handle different library metadata formats
        if self.used_library == 'flowkit':
            # flowkit uses lowercase keys
            info['total_events'] = self.metadata.get('tot', self.metadata.get('$TOT', 'N/A'))
            info['total_parameters'] = self.metadata.get('par', self.metadata.get('$PAR', 'N/A'))
            info['acquisition_date'] = self.metadata.get('date', self.metadata.get('$DATE', 'N/A'))
            info['acquisition_time'] = self.metadata.get('btim', self.metadata.get('$BTIM', 'N/A'))
            info['cytometer'] = self.metadata.get('cyt', self.metadata.get('$CYT', 'N/A'))
        else:
            # Standard FCS metadata keys
            info['total_events'] = self.metadata.get('$TOT', 'N/A')
            info['total_parameters'] = self.metadata.get('$PAR', 'N/A')
            info['acquisition_date'] = self.metadata.get('$DATE', 'N/A')
            info['acquisition_time'] = self.metadata.get('$BTIM', 'N/A')
            info['cytometer'] = self.metadata.get('$CYT', 'N/A')
        
        info['experiment_name'] = self.metadata.get('$EXP', 'N/A')
        info['sample_id'] = self.metadata.get('SAMPLE ID', 'N/A')
        info['operator'] = self.metadata.get('$OP', 'N/A')
        info['software'] = self.metadata.get('$SRC', 'N/A')
        info['used_library'] = self.used_library
        
        return info
    
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
                
                # Handle duplicate channel names
                original_name = channel_name
                counter = 2
                while channel_name in channel_info:
                    channel_name = f"{original_name}_{counter}"
                    counter += 1
                
                # Initialize channel info
                channel_info[channel_name] = {
                    'name': channel_name,
                    'original_name': original_name,
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
                try:
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
                except Exception as e:
                    st.warning(f"統計計算エラー ({col}): {str(e)}")
                    stats[col] = {'error': str(e)}
            return stats
    
    def preprocess_data(self, data: pd.DataFrame, meta: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess raw FCS data
        
        Args:
            data: Raw FCS data
            meta: FCS metadata
            
        Returns:
            Preprocessed DataFrame
        """
        processed_data = data.copy()
        
        # Handle different data types from different libraries
        if self.used_library == 'flowio':
            # flowio might return array.array objects
            for col in processed_data.columns:
                if hasattr(processed_data[col].iloc[0], '__iter__') and not isinstance(processed_data[col].iloc[0], str):
                    try:
                        processed_data[col] = processed_data[col].apply(lambda x: np.array(x) if hasattr(x, '__iter__') else x)
                    except:
                        pass
        
        # Ensure all columns are numeric
        for col in processed_data.columns:
            try:
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
            except:
                pass
        
        # Remove any rows with all NaN values
        processed_data = processed_data.dropna(how='all')
        
        return processed_data
    
    def apply_transform(self, data: pd.Series, transform_type: str) -> pd.Series:
        """
        Apply transformation to a single data series
        
        Args:
            data: Input data series
            transform_type: Type of transformation ('Log10', 'Asinh', 'Biexponential', 'none')
            
        Returns:
            Transformed data series
        """
        if transform_type == 'none' or transform_type is None:
            return data
            
        try:
            if transform_type == 'Log10':
                # Log10 transformation (avoid log of negative values)
                positive_data = np.maximum(data, 1)
                return np.log10(positive_data)
                
            elif transform_type == 'Asinh':
                # Inverse hyperbolic sine transformation
                return np.arcsinh(data / 150)
                
            elif transform_type == 'Biexponential':
                # Biexponential transformation (simplified version)
                result = data.copy()
                pos_mask = data > 0
                neg_mask = data <= 0
                
                result[pos_mask] = np.log10(data[pos_mask])
                result[neg_mask] = -np.log10(-data[neg_mask] + 1)
                
                return result
            else:
                st.warning(f"未知の変換タイプ: {transform_type}")
                return data
                
        except Exception as e:
            st.error(f"データ変換エラー ({transform_type}): {str(e)}")
            return data
    
    def apply_transformation(self, data: pd.DataFrame, 
                           transformation: str, 
                           channels: list = None) -> pd.DataFrame:
        """
        Apply transformation to multiple channels (backward compatibility)
        
        Args:
            data: Input dataframe
            transformation: Type of transformation ('log', 'asinh', 'biexp', 'Log10', 'Asinh', 'Biexponential')
            channels: List of channels to transform, if None transform all
            
        Returns:
            Transformed dataframe
        """
        # Convert old naming to new naming
        transform_map = {
            'log': 'Log10',
            'asinh': 'Asinh', 
            'biexp': 'Biexponential'
        }
        
        if transformation in transform_map:
            transformation = transform_map[transformation]
        
        if channels is None:
            channels = data.columns.tolist()
            
        data_transformed = data.copy()
        
        for channel in channels:
            if channel in data.columns:
                data_transformed[channel] = self.apply_transform(data[channel], transformation)
        
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
    
    def export_data(self, data: pd.DataFrame, format: str = 'csv') -> str:
        """
        Export data to specified format
        
        Args:
            data: DataFrame to export
            format: Export format ('csv')
            
        Returns:
            Exported data as string
        """
        if format == 'csv':
            return data.to_csv(index=False)
        else:
            return data.to_csv(index=False)


def load_and_process_fcs(uploaded_file, transformation='Asinh', max_events=10000):
    """
    Load and process FCS file from uploaded file
    
    Args:
        uploaded_file: Streamlit uploaded file object
        transformation: Transformation to apply ('Log10', 'Asinh', 'Biexponential', 'none')
        max_events: Maximum number of events to keep
        
    Returns:
        Tuple of (processor_instance, processed_data, metadata)
    """
    processor = FCSProcessor()
    
    try:
        # Load FCS file with automatic library selection
        data, metadata = processor.load_fcs_file(uploaded_file)
        
        if data is None:
            return None, None, None
        
        # Preprocess data
        data = processor.preprocess_data(data, metadata)
        
        # Apply transformation if requested
        if transformation != 'none':
            data = processor.apply_transformation(data, transformation)
        
        # Subsample if necessary
        if max_events and len(data) > max_events:
            data = processor.subsample_data(data, max_events)
        
        return processor, data, metadata
        
    except Exception as e:
        st.error(f"FCSファイルの処理中にエラーが発生しました: {str(e)}")
        st.error(f"使用されたライブラリ: {processor.used_library}")
        return None, None, None
