# utils/plotting.py - Plotting utilities for FACS Data Analysis
# Fixed version addressing README-code inconsistencies

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter

# Default configuration if config.py is not available
try:
    from config import PLOT_CONFIG, DATA_CONFIG
except ImportError:
    PLOT_CONFIG = {
        'figure_size': (800, 600),
        'default_bins': 50,
        'default_alpha': 0.6,
        'scatter_size': 3,
        'default_colormap': 'viridis'
    }
    DATA_CONFIG = {
        'default_subsample_size': 10000
    }

class PlottingUtils:
    """
    Main plotting utilities class for FACS data analysis
    Compatible with data from flowio, flowkit, and fcsparser
    """
    
    def __init__(self, data=None, metadata=None):
        """
        Initialize plotter with FCS data
        
        Parameters:
        - data: pandas DataFrame from any FCS reader (flowio, flowkit, fcsparser)
        - metadata: dict with FCS metadata
        """
        self.data = data
        self.metadata = metadata or {}
        self.channels = list(data.columns) if data is not None else []
    
    def _subsample_data(self, n_events=None):
        """Subsample data for better plotting performance"""
        if self.data is None or len(self.data) == 0:
            return self.data
        
        if n_events is None:
            n_events = DATA_CONFIG['default_subsample_size']
        
        if len(self.data) > n_events:
            return self.data.sample(n=n_events, random_state=42)
        return self.data
    
    def _apply_transform(self, data, transform='linear'):
        """Apply data transformation"""
        if transform == 'linear' or transform is None:
            return data
        elif transform == 'log' or transform == 'log10':
            # Add small value to avoid log(0)
            return np.log10(np.maximum(data, 1))
        elif transform == 'asinh':
            return np.arcsinh(data / 150)  # Standard cofactor for flow cytometry
        elif transform == 'biexponential':
            # Simplified biexponential (using asinh approximation)
            return np.arcsinh(data / 150)
        else:
            st.warning(f"Unknown transformation: {transform}. Using linear.")
            return data
    
    def create_histogram(self, channel, bins=None, title=None, transform='linear'):
        """Create histogram for a channel"""
        if self.data is None:
            st.error("No data available for plotting")
            return None
            
        if channel not in self.channels:
            st.error(f"Channel {channel} not found in data. Available channels: {self.channels}")
            return None
        
        data_subset = self._subsample_data()
        
        if bins is None:
            bins = PLOT_CONFIG['default_bins']
        
        try:
            # Apply transformation
            plot_data = self._apply_transform(data_subset[channel], transform)
            
            fig = px.histogram(
                x=plot_data,
                nbins=bins,
                title=title or f"Histogram: {channel}",
                labels={'x': channel, 'y': 'Count'}
            )
            
            fig.update_layout(
                width=PLOT_CONFIG['figure_size'][0],
                height=PLOT_CONFIG['figure_size'][1],
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating histogram: {str(e)}")
            return None
    
    def create_scatter_plot(self, x_channel, y_channel, title=None, 
                          alpha=None, n_points=None, x_transform='linear', y_transform='linear'):
        """
        Create scatter plot for two channels
        
        Parameters:
        - x_channel: X-axis channel name
        - y_channel: Y-axis channel name  
        - title: Plot title
        - alpha: Point transparency (0.0-1.0)
        - n_points: Number of points to display
        - x_transform: Transformation for X-axis data
        - y_transform: Transformation for Y-axis data
        """
        if self.data is None:
            st.error("No data available for plotting")
            return None
            
        if x_channel not in self.channels or y_channel not in self.channels:
            st.error(f"One or both channels not found. Available: {self.channels}")
            return None
        
        # Use provided parameters or defaults
        if alpha is None:
            alpha = PLOT_CONFIG['default_alpha']
        if n_points is not None:
            data_subset = self.data.sample(n=min(n_points, len(self.data)), random_state=42)
        else:
            data_subset = self._subsample_data()
        
        try:
            # Apply transformations
            x_data = self._apply_transform(data_subset[x_channel], x_transform)
            y_data = self._apply_transform(data_subset[y_channel], y_transform)
            
            fig = px.scatter(
                x=x_data, y=y_data,
                title=title or f"{x_channel} vs {y_channel}",
                labels={'x': x_channel, 'y': y_channel},
                opacity=alpha
            )
            
            fig.update_traces(marker=dict(size=PLOT_CONFIG['scatter_size']))
            fig.update_layout(
                width=PLOT_CONFIG['figure_size'][0],
                height=PLOT_CONFIG['figure_size'][1]
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating scatter plot: {str(e)}")
            return None
    
    def create_density_plot(self, x_channel, y_channel, bins=50, title=None, 
                           x_transform='linear', y_transform='linear'):
        """
        Create 2D density plot (contour plot)
        
        Parameters:
        - x_channel: X-axis channel name
        - y_channel: Y-axis channel name
        - bins: Number of bins for density calculation
        - title: Plot title
        - x_transform: Transformation for X-axis data
        - y_transform: Transformation for Y-axis data
        """
        if self.data is None:
            st.error("No data available for plotting")
            return None
            
        if x_channel not in self.channels or y_channel not in self.channels:
            st.error("One or both channels not found in data")
            return None
        
        data_subset = self._subsample_data()
        
        try:
            # Apply transformations
            x_data = self._apply_transform(data_subset[x_channel], x_transform)
            y_data = self._apply_transform(data_subset[y_channel], y_transform)
            
            # Create 2D histogram
            hist, x_edges, y_edges = np.histogram2d(x_data, y_data, bins=bins)
            
            # Smooth the density
            hist_smooth = gaussian_filter(hist, sigma=1)
            
            # Create contour plot
            fig = go.Figure(data=go.Contour(
                z=hist_smooth.T,
                x=x_edges[:-1],
                y=y_edges[:-1],
                colorscale=PLOT_CONFIG['default_colormap'],
                contours=dict(
                    showlabels=True,
                    labelfont=dict(size=12, color='white')
                )
            ))
            
            fig.update_layout(
                title=title or f"Density Plot: {x_channel} vs {y_channel}",
                xaxis_title=x_channel,
                yaxis_title=y_channel,
                width=PLOT_CONFIG['figure_size'][0],
                height=PLOT_CONFIG['figure_size'][1]
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating density plot: {str(e)}")
            return None
    
    def create_multi_histogram(self, channels, bins=None, transform='linear'):
        """Create multiple histograms in subplots"""
        if self.data is None:
            st.error("No data available for plotting")
            return None
            
        if not channels or not all(ch in self.channels for ch in channels):
            st.error("Some channels not found in data")
            return None
        
        data_subset = self._subsample_data()
        
        if bins is None:
            bins = PLOT_CONFIG['default_bins']
        
        try:
            n_channels = len(channels)
            cols = min(3, n_channels)
            rows = (n_channels - 1) // cols + 1
            
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=channels,
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            for i, channel in enumerate(channels):
                row = i // cols + 1
                col = i % cols + 1
                
                plot_data = self._apply_transform(data_subset[channel], transform)
                
                fig.add_trace(
                    go.Histogram(x=plot_data, nbinsx=bins, name=channel, showlegend=False),
                    row=row, col=col
                )
            
            fig.update_layout(
                height=300 * rows,
                title_text="Multiple Channel Histograms"
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating multi-histogram: {str(e)}")
            return None
    
    def create_correlation_heatmap(self, channels=None, method='pearson'):
        """Create correlation heatmap"""
        if self.data is None:
            st.error("No data available for plotting")
            return None
            
        if channels is None:
            # Use numeric channels only
            numeric_channels = self.data.select_dtypes(include=[np.number]).columns
            channels = list(numeric_channels)[:10]  # Limit to 10 channels for readability
        
        if not channels or not all(ch in self.channels for ch in channels):
            st.error("Some channels not found in data")
            return None
        
        try:
            data_subset = self._subsample_data()
            corr_data = data_subset[channels].corr(method=method)
            
            fig = px.imshow(
                corr_data,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title=f"Channel Correlation Heatmap ({method.title()})"
            )
            
            fig.update_layout(
                width=PLOT_CONFIG['figure_size'][0],
                height=PLOT_CONFIG['figure_size'][1]
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating correlation heatmap: {str(e)}")
            return None
    
    def get_channel_statistics(self, channel, gate_data=None):
        """Get basic statistics for a channel"""
        if self.data is None:
            return None
            
        if channel not in self.channels:
            return None
        
        try:
            data_to_analyze = gate_data if gate_data is not None else self.data
            channel_data = data_to_analyze[channel]
            
            stats = {
                'channel': channel,
                'count': len(channel_data),
                'mean': float(np.mean(channel_data)),
                'median': float(np.median(channel_data)),
                'std': float(np.std(channel_data)),
                'min': float(np.min(channel_data)),
                'max': float(np.max(channel_data)),
                'q25': float(np.percentile(channel_data, 25)),
                'q75': float(np.percentile(channel_data, 75))
            }
            
            return stats
            
        except Exception as e:
            st.error(f"Error calculating statistics for {channel}: {str(e)}")
            return None
    
    def create_statistics_table(self, channels=None, gate_data=None):
        """Create a statistics table for channels"""
        if self.data is None:
            return None
            
        if channels is None:
            channels = self.channels[:10]  # Limit to first 10 channels
        
        stats_data = []
        for channel in channels:
            if channel in self.channels:
                stats = self.get_channel_statistics(channel, gate_data)
                if stats:
                    stats_data.append(stats)
        
        if stats_data:
            return pd.DataFrame(stats_data)
        return None

# Backward compatibility: module-level functions that use PlottingUtils
def create_histogram(data, channel, bins=None, title=None, transform='linear', metadata=None):
    """
    Create histogram for a channel (module-level function for backward compatibility)
    
    Parameters:
    - data: pandas DataFrame
    - channel: channel name
    - bins: number of bins
    - title: plot title
    - transform: data transformation
    - metadata: FCS metadata (optional)
    """
    plotter = PlottingUtils(data, metadata)
    return plotter.create_histogram(channel, bins, title, transform)

def create_scatter_plot(data, x_channel, y_channel, title=None, alpha=None, 
                       transform='linear', metadata=None):
    """
    Create scatter plot for two channels (module-level function for backward compatibility)
    
    Parameters:
    - data: pandas DataFrame
    - x_channel: x-axis channel name
    - y_channel: y-axis channel name
    - title: plot title
    - alpha: point transparency
    - transform: data transformation (applied to both axes)
    - metadata: FCS metadata (optional)
    """
    plotter = PlottingUtils(data, metadata)
    return plotter.create_scatter_plot(x_channel, y_channel, title, alpha, 
                                      None, transform, transform)

def create_density_plot(data, x_channel, y_channel, bins=50, title=None, 
                       transform='linear', metadata=None):
    """
    Create 2D density plot (module-level function for backward compatibility)
    
    Parameters:
    - data: pandas DataFrame
    - x_channel: x-axis channel name
    - y_channel: y-axis channel name
    - bins: number of bins
    - title: plot title
    - transform: data transformation (applied to both axes)
    - metadata: FCS metadata (optional)
    """
    plotter = PlottingUtils(data, metadata)
    return plotter.create_density_plot(x_channel, y_channel, bins, title, 
                                      transform, transform)

def create_multi_histogram(data, channels, bins=None, transform='linear', metadata=None):
    """
    Create multiple histograms (module-level function for backward compatibility)
    
    Parameters:
    - data: pandas DataFrame
    - channels: list of channel names
    - bins: number of bins
    - transform: data transformation
    - metadata: FCS metadata (optional)
    """
    plotter = PlottingUtils(data, metadata)
    return plotter.create_multi_histogram(channels, bins, transform)

def create_correlation_heatmap(data, channels=None, method='pearson', metadata=None):
    """
    Create correlation heatmap (module-level function for backward compatibility)
    
    Parameters:
    - data: pandas DataFrame
    - channels: list of channel names
    - method: correlation method
    - metadata: FCS metadata (optional)
    """
    plotter = PlottingUtils(data, metadata)
    return plotter.create_correlation_heatmap(channels, method)

# Additional utility functions
def create_interactive_scatter(data, x_channel, y_channel, width=800, height=600):
    """Create interactive scatter plot using Plotly (simplified version)"""
    if data is None or len(data) == 0:
        st.error("No data available for plotting")
        return None
    
    # Subsample data if too large
    if len(data) > DATA_CONFIG['default_subsample_size']:
        plot_data = data.sample(n=DATA_CONFIG['default_subsample_size'], random_state=42)
    else:
        plot_data = data
    
    try:
        fig = px.scatter(
            plot_data, 
            x=x_channel, 
            y=y_channel,
            title=f"{x_channel} vs {y_channel}",
            width=width,
            height=height,
            opacity=PLOT_CONFIG['default_alpha']
        )
        
        fig.update_traces(marker=dict(size=PLOT_CONFIG['scatter_size']))
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating interactive scatter plot: {str(e)}")
        return None

# Legacy class alias for backward compatibility
class FCSPlotter(PlottingUtils):
    """Legacy alias for PlottingUtils - for backward compatibility"""
    pass
