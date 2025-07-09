# utils/plotting.py - README.md対応版
# FACS Data Analysis用の包括的なプロッティングユーティリティ

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
        'default_subsample_size': 10000,
        'max_events': 100000,
        'min_events': 1000
    }

class PlottingUtils:
    """
    Main plotting utilities class for FACS data analysis
    Compatible with data from flowio, flowkit, and fcsparser
    Supports individual axis transformations and advanced visualization
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
    
    def _validate_data(self):
        """Validate that data is available and not empty"""
        if self.data is None:
            return False, "No data available for plotting"
        if len(self.data) == 0:
            return False, "Data is empty"
        return True, "Data is valid"
    
    def _validate_channels(self, channels):
        """Validate that channels exist in data"""
        if isinstance(channels, str):
            channels = [channels]
        
        missing_channels = [ch for ch in channels if ch not in self.channels]
        if missing_channels:
            return False, f"Channels not found: {missing_channels}. Available: {self.channels[:10]}..."
        return True, "Channels are valid"
    
    def _subsample_data(self, n_events=None):
        """Subsample data for better plotting performance"""
        is_valid, msg = self._validate_data()
        if not is_valid:
            return None
        
        if n_events is None:
            n_events = DATA_CONFIG['default_subsample_size']
        
        # Ensure n_events is within bounds
        n_events = max(DATA_CONFIG['min_events'], min(n_events, DATA_CONFIG['max_events']))
        
        if len(self.data) > n_events:
            return self.data.sample(n=n_events, random_state=42)
        return self.data
    
    def _apply_transform(self, data, transform='linear'):
        """Apply data transformation with enhanced error handling"""
        try:
            if transform == 'linear' or transform is None or transform == 'なし':
                return data
            elif transform == 'log' or transform == 'log10' or transform == 'Log10':
                # Add small value to avoid log(0) and handle negative values
                return np.log10(np.maximum(data, 1))
            elif transform == 'asinh' or transform == 'Asinh':
                # Standard cofactor for flow cytometry
                return np.arcsinh(data / 150)
            elif transform == 'biexponential' or transform == 'Biexponential':
                # Enhanced biexponential transformation
                # Using parametric approach for better flow cytometry compatibility
                pos_mask = data > 0
                result = np.zeros_like(data)
                result[pos_mask] = np.arcsinh(data[pos_mask] / 150)
                result[~pos_mask] = -np.arcsinh(np.abs(data[~pos_mask]) / 150)
                return result
            else:
                st.warning(f"Unknown transformation: {transform}. Using linear.")
                return data
        except Exception as e:
            st.warning(f"Error applying transformation {transform}: {str(e)}. Using original data.")
            return data
    
    def _create_empty_figure(self, title="No data available"):
        """Create an empty figure when data is not available"""
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for plotting",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(
            title=title,
            width=PLOT_CONFIG['figure_size'][0],
            height=PLOT_CONFIG['figure_size'][1],
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig
    
    def create_histogram(self, channel, bins=None, title=None, transform='linear'):
        """Create histogram for a channel - always returns a figure"""
        # Validate data
        is_valid, msg = self._validate_data()
        if not is_valid:
            st.error(msg)
            return self._create_empty_figure(f"Histogram: {channel}")
        
        # Validate channel
        is_valid, msg = self._validate_channels([channel])
        if not is_valid:
            st.error(msg)
            return self._create_empty_figure(f"Histogram: {channel}")
        
        data_subset = self._subsample_data()
        if data_subset is None:
            return self._create_empty_figure(f"Histogram: {channel}")
        
        if bins is None:
            bins = PLOT_CONFIG['default_bins']
        
        try:
            # Apply transformation
            plot_data = self._apply_transform(data_subset[channel], transform)
            
            # Create histogram with enhanced styling
            fig = px.histogram(
                x=plot_data,
                nbins=bins,
                title=title or f"Histogram: {channel}",
                labels={'x': channel, 'y': 'Count'},
                opacity=0.8
            )
            
            # Add statistics annotation
            mean_val = np.mean(plot_data)
            median_val = np.median(plot_data)
            fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                         annotation_text=f"Mean: {mean_val:.2f}")
            fig.add_vline(x=median_val, line_dash="dash", line_color="blue", 
                         annotation_text=f"Median: {median_val:.2f}")
            
            fig.update_layout(
                width=PLOT_CONFIG['figure_size'][0],
                height=PLOT_CONFIG['figure_size'][1],
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating histogram: {str(e)}")
            return self._create_empty_figure(f"Histogram: {channel} (Error)")
    
    def create_scatter_plot(self, x_channel, y_channel, title=None, 
                          alpha=None, n_points=None, x_transform='linear', y_transform='linear'):
        """Create scatter plot for two channels with individual axis transformations"""
        # Validate data
        is_valid, msg = self._validate_data()
        if not is_valid:
            st.error(msg)
            return self._create_empty_figure(f"{x_channel} vs {y_channel}")
        
        # Validate channels
        is_valid, msg = self._validate_channels([x_channel, y_channel])
        if not is_valid:
            st.error(msg)
            return self._create_empty_figure(f"{x_channel} vs {y_channel}")
        
        # Use provided parameters or defaults
        if alpha is None:
            alpha = PLOT_CONFIG['default_alpha']
        if n_points is not None:
            data_subset = self.data.sample(n=min(n_points, len(self.data)), random_state=42)
        else:
            data_subset = self._subsample_data()
        
        if data_subset is None:
            return self._create_empty_figure(f"{x_channel} vs {y_channel}")
        
        try:
            # Apply individual transformations
            x_data = self._apply_transform(data_subset[x_channel], x_transform)
            y_data = self._apply_transform(data_subset[y_channel], y_transform)
            
            fig = px.scatter(
                x=x_data, y=y_data,
                title=title or f"{x_channel} vs {y_channel}",
                labels={'x': f"{x_channel} ({x_transform})", 'y': f"{y_channel} ({y_transform})"},
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
            return self._create_empty_figure(f"{x_channel} vs {y_channel} (Error)")
    
    def create_density_plot(self, x_channel, y_channel, bins=50, title=None, 
                           x_transform='linear', y_transform='linear'):
        """Create 2D density plot with individual axis transformations"""
        # Validate data
        is_valid, msg = self._validate_data()
        if not is_valid:
            st.error(msg)
            return self._create_empty_figure(f"Density: {x_channel} vs {y_channel}")
        
        # Validate channels
        is_valid, msg = self._validate_channels([x_channel, y_channel])
        if not is_valid:
            st.error(msg)
            return self._create_empty_figure(f"Density: {x_channel} vs {y_channel}")
        
        data_subset = self._subsample_data()
        if data_subset is None:
            return self._create_empty_figure(f"Density: {x_channel} vs {y_channel}")
        
        try:
            # Apply individual transformations
            x_data = self._apply_transform(data_subset[x_channel], x_transform)
            y_data = self._apply_transform(data_subset[y_channel], y_transform)
            
            # Create 2D histogram
            hist, x_edges, y_edges = np.histogram2d(x_data, y_data, bins=bins)
            
            # Smooth the density for better visualization
            hist_smooth = gaussian_filter(hist, sigma=1)
            
            # Create enhanced contour plot
            fig = go.Figure(data=go.Contour(
                z=hist_smooth.T,
                x=x_edges[:-1],
                y=y_edges[:-1],
                colorscale=PLOT_CONFIG['default_colormap'],
                contours=dict(
                    showlabels=True,
                    labelfont=dict(size=10, color='white')
                ),
                showscale=True
            ))
            
            fig.update_layout(
                title=title or f"Density Plot: {x_channel} vs {y_channel}",
                xaxis_title=f"{x_channel} ({x_transform})",
                yaxis_title=f"{y_channel} ({y_transform})",
                width=PLOT_CONFIG['figure_size'][0],
                height=PLOT_CONFIG['figure_size'][1]
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating density plot: {str(e)}")
            return self._create_empty_figure(f"Density: {x_channel} vs {y_channel} (Error)")
    
    def create_multi_histogram(self, channels, bins=None, transform='linear'):
        """Create multiple histograms in subplots"""
        # Validate data
        is_valid, msg = self._validate_data()
        if not is_valid:
            st.error(msg)
            return self._create_empty_figure("Multiple Histograms")
        
        # Validate channels
        is_valid, msg = self._validate_channels(channels)
        if not is_valid:
            st.error(msg)
            return self._create_empty_figure("Multiple Histograms")
        
        data_subset = self._subsample_data()
        if data_subset is None:
            return self._create_empty_figure("Multiple Histograms")
        
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
                title_text="Multiple Channel Histograms",
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating multi-histogram: {str(e)}")
            return self._create_empty_figure("Multiple Histograms (Error)")
    
    def create_correlation_heatmap(self, channels=None, method='pearson'):
        """Create correlation heatmap for selected channels"""
        # Validate data
        is_valid, msg = self._validate_data()
        if not is_valid:
            st.error(msg)
            return self._create_empty_figure("Correlation Heatmap")
        
        if channels is None:
            # Use numeric channels only
            numeric_channels = self.data.select_dtypes(include=[np.number]).columns
            channels = list(numeric_channels)[:10]  # Limit to 10 channels for readability
        
        # Validate channels
        is_valid, msg = self._validate_channels(channels)
        if not is_valid:
            st.error(msg)
            return self._create_empty_figure("Correlation Heatmap")
        
        try:
            data_subset = self._subsample_data()
            if data_subset is None:
                return self._create_empty_figure("Correlation Heatmap")
            
            corr_data = data_subset[channels].corr(method=method)
            
            fig = px.imshow(
                corr_data,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title=f"Channel Correlation Heatmap ({method.title()})",
                color_continuous_midpoint=0
            )
            
            fig.update_layout(
                width=PLOT_CONFIG['figure_size'][0],
                height=PLOT_CONFIG['figure_size'][1]
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating correlation heatmap: {str(e)}")
            return self._create_empty_figure("Correlation Heatmap (Error)")
    
    def get_channel_statistics(self, channel, gate_data=None, transform='linear'):
        """Get basic statistics for a channel with optional transformation"""
        is_valid, msg = self._validate_data()
        if not is_valid:
            return None
        
        is_valid, msg = self._validate_channels([channel])
        if not is_valid:
            return None
        
        try:
            data_to_analyze = gate_data if gate_data is not None else self.data
            channel_data = self._apply_transform(data_to_analyze[channel], transform)
            
            stats = {
                'channel': channel,
                'transform': transform,
                'count': len(channel_data),
                'mean': float(np.mean(channel_data)),
                'median': float(np.median(channel_data)),
                'std': float(np.std(channel_data)),
                'min': float(np.min(channel_data)),
                'max': float(np.max(channel_data)),
                'q25': float(np.percentile(channel_data, 25)),
                'q75': float(np.percentile(channel_data, 75)),
                'cv': float(np.std(channel_data) / np.mean(channel_data) * 100) if np.mean(channel_data) != 0 else 0
            }
            
            return stats
            
        except Exception as e:
            st.error(f"Error calculating statistics for {channel}: {str(e)}")
            return None
    
    def create_statistics_table(self, channels=None, gate_data=None, transform='linear'):
        """Create a comprehensive statistics table for channels"""
        is_valid, msg = self._validate_data()
        if not is_valid:
            return None
        
        if channels is None:
            channels = self.channels[:10]  # Limit to first 10 channels
        
        stats_data = []
        for channel in channels:
            if channel in self.channels:
                stats = self.get_channel_statistics(channel, gate_data, transform)
                if stats:
                    stats_data.append(stats)
        
        if stats_data:
            df = pd.DataFrame(stats_data)
            # Round numeric columns for better display
            numeric_columns = ['mean', 'median', 'std', 'min', 'max', 'q25', 'q75', 'cv']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = df[col].round(3)
            return df
        return None

# README.md準拠のモジュールレベル関数（後方互換性）
def create_histogram(data, channel, bins=50, title=None, transform='linear', metadata=None):
    """Create histogram - README.md準拠のシグネチャ"""
    plotter = PlottingUtils(data, metadata)
    return plotter.create_histogram(channel, bins, title, transform)

def create_scatter_plot(data, x_channel, y_channel, title=None, alpha=0.6, 
                       n_points=None, transform='linear', metadata=None):
    """Create scatter plot - README.md準拠のシグネチャ"""
    plotter = PlottingUtils(data, metadata)
    return plotter.create_scatter_plot(x_channel, y_channel, title, alpha, 
                                      n_points, transform, transform)

def create_density_plot(data, x_channel, y_channel, bins=50, title=None, 
                       transform='linear', metadata=None):
    """Create density plot - README.md準拠のシグネチャ"""
    plotter = PlottingUtils(data, metadata)
    return plotter.create_density_plot(x_channel, y_channel, bins, title, 
                                      transform, transform)

def create_multi_histogram(data, channels, bins=None, transform='linear', metadata=None):
    """Create multiple histograms"""
    plotter = PlottingUtils(data, metadata)
    return plotter.create_multi_histogram(channels, bins, transform)

def create_correlation_heatmap(data, channels=None, method='pearson', metadata=None):
    """Create correlation heatmap"""
    plotter = PlottingUtils(data, metadata)
    return plotter.create_correlation_heatmap(channels, method)

# 高度な機能のための新しい関数
def create_interactive_scatter(data, x_channel, y_channel, width=800, height=600, 
                              x_transform='linear', y_transform='linear', alpha=0.6):
    """Create interactive scatter plot with enhanced features"""
    if data is None or len(data) == 0:
        st.error("No data available for plotting")
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for plotting",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(
            title=f"{x_channel} vs {y_channel}",
            width=width, height=height,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig
    
    # Check if channels exist
    if x_channel not in data.columns or y_channel not in data.columns:
        st.error(f"Channels not found. Available: {list(data.columns)[:10]}...")
        fig = go.Figure()
        fig.add_annotation(
            text="Channels not found",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(
            title=f"{x_channel} vs {y_channel}",
            width=width, height=height,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig
    
    # Subsample data if too large
    if len(data) > DATA_CONFIG['default_subsample_size']:
        plot_data = data.sample(n=DATA_CONFIG['default_subsample_size'], random_state=42)
    else:
        plot_data = data
    
    try:
        plotter = PlottingUtils(data)
        x_data = plotter._apply_transform(plot_data[x_channel], x_transform)
        y_data = plotter._apply_transform(plot_data[y_channel], y_transform)
        
        fig = px.scatter(
            x=x_data, y=y_data,
            title=f"{x_channel} vs {y_channel}",
            labels={'x': f"{x_channel} ({x_transform})", 'y': f"{y_channel} ({y_transform})"},
            width=width,
            height=height,
            opacity=alpha
        )
        
        fig.update_traces(marker=dict(size=PLOT_CONFIG['scatter_size']))
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating interactive scatter plot: {str(e)}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(
            title=f"{x_channel} vs {y_channel} (Error)",
            width=width, height=height,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig

def create_enhanced_histogram(data, channel, bins=50, transform='linear', show_stats=True, metadata=None):
    """Create enhanced histogram with statistics overlay"""
    plotter = PlottingUtils(data, metadata)
    fig = plotter.create_histogram(channel, bins, None, transform)
    
    if show_stats and data is not None and channel in data.columns:
        try:
            stats = plotter.get_channel_statistics(channel, transform=transform)
            if stats:
                # Add statistics text box
                stats_text = f"Mean: {stats['mean']:.2f}<br>Median: {stats['median']:.2f}<br>Std: {stats['std']:.2f}<br>CV: {stats['cv']:.1f}%"
                fig.add_annotation(
                    text=stats_text,
                    xref="paper", yref="paper",
                    x=0.02, y=0.98, xanchor='left', yanchor='top',
                    showarrow=False,
                    font=dict(size=12),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="black",
                    borderwidth=1
                )
        except Exception as e:
            st.warning(f"Could not add statistics overlay: {str(e)}")
    
    return fig

# Legacy class alias for backward compatibility
class FCSPlotter(PlottingUtils):
    """Legacy alias for PlottingUtils - for backward compatibility"""
    pass
