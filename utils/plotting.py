# utils/plotting.py - Plotting utilities for FACS Data Analysis
# Updated for fcsparser migration from flowkit

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import bokeh.plotting as bp
from bokeh.models import HoverTool, ColorBar, LinearColorMapper
from bokeh.palettes import Viridis256
from bokeh.transform import transform
from bokeh.layouts import column, row
import altair as alt
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
import streamlit as st
from config import PLOT_CONFIG, DATA_CONFIG

class FCSPlotter:
    """Plotting utilities for FCS data using fcsparser"""
    
    def __init__(self, data, metadata=None):
        """
        Initialize plotter with FCS data
        
        Parameters:
        - data: pandas DataFrame from fcsparser
        - metadata: dict with FCS metadata from fcsparser
        """
        self.data = data
        self.metadata = metadata or {}
        self.channels = list(data.columns) if data is not None else []
    
    def subsample_data(self, n_events=None):
        """Subsample data for better plotting performance"""
        if self.data is None or len(self.data) == 0:
            return self.data
        
        if n_events is None:
            n_events = DATA_CONFIG['default_subsample_size']
        
        if len(self.data) > n_events:
            return self.data.sample(n=n_events, random_state=42)
        return self.data
    
    def create_histogram(self, channel, bins=None, title=None, transform='linear'):
        """Create histogram for a channel"""
        if channel not in self.channels:
            st.error(f"Channel {channel} not found in data")
            return None
        
        data_subset = self.subsample_data()
        
        if bins is None:
            bins = PLOT_CONFIG['default_bins']
        
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
    
    def create_scatter_plot(self, x_channel, y_channel, title=None, 
                          color_channel=None, transform='linear'):
        """Create scatter plot for two channels"""
        if x_channel not in self.channels or y_channel not in self.channels:
            st.error(f"One or both channels not found in data")
            return None
        
        data_subset = self.subsample_data()
        
        # Apply transformations
        x_data = self._apply_transform(data_subset[x_channel], transform)
        y_data = self._apply_transform(data_subset[y_channel], transform)
        
        if color_channel and color_channel in self.channels:
            color_data = self._apply_transform(data_subset[color_channel], transform)
            fig = px.scatter(
                x=x_data, y=y_data, color=color_data,
                title=title or f"{x_channel} vs {y_channel}",
                labels={'x': x_channel, 'y': y_channel, 'color': color_channel},
                opacity=PLOT_CONFIG['default_alpha']
            )
        else:
            fig = px.scatter(
                x=x_data, y=y_data,
                title=title or f"{x_channel} vs {y_channel}",
                labels={'x': x_channel, 'y': y_channel},
                opacity=PLOT_CONFIG['default_alpha']
            )
        
        fig.update_traces(marker=dict(size=PLOT_CONFIG['scatter_size']))
        fig.update_layout(
            width=PLOT_CONFIG['figure_size'][0],
            height=PLOT_CONFIG['figure_size'][1]
        )
        
        return fig
    
    def create_density_plot(self, x_channel, y_channel, bins=50, title=None, 
                           transform='linear'):
        """Create 2D density plot (contour plot)"""
        if x_channel not in self.channels or y_channel not in self.channels:
            st.error("One or both channels not found in data")
            return None
        
        data_subset = self.subsample_data()
        
        # Apply transformations
        x_data = self._apply_transform(data_subset[x_channel], transform)
        y_data = self._apply_transform(data_subset[y_channel], transform)
        
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
    
    def create_hexbin_plot(self, x_channel, y_channel, gridsize=50, title=None, 
                          transform='linear'):
        """Create hexagonal binning plot using Altair"""
        if x_channel not in self.channels or y_channel not in self.channels:
            st.error("One or both channels not found in data")
            return None
        
        data_subset = self.subsample_data()
        
        # Apply transformations
        plot_data = data_subset.copy()
        plot_data[x_channel] = self._apply_transform(data_subset[x_channel], transform)
        plot_data[y_channel] = self._apply_transform(data_subset[y_channel], transform)
        
        chart = alt.Chart(plot_data).mark_circle(size=20, opacity=0.6).encode(
            x=alt.X(f'{x_channel}:Q', title=x_channel),
            y=alt.Y(f'{y_channel}:Q', title=y_channel),
            color=alt.Color('count():Q', scale=alt.Scale(scheme='viridis'))
        ).resolve_scale(
            color='independent'
        ).properties(
            width=PLOT_CONFIG['figure_size'][0]-100,
            height=PLOT_CONFIG['figure_size'][1]-100,
            title=title or f"Hexbin Plot: {x_channel} vs {y_channel}"
        )
        
        return chart
    
    def create_multi_histogram(self, channels, bins=None, transform='linear'):
        """Create multiple histograms in subplots"""
        if not channels or not all(ch in self.channels for ch in channels):
            st.error("Some channels not found in data")
            return None
        
        data_subset = self.subsample_data()
        
        if bins is None:
            bins = PLOT_CONFIG['default_bins']
        
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
    
    def create_correlation_heatmap(self, channels=None, method='pearson'):
        """Create correlation heatmap"""
        if channels is None:
            # Use numeric channels only
            numeric_channels = self.data.select_dtypes(include=[np.number]).columns
            channels = list(numeric_channels)[:10]  # Limit to 10 channels for readability
        
        if not channels or not all(ch in self.channels for ch in channels):
            st.error("Some channels not found in data")
            return None
        
        data_subset = self.subsample_data()
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
    
    def _apply_transform(self, data, transform='linear'):
        """Apply data transformation"""
        if transform == 'linear':
            return data
        elif transform == 'log':
            # Add small value to avoid log(0)
            return np.log10(data + 1)
        elif transform == 'asinh':
            return np.arcsinh(data / 150)  # Standard cofactor for flow cytometry
        elif transform == 'biexponential':
            # Simplified biexponential (just asinh for now)
            return np.arcsinh(data / 150)
        else:
            return data
    
    def get_channel_statistics(self, channel, gate_data=None):
        """Get basic statistics for a channel"""
        if channel not in self.channels:
            return None
        
        data_to_analyze = gate_data if gate_data is not None else self.data
        channel_data = data_to_analyze[channel]
        
        stats = {
            'count': len(channel_data),
            'mean': np.mean(channel_data),
            'median': np.median(channel_data),
            'std': np.std(channel_data),
            'min': np.min(channel_data),
            'max': np.max(channel_data),
            'q25': np.percentile(channel_data, 25),
            'q75': np.percentile(channel_data, 75)
        }
        
        return stats
    
    def create_statistics_table(self, channels=None, gate_data=None):
        """Create a statistics table for channels"""
        if channels is None:
            channels = self.channels[:10]  # Limit to first 10 channels
        
        stats_data = []
        for channel in channels:
            if channel in self.channels:
                stats = self.get_channel_statistics(channel, gate_data)
                if stats:
                    stats['channel'] = channel
                    stats_data.append(stats)
        
        if stats_data:
            return pd.DataFrame(stats_data)
        return None

def create_interactive_scatter(data, x_channel, y_channel, width=800, height=600):
    """Create interactive scatter plot using Bokeh"""
    from bokeh.plotting import figure
    from bokeh.models import HoverTool
    
    # Subsample data if too large
    if len(data) > DATA_CONFIG['default_subsample_size']:
        plot_data = data.sample(n=DATA_CONFIG['default_subsample_size'], random_state=42)
    else:
        plot_data = data
    
    p = figure(
        width=width, 
        height=height,
        title=f"{x_channel} vs {y_channel}",
        x_axis_label=x_channel,
        y_axis_label=y_channel,
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )
    
    # Add hover tool
    hover = HoverTool(tooltips=[
        (x_channel, f"@{{{x_channel}}}"),
        (y_channel, f"@{{{y_channel}}}")
    ])
    p.add_tools(hover)
    
    # Create scatter plot
    p.circle(
        x=x_channel, 
        y=y_channel, 
        source=plot_data,
        size=PLOT_CONFIG['scatter_size'],
        alpha=PLOT_CONFIG['default_alpha']
    )
    
    return p

# モジュールレベルの便利関数（既存のFCSPlotterクラスを使用）
def create_histogram(data, channel, bins=None, title=None, transform='linear', metadata=None):
    """
    Create histogram for a channel (module-level function)
    
    Parameters:
    - data: pandas DataFrame
    - channel: channel name
    - bins: number of bins
    - title: plot title
    - transform: data transformation
    - metadata: FCS metadata
    """
    plotter = FCSPlotter(data, metadata)
    return plotter.create_histogram(channel, bins, title, transform)

def create_scatter_plot(data, x_channel, y_channel, title=None, color_channel=None, 
                       transform='linear', metadata=None):
    """
    Create scatter plot for two channels (module-level function)
    
    Parameters:
    - data: pandas DataFrame
    - x_channel: x-axis channel name
    - y_channel: y-axis channel name
    - title: plot title
    - color_channel: optional color channel
    - transform: data transformation
    - metadata: FCS metadata
    """
    plotter = FCSPlotter(data, metadata)
    return plotter.create_scatter_plot(x_channel, y_channel, title, color_channel, transform)

def create_density_plot(data, x_channel, y_channel, bins=50, title=None, 
                       transform='linear', metadata=None):
    """
    Create 2D density plot (module-level function)
    
    Parameters:
    - data: pandas DataFrame
    - x_channel: x-axis channel name
    - y_channel: y-axis channel name
    - bins: number of bins
    - title: plot title
    - transform: data transformation
    - metadata: FCS metadata
    """
    plotter = FCSPlotter(data, metadata)
    return plotter.create_density_plot(x_channel, y_channel, bins, title, transform)

def create_multi_histogram(data, channels, bins=None, transform='linear', metadata=None):
    """
    Create multiple histograms (module-level function)
    
    Parameters:
    - data: pandas DataFrame
    - channels: list of channel names
    - bins: number of bins
    - transform: data transformation
    - metadata: FCS metadata
    """
    plotter = FCSPlotter(data, metadata)
    return plotter.create_multi_histogram(channels, bins, transform)

def create_correlation_heatmap(data, channels=None, method='pearson', metadata=None):
    """
    Create correlation heatmap (module-level function)
    
    Parameters:
    - data: pandas DataFrame
    - channels: list of channel names
    - method: correlation method
    - metadata: FCS metadata
    """
    plotter = FCSPlotter(data, metadata)
    return plotter.create_correlation_heatmap(channels, method)

def create_hexbin_plot(data, x_channel, y_channel, gridsize=50, title=None, 
                      transform='linear', metadata=None):
    """
    Create hexagonal binning plot (module-level function)
    
    Parameters:
    - data: pandas DataFrame
    - x_channel: x-axis channel name
    - y_channel: y-axis channel name
    - gridsize: grid size
    - title: plot title
    - transform: data transformation
    - metadata: FCS metadata
    """
    plotter = FCSPlotter(data, metadata)
    return plotter.create_hexbin_plot(x_channel, y_channel, gridsize, title, transform)

class PlottingUtils:
    """Backward compatibility class - wrapper around FCSPlotter"""
    
    def __init__(self, data=None, metadata=None):
        self.plotter = FCSPlotter(data, metadata)
        self.data = data
        self.metadata = metadata
    
    def create_histogram(self, channel, bins=None, title=None, transform='linear'):
        """Create histogram using FCSPlotter"""
        return self.plotter.create_histogram(channel, bins, title, transform)
    
    def create_scatter_plot(self, x_channel, y_channel, title=None, 
                          color_channel=None, transform='linear'):
        """Create scatter plot using FCSPlotter"""
        return self.plotter.create_scatter_plot(x_channel, y_channel, title, 
                                              color_channel, transform)
    
    def create_density_plot(self, x_channel, y_channel, bins=50, title=None, 
                           transform='linear'):
        """Create density plot using FCSPlotter"""
        return self.plotter.create_density_plot(x_channel, y_channel, bins, 
                                               title, transform)
    
    def create_multi_histogram(self, channels, bins=None, transform='linear'):
        """Create multiple histograms using FCSPlotter"""
        return self.plotter.create_multi_histogram(channels, bins, transform)
    
    def create_correlation_heatmap(self, channels=None, method='pearson'):
        """Create correlation heatmap using FCSPlotter"""
        return self.plotter.create_correlation_heatmap(channels, method)
    
    def get_channel_statistics(self, channel, gate_data=None):
        """Get channel statistics using FCSPlotter"""
        return self.plotter.get_channel_statistics(channel, gate_data)
    
    def create_statistics_table(self, channels=None, gate_data=None):
        """Create statistics table using FCSPlotter"""
        return self.plotter.create_statistics_table(channels, gate_data)
