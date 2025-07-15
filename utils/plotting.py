# utils/plotting.py - シンプル化版
# FACS Data Analysis用のプロッティングユーティリティ

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.ndimage import gaussian_filter

# デフォルト設定
DEFAULT_CONFIG = {
    'figure_size': (800, 600),
    'default_bins': 50,
    'default_alpha': 0.6,
    'scatter_size': 3,
    'default_subsample_size': 10000,
    'max_events': 100000,
    'min_events': 1000
}

class PlottingUtils:
    """FACS データ解析用のシンプルなプロッティングユーティリティ"""
    
    def __init__(self, data=None, metadata=None):
        self.data = data
        self.metadata = metadata or {}
        self.channels = list(data.columns) if data is not None else []
    
    def _validate_inputs(self, channels):
        """データとチャンネルの基本的な検証"""
        if self.data is None or len(self.data) == 0:
            st.error("データが利用できません")
            return False
        
        if isinstance(channels, str):
            channels = [channels]
        
        missing = [ch for ch in channels if ch not in self.channels]
        if missing:
            st.error(f"チャンネルが見つかりません: {missing}")
            return False
        
        return True
    
    def _subsample_data(self, n_events=None):
        """データのサブサンプリング"""
        if n_events is None:
            n_events = DEFAULT_CONFIG['default_subsample_size']
        
        if len(self.data) > n_events:
            return self.data.sample(n=n_events, random_state=42)
        return self.data
    
    def _apply_transform(self, data, transform):
        """データ変換の適用"""
        try:
            if transform == 'log10' or transform == 'Log10':
                return np.log10(np.maximum(data, 1))
            elif transform == 'asinh' or transform == 'Asinh':
                return np.arcsinh(data / 150)
            else:  # 'linear', 'なし', None
                return data
        except Exception as e:
            st.warning(f"変換エラー: {e}。元のデータを使用します。")
            return data
    
    def _create_empty_figure(self, title="データがありません"):
        """空のプロットを作成"""
        fig = go.Figure()
        fig.add_annotation(
            text="データがありません",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(
            title=title,
            width=DEFAULT_CONFIG['figure_size'][0],
            height=DEFAULT_CONFIG['figure_size'][1],
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig
    
    def create_histogram(self, channel, bins=None, title=None, transform='linear'):
        """ヒストグラムを作成"""
        if not self._validate_inputs([channel]):
            return self._create_empty_figure(f"ヒストグラム: {channel}")
        
        try:
            data_subset = self._subsample_data()
            plot_data = self._apply_transform(data_subset[channel], transform)
            
            if bins is None:
                bins = DEFAULT_CONFIG['default_bins']
            
            fig = px.histogram(
                x=plot_data,
                nbins=bins,
                title=title or f"ヒストグラム: {channel}",
                labels={'x': channel, 'y': 'カウント'}
            )
            
            # 統計情報を追加
            mean_val = np.mean(plot_data)
            median_val = np.median(plot_data)
            fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                         annotation_text=f"平均: {mean_val:.2f}")
            fig.add_vline(x=median_val, line_dash="dash", line_color="blue", 
                         annotation_text=f"中央値: {median_val:.2f}")
            
            fig.update_layout(
                width=DEFAULT_CONFIG['figure_size'][0],
                height=DEFAULT_CONFIG['figure_size'][1]
            )
            
            return fig
            
        except Exception as e:
            st.error(f"ヒストグラム作成エラー: {e}")
            return self._create_empty_figure(f"ヒストグラム: {channel}")
    
    def create_scatter_plot(self, x_channel, y_channel, title=None, 
                          alpha=None, n_points=None, x_transform='linear', y_transform='linear'):
        """散布図を作成"""
        if not self._validate_inputs([x_channel, y_channel]):
            return self._create_empty_figure(f"{x_channel} vs {y_channel}")
        
        try:
            if alpha is None:
                alpha = DEFAULT_CONFIG['default_alpha']
            
            if n_points is not None:
                data_subset = self.data.sample(n=min(n_points, len(self.data)), random_state=42)
            else:
                data_subset = self._subsample_data()
            
            x_data = self._apply_transform(data_subset[x_channel], x_transform)
            y_data = self._apply_transform(data_subset[y_channel], y_transform)
            
            fig = px.scatter(
                x=x_data, y=y_data,
                title=title or f"{x_channel} vs {y_channel}",
                labels={'x': f"{x_channel}", 'y': f"{y_channel}"},
                opacity=alpha
            )
            
            fig.update_traces(marker=dict(size=DEFAULT_CONFIG['scatter_size']))
            fig.update_layout(
                width=DEFAULT_CONFIG['figure_size'][0],
                height=DEFAULT_CONFIG['figure_size'][1]
            )
            
            return fig
            
        except Exception as e:
            st.error(f"散布図作成エラー: {e}")
            return self._create_empty_figure(f"{x_channel} vs {y_channel}")
    
    def create_density_plot(self, x_channel, y_channel, bins=50, title=None, 
                           x_transform='linear', y_transform='linear'):
        """密度プロットを作成"""
        if not self._validate_inputs([x_channel, y_channel]):
            return self._create_empty_figure(f"密度プロット: {x_channel} vs {y_channel}")
        
        try:
            data_subset = self._subsample_data()
            x_data = self._apply_transform(data_subset[x_channel], x_transform)
            y_data = self._apply_transform(data_subset[y_channel], y_transform)
            
            # 2Dヒストグラムを作成
            hist, x_edges, y_edges = np.histogram2d(x_data, y_data, bins=bins)
            hist_smooth = gaussian_filter(hist, sigma=1)
            
            fig = go.Figure(data=go.Contour(
                z=hist_smooth.T,
                x=x_edges[:-1],
                y=y_edges[:-1],
                colorscale='viridis',
                showscale=True
            ))
            
            fig.update_layout(
                title=title or f"密度プロット: {x_channel} vs {y_channel}",
                xaxis_title=x_channel,
                yaxis_title=y_channel,
                width=DEFAULT_CONFIG['figure_size'][0],
                height=DEFAULT_CONFIG['figure_size'][1]
            )
            
            return fig
            
        except Exception as e:
            st.error(f"密度プロット作成エラー: {e}")
            return self._create_empty_figure(f"密度プロット: {x_channel} vs {y_channel}")
    
    def create_multi_histogram(self, channels, bins=None, transform='linear'):
        """複数のヒストグラムを作成"""
        if not self._validate_inputs(channels):
            return self._create_empty_figure("複数ヒストグラム")
        
        try:
            data_subset = self._subsample_data()
            
            if bins is None:
                bins = DEFAULT_CONFIG['default_bins']
            
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
                title_text="複数チャンネルヒストグラム",
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            st.error(f"複数ヒストグラム作成エラー: {e}")
            return self._create_empty_figure("複数ヒストグラム")
    
    def create_correlation_heatmap(self, channels=None, method='pearson'):
        """相関ヒートマップを作成"""
        if not self._validate_inputs(channels or self.channels):
            return self._create_empty_figure("相関ヒートマップ")
        
        try:
            if channels is None:
                # 数値チャンネルのみを選択
                numeric_channels = self.data.select_dtypes(include=[np.number]).columns
                channels = list(numeric_channels)[:10]  # 最大10チャンネル
            
            data_subset = self._subsample_data()
            corr_data = data_subset[channels].corr(method=method)
            
            fig = px.imshow(
                corr_data,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title=f"チャンネル相関ヒートマップ ({method})",
                color_continuous_midpoint=0
            )
            
            fig.update_layout(
                width=DEFAULT_CONFIG['figure_size'][0],
                height=DEFAULT_CONFIG['figure_size'][1]
            )
            
            return fig
            
        except Exception as e:
            st.error(f"相関ヒートマップ作成エラー: {e}")
            return self._create_empty_figure("相関ヒートマップ")
    
    def get_channel_statistics(self, channel, gate_data=None, transform='linear'):
        """チャンネルの統計情報を取得"""
        if not self._validate_inputs([channel]):
            return None
        
        try:
            data_to_analyze = gate_data if gate_data is not None else self.data
            channel_data = self._apply_transform(data_to_analyze[channel], transform)
            
            stats = {
                'チャンネル': channel,
                '変換': transform,
                'カウント': len(channel_data),
                '平均': float(np.mean(channel_data)),
                '中央値': float(np.median(channel_data)),
                '標準偏差': float(np.std(channel_data)),
                '最小値': float(np.min(channel_data)),
                '最大値': float(np.max(channel_data)),
                '25%': float(np.percentile(channel_data, 25)),
                '75%': float(np.percentile(channel_data, 75)),
                'CV(%)': float(np.std(channel_data) / np.mean(channel_data) * 100) if np.mean(channel_data) != 0 else 0
            }
            
            return stats
            
        except Exception as e:
            st.error(f"統計情報取得エラー: {e}")
            return None
    
    def create_statistics_table(self, channels=None, gate_data=None, transform='linear'):
        """統計テーブルを作成"""
        if not self._validate_inputs(channels or self.channels):
            return None
        
        try:
            if channels is None:
                channels = self.channels[:10]  # 最大10チャンネル
            
            stats_data = []
            for channel in channels:
                if channel in self.channels:
                    stats = self.get_channel_statistics(channel, gate_data, transform)
                    if stats:
                        stats_data.append(stats)
            
            if stats_data:
                df = pd.DataFrame(stats_data)
                # 数値列を3桁に丸める
                numeric_columns = ['平均', '中央値', '標準偏差', '最小値', '最大値', '25%', '75%', 'CV(%)']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = df[col].round(3)
                return df
            return None
            
        except Exception as e:
            st.error(f"統計テーブル作成エラー: {e}")
            return None

# README.md準拠のモジュールレベル関数
def create_histogram(data, channel, bins=50, title=None, transform='linear', metadata=None):
    """ヒストグラムを作成"""
    plotter = PlottingUtils(data, metadata)
    return plotter.create_histogram(channel, bins, title, transform)

def create_scatter_plot(data, x_channel, y_channel, title=None, alpha=0.6, 
                       n_points=None, transform='linear', metadata=None):
    """散布図を作成"""
    plotter = PlottingUtils(data, metadata)
    return plotter.create_scatter_plot(x_channel, y_channel, title, alpha, 
                                      n_points, transform, transform)

def create_density_plot(data, x_channel, y_channel, bins=50, title=None, 
                       transform='linear', metadata=None):
    """密度プロットを作成"""
    plotter = PlottingUtils(data, metadata)
    return plotter.create_density_plot(x_channel, y_channel, bins, title, 
                                      transform, transform)

def create_multi_histogram(data, channels, bins=None, transform='linear', metadata=None):
    """複数ヒストグラムを作成"""
    plotter = PlottingUtils(data, metadata)
    return plotter.create_multi_histogram(channels, bins, transform)

def create_correlation_heatmap(data, channels=None, method='pearson', metadata=None):
    """相関ヒートマップを作成"""
    plotter = PlottingUtils(data, metadata)
    return plotter.create_correlation_heatmap(channels, method)

def create_interactive_scatter(data, x_channel, y_channel, width=800, height=600, 
                              x_transform='linear', y_transform='linear', alpha=0.6):
    """インタラクティブ散布図を作成"""
    plotter = PlottingUtils(data)
    return plotter.create_scatter_plot(x_channel, y_channel, None, alpha, 
                                      None, x_transform, y_transform)

def create_enhanced_histogram(data, channel, bins=50, transform='linear', show_stats=True, metadata=None):
    """統計情報付きヒストグラムを作成"""
    plotter = PlottingUtils(data, metadata)
    fig = plotter.create_histogram(channel, bins, None, transform)
    
    if show_stats and data is not None and channel in data.columns:
        try:
            stats = plotter.get_channel_statistics(channel, transform=transform)
            if stats:
                stats_text = f"平均: {stats['平均']:.2f}<br>中央値: {stats['中央値']:.2f}<br>標準偏差: {stats['標準偏差']:.2f}<br>CV: {stats['CV(%)']:.1f}%"
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
            st.warning(f"統計情報の追加でエラーが発生しました: {e}")
    
    return fig

# 後方互換性のための別名
FCSPlotter = PlottingUtils
