"""
FACS解析用プロッティングユーティリティ
Bokehを使用した高品質な可視化機能を提供
"""

import numpy as np
import pandas as pd
import streamlit as st
from bokeh.plotting import figure
from bokeh.models import (
    HoverTool, ColorBar, LinearColorMapper, ColumnDataSource,
    BoxSelectTool, LassoSelectTool, PolySelectTool, ResetTool, PanTool, WheelZoomTool,
    CustomJS, Div, Button, Select, Slider, RangeSlider
)
from bokeh.layouts import column, row
from bokeh.palettes import Viridis256, Category10
from bokeh.transform import linear_cmap
from bokeh.io import curdoc
from scipy import stats
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')


class FACSPlotter:
    """FACS データ用の包括的プロッティングクラス"""
    
    def __init__(self, data: pd.DataFrame, meta: dict = None):
        """
        初期化
        
        Parameters:
        -----------
        data : pd.DataFrame
            FACS データ
        meta : dict, optional
            メタデータ情報
        """
        self.data = data
        self.meta = meta or {}
        self.default_plot_width = 600
        self.default_plot_height = 500
        
    def create_histogram(self, 
                        channel: str, 
                        bins: int = 50,
                        show_stats: bool = True,
                        x_range: tuple = None,
                        title: str = None,
                        color: str = "steelblue",
                        alpha: float = 0.7) -> figure:
        """
        ヒストグラムを作成
        
        Parameters:
        -----------
        channel : str
            解析チャンネル名
        bins : int
            ビン数
        show_stats : bool
            統計線を表示するか
        x_range : tuple
            X軸範囲 (min, max)
        title : str
            プロットタイトル
        color : str
            ヒストグラム色
        alpha : float
            透明度
            
        Returns:
        --------
        figure : bokeh.plotting.figure
        """
        if channel not in self.data.columns:
            raise ValueError(f"Channel '{channel}' not found in data")
            
        data_channel = self.data[channel].dropna()
        
        # ヒストグラム計算
        if x_range:
            hist_data = data_channel[(data_channel >= x_range[0]) & 
                                   (data_channel <= x_range[1])]
        else:
            hist_data = data_channel
            x_range = (hist_data.min(), hist_data.max())
            
        hist, edges = np.histogram(hist_data, bins=bins, range=x_range)
        
        # プロット作成
        p = figure(
            title=title or f"Histogram: {channel}",
            width=self.default_plot_width,
            height=self.default_plot_height,
            x_axis_label=channel,
            y_axis_label="Count",
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )
        
        # ヒストグラム描画
        p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
               fill_color=color, line_color="white", alpha=alpha)
        
        # 統計線追加
        if show_stats:
            mean_val = hist_data.mean()
            median_val = hist_data.median()
            
            p.line([mean_val, mean_val], [0, hist.max()], 
                   line_color="red", line_width=2, legend_label="Mean", line_dash="dashed")
            p.line([median_val, median_val], [0, hist.max()], 
                   line_color="green", line_width=2, legend_label="Median", line_dash="dotted")
            
            p.legend.location = "top_right"
            
        return p
        
    def create_scatter_plot(self,
                           x_channel: str,
                           y_channel: str,
                           color_channel: str = None,
                           size: int = 3,
                           alpha: float = 0.6,
                           x_range: tuple = None,
                           y_range: tuple = None,
                           title: str = None,
                           sample_size: int = None,
                           add_selection_tools: bool = False) -> figure:
        """
        散布図を作成
        
        Parameters:
        -----------
        x_channel, y_channel : str
            X, Y軸チャンネル名
        color_channel : str, optional
            色分けチャンネル名
        size : int
            マーカーサイズ
        alpha : float
            透明度
        x_range, y_range : tuple, optional
            軸範囲 (min, max)
        title : str, optional
            プロットタイトル
        sample_size : int, optional
            サンプリングサイズ
        add_selection_tools : bool
            選択ツールを追加するか
            
        Returns:
        --------
        figure : bokeh.plotting.figure
        """
        # データ準備
        plot_data = self.data[[x_channel, y_channel]].dropna()
        
        if color_channel and color_channel in self.data.columns:
            plot_data[color_channel] = self.data[color_channel]
            
        # サンプリング
        if sample_size and len(plot_data) > sample_size:
            plot_data = plot_data.sample(n=sample_size, random_state=42)
            
        # 範囲設定
        if x_range:
            plot_data = plot_data[(plot_data[x_channel] >= x_range[0]) & 
                                (plot_data[x_channel] <= x_range[1])]
        if y_range:
            plot_data = plot_data[(plot_data[y_channel] >= y_range[0]) & 
                                (plot_data[y_channel] <= y_range[1])]
        
        # プロット作成
        tools = "pan,wheel_zoom,box_zoom,reset,save"
        if add_selection_tools:
            tools += ",box_select,lasso_select,poly_select"
            
        p = figure(
            title=title or f"{y_channel} vs {x_channel}",
            width=self.default_plot_width,
            height=self.default_plot_height,
            x_axis_label=x_channel,
            y_axis_label=y_channel,
            tools=tools
        )
        
        # データソース作成
        source = ColumnDataSource(plot_data)
        
        # 色設定
        if color_channel and color_channel in plot_data.columns:
            mapper = LinearColorMapper(
                palette=Viridis256,
                low=plot_data[color_channel].min(),
                high=plot_data[color_channel].max()
            )
            
            scatter = p.scatter(
                x=x_channel, y=y_channel, 
                source=source, 
                size=size, alpha=alpha,
                color={'field': color_channel, 'transform': mapper}
            )
            
            # カラーバー追加
            color_bar = ColorBar(color_mapper=mapper, width=8, location=(0,0))
            p.add_layout(color_bar, 'right')
            
        else:
            scatter = p.scatter(
                x=x_channel, y=y_channel,
                source=source,
                size=size, alpha=alpha,
                color="navy"
            )
        
        # ホバーツール
        hover = HoverTool(
            tooltips=[
                (f"{x_channel}", f"@{x_channel}"),
                (f"{y_channel}", f"@{y_channel}"),
                ("Index", "$index")
            ]
        )
        if color_channel and color_channel in plot_data.columns:
            hover.tooltips.append((f"{color_channel}", f"@{color_channel}"))
            
        p.add_tools(hover)
        
        return p, source
        
    def create_contour_plot(self,
                           x_channel: str,
                           y_channel: str,
                           bins: int = 50,
                           smoothing: float = 1.0,
                           x_range: tuple = None,
                           y_range: tuple = None,
                           title: str = None,
                           show_scatter: bool = True,
                           scatter_alpha: float = 0.3,
                           sample_size: int = 10000) -> figure:
        """
        等高線プロットを作成
        
        Parameters:
        -----------
        x_channel, y_channel : str
            X, Y軸チャンネル名
        bins : int
            ビン数
        smoothing : float
            スムージング強度
        x_range, y_range : tuple, optional
            軸範囲
        title : str, optional
            プロットタイトル
        show_scatter : bool
            散布図を重ねて表示するか
        scatter_alpha : float
            散布図の透明度
        sample_size : int
            散布図用サンプルサイズ
            
        Returns:
        --------
        figure : bokeh.plotting.figure
        """
        # データ準備
        plot_data = self.data[[x_channel, y_channel]].dropna()
        
        if x_range:
            plot_data = plot_data[(plot_data[x_channel] >= x_range[0]) & 
                                (plot_data[x_channel] <= x_range[1])]
        if y_range:
            plot_data = plot_data[(plot_data[y_channel] >= y_range[0]) & 
                                (plot_data[y_channel] <= y_range[1])]
            
        # 2Dヒストグラム計算
        x_data = plot_data[x_channel].values
        y_data = plot_data[y_channel].values
        
        # 範囲設定
        x_min, x_max = x_data.min(), x_data.max()
        y_min, y_max = y_data.min(), y_data.max()
        
        if x_range:
            x_min, x_max = x_range
        if y_range:
            y_min, y_max = y_range
            
        # ヒストグラム計算
        hist, x_edges, y_edges = np.histogram2d(
            x_data, y_data, bins=bins,
            range=[[x_min, x_max], [y_min, y_max]]
        )
        
        # スムージング適用
        if smoothing > 0:
            hist = gaussian_filter(hist, sigma=smoothing)
            
        # プロット作成
        p = figure(
            title=title or f"Contour: {y_channel} vs {x_channel}",
            width=self.default_plot_width,
            height=self.default_plot_height,
            x_axis_label=x_channel,
            y_axis_label=y_channel,
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )
        
        # 等高線データ準備
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        
        # メッシュグリッド作成
        X, Y = np.meshgrid(x_centers, y_centers)
        
        # 等高線レベル設定
        levels = np.linspace(hist.min(), hist.max(), 10)[1:]  # 最小値を除外
        
        # 等高線描画
        p.image(
            image=[hist.T], x=x_min, y=y_min, 
            dw=x_max-x_min, dh=y_max-y_min,
            palette=Viridis256, alpha=0.8
        )
        
        # 散布図を重ねて表示
        if show_scatter:
            scatter_data = plot_data.sample(
                n=min(sample_size, len(plot_data)), 
                random_state=42
            )
            p.scatter(
                x=scatter_data[x_channel], 
                y=scatter_data[y_channel],
                size=2, alpha=scatter_alpha, color="white"
            )
            
        return p
        
    def create_density_plot(self,
                           x_channel: str,
                           y_channel: str,
                           bins: int = 100,
                           x_range: tuple = None,
                           y_range: tuple = None,
                           title: str = None) -> figure:
        """
        密度プロット（ヒートマップ）を作成
        
        Parameters:
        -----------
        x_channel, y_channel : str
            X, Y軸チャンネル名
        bins : int
            ビン数
        x_range, y_range : tuple, optional
            軸範囲
        title : str, optional
            プロットタイトル
            
        Returns:
        --------
        figure : bokeh.plotting.figure
        """
        # データ準備
        plot_data = self.data[[x_channel, y_channel]].dropna()
        
        if x_range:
            plot_data = plot_data[(plot_data[x_channel] >= x_range[0]) & 
                                (plot_data[x_channel] <= x_range[1])]
        if y_range:
            plot_data = plot_data[(plot_data[y_channel] >= y_range[0]) & 
                                (plot_data[y_channel] <= y_range[1])]
            
        # 2Dヒストグラム計算
        x_data = plot_data[x_channel].values
        y_data = plot_data[y_channel].values
        
        hist, x_edges, y_edges = np.histogram2d(
            x_data, y_data, bins=bins
        )
        
        # プロット作成
        p = figure(
            title=title or f"Density: {y_channel} vs {x_channel}",
            width=self.default_plot_width,
            height=self.default_plot_height,
            x_axis_label=x_channel,
            y_axis_label=y_channel,
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )
        
        # 密度プロット描画
        p.image(
            image=[hist.T], 
            x=x_edges[0], y=y_edges[0],
            dw=x_edges[-1]-x_edges[0], 
            dh=y_edges[-1]-y_edges[0],
            palette=Viridis256
        )
        
        # カラーバー追加
        color_mapper = LinearColorMapper(
            palette=Viridis256,
            low=hist.min(),
            high=hist.max()
        )
        color_bar = ColorBar(color_mapper=color_mapper, width=8, location=(0,0))
        p.add_layout(color_bar, 'right')
        
        return p
        
    def create_interactive_gating_plot(self,
                                     x_channel: str,
                                     y_channel: str,
                                     gate_callback: str = None,
                                     x_range: tuple = None,
                                     y_range: tuple = None,
                                     title: str = None,
                                     sample_size: int = 50000) -> tuple:
        """
        インタラクティブなゲーティング用プロットを作成
        
        Parameters:
        -----------
        x_channel, y_channel : str
            X, Y軸チャンネル名
        gate_callback : str, optional
            ゲート選択時のJavaScriptコールバック
        x_range, y_range : tuple, optional
            軸範囲
        title : str, optional
            プロットタイトル
        sample_size : int
            表示用サンプルサイズ
            
        Returns:
        --------
        tuple : (figure, ColumnDataSource, selection_tools)
        """
        # データ準備
        plot_data = self.data[[x_channel, y_channel]].dropna()
        
        # サンプリング
        if len(plot_data) > sample_size:
            plot_data = plot_data.sample(n=sample_size, random_state=42)
            
        # 範囲フィルタリング
        if x_range:
            plot_data = plot_data[(plot_data[x_channel] >= x_range[0]) & 
                                (plot_data[x_channel] <= x_range[1])]
        if y_range:
            plot_data = plot_data[(plot_data[y_channel] >= y_range[0]) & 
                                (plot_data[y_channel] <= y_range[1])]
            
        # データソース作成
        source = ColumnDataSource(plot_data)
        
        # 選択ツール作成
        box_select = BoxSelectTool()
        lasso_select = LassoSelectTool()
        poly_select = PolySelectTool()
        
        # プロット作成
        p = figure(
            title=title or f"Gating: {y_channel} vs {x_channel}",
            width=self.default_plot_width,
            height=self.default_plot_height,
            x_axis_label=x_channel,
            y_axis_label=y_channel,
            tools=[PanTool(), WheelZoomTool(), box_select, lasso_select, 
                   poly_select, ResetTool()]
        )
        
        # 散布図描画
        scatter = p.scatter(
            x=x_channel, y=y_channel,
            source=source,
            size=3, alpha=0.6,
            color="navy",
            selection_color="red",
            nonselection_alpha=0.1
        )
        
        # カスタムJSコールバック
        if gate_callback:
            source.selected.js_on_change('indices', CustomJS(
                args=dict(source=source),
                code=gate_callback
            ))
            
        selection_tools = {
            'box_select': box_select,
            'lasso_select': lasso_select,
            'poly_select': poly_select
        }
        
        return p, source, selection_tools
        
    def create_multi_parameter_plot(self,
                                   channels: list,
                                   plot_type: str = "scatter_matrix",
                                   sample_size: int = 10000) -> list:
        """
        多次元パラメータプロットを作成
        
        Parameters:
        -----------
        channels : list
            解析チャンネルリスト
        plot_type : str
            プロットタイプ ("scatter_matrix", "parallel_coordinates")
        sample_size : int
            サンプルサイズ
            
        Returns:
        --------
        list : プロットリスト
        """
        if plot_type == "scatter_matrix":
            return self._create_scatter_matrix(channels, sample_size)
        elif plot_type == "parallel_coordinates":
            return self._create_parallel_coordinates(channels, sample_size)
        else:
            raise ValueError(f"Unsupported plot_type: {plot_type}")
            
    def _create_scatter_matrix(self, channels: list, sample_size: int) -> list:
        """散布図マトリックスを作成"""
        plots = []
        n_channels = len(channels)
        
        # データサンプリング
        plot_data = self.data[channels].dropna()
        if len(plot_data) > sample_size:
            plot_data = plot_data.sample(n=sample_size, random_state=42)
            
        for i in range(n_channels):
            plot_row = []
            for j in range(n_channels):
                if i == j:
                    # 対角線: ヒストグラム
                    p = self.create_histogram(
                        channels[i], 
                        title=f"{channels[i]}",
                        bins=30
                    )
                    p.width = 300
                    p.height = 300
                else:
                    # 非対角線: 散布図
                    p, _ = self.create_scatter_plot(
                        channels[j], channels[i],
                        size=2, alpha=0.5,
                        title=""
                    )
                    p.width = 300
                    p.height = 300
                    
                plot_row.append(p)
            plots.append(plot_row)
            
        return plots
        
    def _create_parallel_coordinates(self, channels: list, sample_size: int) -> list:
        """平行座標プロットを作成"""
        # データ準備
        plot_data = self.data[channels].dropna()
        if len(plot_data) > sample_size:
            plot_data = plot_data.sample(n=sample_size, random_state=42)
            
        # 正規化
        normalized_data = (plot_data - plot_data.min()) / (plot_data.max() - plot_data.min())
        
        # プロット作成
        p = figure(
            title="Parallel Coordinates",
            width=800,
            height=400,
            x_range=list(range(len(channels))),
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )
        
        # 各サンプルの線を描画
        for idx in range(len(normalized_data)):
            y_values = normalized_data.iloc[idx].values
            x_values = list(range(len(channels)))
            p.line(x_values, y_values, alpha=0.1, color="navy")
            
        # 軸ラベル設定
        p.xaxis.ticker = list(range(len(channels)))
        p.xaxis.major_label_overrides = {i: channels[i] for i in range(len(channels))}
        
        return [p]
        
    def get_plot_statistics(self, channel: str, gate_indices: list = None) -> dict:
        """
        プロット統計情報を取得
        
        Parameters:
        -----------
        channel : str
            チャンネル名
        gate_indices : list, optional
            ゲート内インデックス
            
        Returns:
        --------
        dict : 統計情報
        """
        if gate_indices is not None:
            data = self.data.iloc[gate_indices][channel]
        else:
            data = self.data[channel]
            
        data = data.dropna()
        
        stats_dict = {
            'count': len(data),
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'q25': data.quantile(0.25),
            'q75': data.quantile(0.75),
            'cv': data.std() / data.mean() * 100 if data.mean() != 0 else 0
        }
        
        return stats_dict


def create_plot_controls(channels: list) -> dict:
    """
    プロット制御用UIコンポーネントを作成
    
    Parameters:
    -----------
    channels : list
        利用可能チャンネルリスト
        
    Returns:
    --------
    dict : 制御コンポーネント辞書
    """
    controls = {}
    
    # チャンネル選択
    controls['x_channel'] = Select(
        title="X Channel:",
        value=channels[0] if channels else "",
        options=channels
    )
    
    controls['y_channel'] = Select(
        title="Y Channel:",
        value=channels[1] if len(channels) > 1 else channels[0],
        options=channels
    )
    
    # プロットタイプ選択
    controls['plot_type'] = Select(
        title="Plot Type:",
        value="scatter",
        options=["scatter", "contour", "density", "histogram"]
    )
    
    # サンプルサイズ
    controls['sample_size'] = Slider(
        title="Sample Size:",
        start=1000, end=100000, value=10000, step=1000
    )
    
    # ビン数
    controls['bins'] = Slider(
        title="Bins:",
        start=10, end=200, value=50, step=5
    )
    
    # 透明度
    controls['alpha'] = Slider(
        title="Alpha:",
        start=0.1, end=1.0, value=0.6, step=0.1
    )
    
    return controls


def create_gate_info_display() -> Div:
    """
    ゲート情報表示用divを作成
    
    Returns:
    --------
    Div : 情報表示用div
    """
    return Div(
        text="<h3>Gate Information</h3><p>No gate selected</p>",
        width=300,
        height=200
    )
