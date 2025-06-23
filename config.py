# config.py - Configuration file for FACS Data Analysis App
# Updated for fcsparser migration from flowkit

import streamlit as st

# App configuration
APP_CONFIG = {
    'title': 'FACS Data Analysis',
    'page_title': 'FACSデータ解析',
    'page_icon': '🔬',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# File upload configuration
UPLOAD_CONFIG = {
    'max_file_size': 100,  # MB
    'allowed_extensions': ['.fcs'],
    'upload_help': 'FCS 2.0/3.0/3.1形式のファイルをアップロードしてください（最大100MB）'
}

# Plotting configuration
PLOT_CONFIG = {
    'default_bins': 100,
    'max_bins': 500,
    'default_alpha': 0.7,
    'colormap_options': ['viridis', 'plasma', 'inferno', 'magma', 'cividis'],
    'default_colormap': 'viridis',
    'figure_size': (800, 600),
    'scatter_size': 2
}

# Gating configuration
GATING_CONFIG = {
    'gate_colors': ['red', 'blue', 'green', 'orange', 'purple'],
    'default_gate_color': 'red',
    'gate_line_width': 2,
    'gate_alpha': 0.3,
    'min_gate_points': 3
}

# Data processing configuration
DATA_CONFIG = {
    'max_events_display': 50000,  # Maximum events to display for performance
    'subsample_for_plot': True,
    'default_subsample_size': 10000,
    'compensation_methods': ['none', 'spillover_matrix'],
    'transform_methods': ['linear', 'log', 'asinh', 'biexponential']
}

# Error messages (Japanese)
ERROR_MESSAGES = {
    'file_too_large': f'ファイルサイズが{UPLOAD_CONFIG["max_file_size"]}MBを超えています。',
    'invalid_file_format': 'サポートされていないファイル形式です。FCSファイルをアップロードしてください。',
    'file_read_error': 'ファイルの読み込みに失敗しました。ファイルが破損している可能性があります。',
    'insufficient_data': 'データが不十分です。より多くのイベントが必要です。',
    'channel_not_found': '指定されたチャンネルが見つかりません。',
    'gating_error': 'ゲーティング処理中にエラーが発生しました。',
    'export_error': 'データのエクスポート中にエラーが発生しました。',
    'plot_error': 'プロットの作成中にエラーが発生しました。',
    'memory_error': 'メモリ不足です。データサイズを小さくしてください。'
}

# Success messages (Japanese)
SUCCESS_MESSAGES = {
    'file_loaded': 'ファイルが正常に読み込まれました。',
    'gate_created': 'ゲートが正常に作成されました。',
    'data_exported': 'データが正常にエクスポートされました。',
    'settings_saved': '設定が保存されました。'
}

# Warning messages (Japanese)
WARNING_MESSAGES = {
    'large_file': 'ファイルサイズが大きいため、処理に時間がかかる場合があります。',
    'many_events': 'イベント数が多いため、表示をサブサンプリングしています。',
    'no_compensation': '補正行列が適用されていません。',
    'performance_warning': 'パフォーマンスを向上させるため、一部のデータのみ表示しています。'
}

# Channel name mappings (common flow cytometry channels)
CHANNEL_MAPPINGS = {
    'FSC-A': 'Forward Scatter Area',
    'FSC-H': 'Forward Scatter Height',
    'FSC-W': 'Forward Scatter Width',
    'SSC-A': 'Side Scatter Area',
    'SSC-H': 'Side Scatter Height',
    'SSC-W': 'Side Scatter Width',
    'FITC-A': 'FITC Area',
    'PE-A': 'PE Area',
    'APC-A': 'APC Area',
    'PerCP-A': 'PerCP Area',
    'PE-Cy7-A': 'PE-Cy7 Area',
    'APC-Cy7-A': 'APC-Cy7 Area'
}

# FCSParser specific configuration
# Updated to use fcsparser instead of flowkit
FCSPARSER_CONFIG = {
    'read_data': True,
    'reformat_meta': True,
    'data_set': 0,  # For multi-data FCS files
    'channel_naming': '$PnN',  # Use $PnN for channel names
    'apply_compensation': False,  # Will be handled separately
    'transform': 'linear'  # Default transform
}

# Statistics configuration
STATS_CONFIG = {
    'default_statistics': ['count', 'mean', 'median', 'std', 'min', 'max'],
    'percentiles': [5, 25, 50, 75, 95],
    'correlation_methods': ['pearson', 'spearman'],
    'density_estimation': 'gaussian_kde'
}

# Export configuration
EXPORT_CONFIG = {
    'csv_separator': ',',
    'include_metadata': True,
    'include_statistics': True,
    'timestamp_format': '%Y%m%d_%H%M%S',
    'file_prefix': 'facs_analysis_'
}

# Configクラスを追加（advanced_gating.py の互換性のため）
class Config:
    """Configuration class for backward compatibility"""
    
    # App settings
    APP_TITLE = APP_CONFIG['title']
    PAGE_TITLE = APP_CONFIG['page_title']
    PAGE_ICON = APP_CONFIG['page_icon']
    
    # File upload settings
    MAX_FILE_SIZE = UPLOAD_CONFIG['max_file_size']
    ALLOWED_EXTENSIONS = UPLOAD_CONFIG['allowed_extensions']
    
    # Plot settings
    DEFAULT_BINS = PLOT_CONFIG['default_bins']
    MAX_BINS = PLOT_CONFIG['max_bins']
    DEFAULT_ALPHA = PLOT_CONFIG['default_alpha']
    COLORMAP_OPTIONS = PLOT_CONFIG['colormap_options']
    DEFAULT_COLORMAP = PLOT_CONFIG['default_colormap']
    
    # Gating settings
    GATE_COLORS = GATING_CONFIG['gate_colors']
    DEFAULT_GATE_COLOR = GATING_CONFIG['default_gate_color']
    GATE_LINE_WIDTH = GATING_CONFIG['gate_line_width']
    GATE_ALPHA = GATING_CONFIG['gate_alpha']
    MIN_GATE_POINTS = GATING_CONFIG['min_gate_points']
    
    # Data processing settings
    MAX_EVENTS_DISPLAY = DATA_CONFIG['max_events_display']
    SUBSAMPLE_FOR_PLOT = DATA_CONFIG['subsample_for_plot']
    DEFAULT_SUBSAMPLE_SIZE = DATA_CONFIG['default_subsample_size']
    TRANSFORM_METHODS = DATA_CONFIG['transform_methods']
    
    # Channel mappings
    CHANNEL_MAPPINGS = CHANNEL_MAPPINGS
    
    @classmethod
    def get_error_message(cls, error_key):
        """Get error message by key"""
        return ERROR_MESSAGES.get(error_key, 'Unknown error occurred.')
    
    @classmethod
    def get_success_message(cls, success_key):
        """Get success message by key"""
        return SUCCESS_MESSAGES.get(success_key, 'Operation completed successfully.')
    
    @classmethod
    def get_warning_message(cls, warning_key):
        """Get warning message by key"""
        return WARNING_MESSAGES.get(warning_key, 'Warning: Please check your data.')
    
    @classmethod
    def get_gate_colors(cls):
        """Get available gate colors"""
        return cls.GATE_COLORS
    
    @classmethod
    def get_transform_methods(cls):
        """Get available transform methods"""
        return cls.TRANSFORM_METHODS
    
    @classmethod
    def get_colormap_options(cls):
        """Get available colormap options"""
        return cls.COLORMAP_OPTIONS

def get_config_dict():
    """Return all configuration as a dictionary"""
    return {
        'app': APP_CONFIG,
        'upload': UPLOAD_CONFIG,
        'plot': PLOT_CONFIG,
        'gating': GATING_CONFIG,
        'data': DATA_CONFIG,
        'fcsparser': FCSPARSER_CONFIG,
        'stats': STATS_CONFIG,
        'export': EXPORT_CONFIG
    }

def validate_file_size(file_size_bytes):
    """Validate uploaded file size"""
    max_size_bytes = UPLOAD_CONFIG['max_file_size'] * 1024 * 1024
    return file_size_bytes <= max_size_bytes

def get_error_message(error_key):
    """Get localized error message"""
    return ERROR_MESSAGES.get(error_key, 'Unknown error occurred.')

def get_success_message(success_key):
    """Get localized success message"""
    return SUCCESS_MESSAGES.get(success_key, 'Operation completed successfully.')

def get_warning_message(warning_key):
    """Get localized warning message"""
    return WARNING_MESSAGES.get(warning_key, 'Warning: Please check your data.')

# =============================================================================
# 拡張設定 - README.mdとの整合性確保
# =============================================================================

# ページ固有の設定
PAGE_CONFIG = {
    'main_app': {
        'use_fcs_processor': True,
        'max_events_range': (1000, 100000),
        'default_max_events': DATA_CONFIG['max_events_display'],  # 既存設定を参照
        'export_filenames': {
            'statistics': 'facs_statistics.csv',
            'raw_data': 'facs_raw_data.csv'
        }
    },
    'basic_analysis': {
        'use_fcsparser_direct': True,
        'max_events_range': (1000, 100000),
        'default_max_events': DATA_CONFIG['max_events_display'],
        'dynamic_filename': True
    }
}

# 変換方法の統一（既存のDATA_CONFIG['transform_methods']を拡張）
TRANSFORM_METHODS_UNIFIED = {
    'none': {'label': 'なし', 'function': 'linear'},
    'log10': {'label': 'Log10', 'function': 'log'},
    'asinh': {'label': 'Asinh', 'function': 'asinh'},
    'biexponential': {'label': 'Biexponential', 'function': 'biexponential'}
}

# 既存のDATA_CONFIGを更新
DATA_CONFIG.update({
    'max_events_range': (1000, 100000),
    'min_events_display': 1000,
    'transform_methods_unified': TRANSFORM_METHODS_UNIFIED
})

# 既存のConfigクラスを拡張
class Config:
    # 既存の設定はそのまま維持
    APP_TITLE = APP_CONFIG['title']
    PAGE_TITLE = APP_CONFIG['page_title']
    # ... 既存の設定 ...
    
    # 新しい設定を追加
    PAGE_CONFIG = PAGE_CONFIG
    TRANSFORM_METHODS_UNIFIED = TRANSFORM_METHODS_UNIFIED
    
    @classmethod
    def get_page_config(cls, page_name):
        """ページ固有の設定を取得"""
        return PAGE_CONFIG.get(page_name, {})
    
    @classmethod
    def get_transform_options(cls):
        """変換方法のオプションを取得"""
        return list(TRANSFORM_METHODS_UNIFIED.keys())
    
    @classmethod
    def get_transform_label(cls, transform_key):
        """変換方法の日本語ラベルを取得"""
        return TRANSFORM_METHODS_UNIFIED.get(transform_key, {}).get('label', transform_key)
