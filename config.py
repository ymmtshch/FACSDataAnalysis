# config.py - Simplified Configuration for FACS Data Analysis App

import streamlit as st

# App Configuration
APP_CONFIG = {
    'title': 'FACS Data Analysis',
    'page_title': 'FACSデータ解析',
    'page_icon': '🔬',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# File Upload Configuration
UPLOAD_CONFIG = {
    'max_file_size': 100,  # MB
    'allowed_extensions': ['.fcs'],
    'supported_formats': ['FCS 2.0', 'FCS 3.0', 'FCS 3.1']
}

# FCS Library Configuration
FCS_LIBRARY_CONFIG = {
    'primary_library': 'fcsparser',
    'fallback_libraries': ['flowio', 'flowkit']
}

# Plot Configuration
PLOT_CONFIG = {
    'default_bins': 100,
    'max_bins': 500,
    'default_alpha': 0.7,
    'colormap_options': ['viridis', 'plasma', 'inferno', 'magma', 'cividis'],
    'default_colormap': 'viridis',
    'figure_size': (800, 600),
    'plot_types': ['scatter', 'density', 'histogram']
}

# Gating Configuration
GATING_CONFIG = {
    'gate_types': {
        'threshold': {'label': '閾値ゲート', 'description': '単一パラメータでの閾値設定'},
        'rectangular': {'label': '矩形ゲート', 'description': '2次元での矩形領域選択'},
        'polygon': {'label': 'ポリゴンゲート', 'description': '任意の多角形領域での選択'},
        'ellipse': {'label': '楕円ゲート', 'description': '楕円形領域での選択'}
    },
    'gate_colors': ['red', 'blue', 'green', 'orange', 'purple', 'cyan'],
    'default_gate_color': 'red',
    'max_gates': 10
}

# Data Processing Configuration
DATA_CONFIG = {
    'min_events_display': 1000,
    'max_events_display': 100000,
    'default_max_events': 50000,
    'transform_methods': {
        'none': {'label': 'なし'},
        'log10': {'label': 'Log10'},
        'asinh': {'label': 'Asinh'},
        'biexponential': {'label': 'Biexponential'}
    }
}

# Page Configuration
PAGE_CONFIG = {
    'main_tabs': ['📊 基本情報', '📈 可視化', '🎯 ゲーティング', '📋 統計解析'],
    'basic_analysis': {
        'export_format': '{filename}_stats.csv',
        'data_export_format': '{filename}_data.csv'
    },
    'advanced_gating': {
        'gate_types': ['rectangular', 'polygon', 'ellipse', 'threshold'],
        'realtime_display': True
    }
}

# Statistics Configuration
STATS_CONFIG = {
    'basic_statistics': ['count', 'mean', 'median', 'std', 'min', 'max'],
    'percentiles': [25, 50, 75, 95]
}

# Export Configuration
EXPORT_CONFIG = {
    'csv_separator': ',',
    'include_metadata': True,
    'file_prefix': 'facs_analysis_',
    'export_types': ['statistics', 'raw_data', 'gate_data']
}

# Error Messages (Japanese)
ERROR_MESSAGES = {
    'file_too_large': f'ファイルサイズが{UPLOAD_CONFIG["max_file_size"]}MBを超えています。',
    'invalid_file_format': 'FCSファイルをアップロードしてください。',
    'file_read_error': 'ファイルの読み込みに失敗しました。',
    'insufficient_data': 'データが不十分です。',
    'memory_error': 'メモリ不足です。データサイズを小さくしてください。',
    'gating_error': 'ゲーティング処理中にエラーが発生しました。'
}

# Success Messages (Japanese)
SUCCESS_MESSAGES = {
    'file_loaded': 'ファイルが正常に読み込まれました。',
    'gate_created': 'ゲートが正常に作成されました。',
    'data_exported': 'データが正常にエクスポートされました。'
}

# Warning Messages (Japanese)
WARNING_MESSAGES = {
    'large_file': 'ファイルサイズが大きいため、処理に時間がかかる場合があります。',
    'many_events': 'イベント数が多いため、表示をサブサンプリングしています。',
    'memory_optimization': 'メモリ最適化のため、サンプリングが適用されています。'
}

# Common Channel Mappings
CHANNEL_MAPPINGS = {
    'FSC-A': 'Forward Scatter Area',
    'FSC-H': 'Forward Scatter Height',
    'SSC-A': 'Side Scatter Area',
    'SSC-H': 'Side Scatter Height',
    'FITC-A': 'FITC Area',
    'PE-A': 'PE Area',
    'APC-A': 'APC Area',
    'PerCP-A': 'PerCP Area'
}

# Utility Functions
def get_transform_methods():
    """Get available transform methods"""
    return DATA_CONFIG['transform_methods']

def get_gate_types():
    """Get available gate types"""
    return GATING_CONFIG['gate_types']

def get_export_filename(base_filename, export_type):
    """Generate export filename"""
    if export_type == 'statistics':
        return f"{base_filename}_stats.csv"
    elif export_type == 'raw_data':
        return f"{base_filename}_data.csv"
    elif export_type == 'gate_data':
        return f"{base_filename}_gate_data.csv"
    else:
        return f"{base_filename}_export.csv"

def validate_max_events(num_events):
    """Validate maximum events setting"""
    min_events = DATA_CONFIG['min_events_display']
    max_events = DATA_CONFIG['max_events_display']
    return max(min_events, min(num_events, max_events))

def validate_file_size(file_size_bytes):
    """Validate uploaded file size"""
    max_size_bytes = UPLOAD_CONFIG['max_file_size'] * 1024 * 1024
    return file_size_bytes <= max_size_bytes

def get_error_message(error_key):
    """Get localized error message"""
    return ERROR_MESSAGES.get(error_key, 'エラーが発生しました。')

def get_success_message(success_key):
    """Get localized success message"""
    return SUCCESS_MESSAGES.get(success_key, '処理が正常に完了しました。')

def get_warning_message(warning_key):
    """Get localized warning message"""
    return WARNING_MESSAGES.get(warning_key, '警告: データを確認してください。')

# Legacy Config class for backward compatibility
class Config:
    """Simplified configuration class for backward compatibility"""
    
    APP_TITLE = APP_CONFIG['title']
    PAGE_TITLE = APP_CONFIG['page_title']
    PAGE_ICON = APP_CONFIG['page_icon']
    
    MAX_FILE_SIZE = UPLOAD_CONFIG['max_file_size']
    ALLOWED_EXTENSIONS = UPLOAD_CONFIG['allowed_extensions']
    
    DEFAULT_BINS = PLOT_CONFIG['default_bins']
    MAX_BINS = PLOT_CONFIG['max_bins']
    DEFAULT_ALPHA = PLOT_CONFIG['default_alpha']
    COLORMAP_OPTIONS = PLOT_CONFIG['colormap_options']
    DEFAULT_COLORMAP = PLOT_CONFIG['default_colormap']
    
    GATE_COLORS = GATING_CONFIG['gate_colors']
    DEFAULT_GATE_COLOR = GATING_CONFIG['default_gate_color']
    
    MAX_EVENTS_DISPLAY = DATA_CONFIG['max_events_display']
    CHANNEL_MAPPINGS = CHANNEL_MAPPINGS
    
    @classmethod
    def get_error_message(cls, error_key):
        return get_error_message(error_key)
    
    @classmethod
    def get_success_message(cls, success_key):
        return get_success_message(success_key)
    
    @classmethod
    def get_warning_message(cls, warning_key):
        return get_warning_message(warning_key)
