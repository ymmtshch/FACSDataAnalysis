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
