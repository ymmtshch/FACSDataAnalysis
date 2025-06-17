# config.py
"""
FACS解析アプリケーション設定ファイル
"""

# アプリケーション設定
APP_CONFIG = {
    'title': 'FACS Data Analysis Platform',
    'version': '1.0.0',
    'description': 'Streamlit-based Flow Cytometry Analysis Tool',
    'author': 'FACS Analysis Team',
    'max_file_size': 100,  # MB
    'supported_formats': ['.fcs', '.FCS', '.csv', '.xlsx']
}

# UI設定
UI_CONFIG = {
    'page_icon': '🔬',
    'layout': 'wide',
    'sidebar_width': 300,
    'plot_height': 500,
    'plot_width': 700,
    'color_palette': [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf'
    ]
}

# プロット設定
PLOT_CONFIG = {
    'default_bins': 50,
    'contour_levels': 10,
    'density_colormap': 'viridis',
    'histogram_alpha': 0.7,
    'scatter_alpha': 0.6,
    'scatter_size': 2,
    'line_width': 2,
    'grid_alpha': 0.3
}

# ゲーティング設定
GATING_CONFIG = {
    'default_gate_color': '#ff0000',
    'gate_line_width': 2,
    'gate_alpha': 0.3,
    'selection_color': '#00ff00',
    'min_gate_points': 3,
    'gate_types': ['polygon', 'rectangle', 'ellipse', 'threshold']
}

# 統計設定
STATS_CONFIG = {
    'percentiles': [1, 5, 10, 25, 50, 75, 90, 95, 99],
    'statistical_measures': ['count', 'mean', 'median', 'std', 'min', 'max'],
    'outlier_threshold': 3  # 標準偏差の倍数
}

# データ処理設定
DATA_CONFIG = {
    'compensation_method': 'spillover_matrix',
    'transformation_methods': ['asinh', 'log', 'biexponential', 'linear'],
    'default_transformation': 'asinh',
    'asinh_cofactor': 150,
    'log_base': 10,
    'negative_handling': 'truncate'  # 'truncate' or 'shift'
}

# セッション状態のキー
SESSION_KEYS = {
    'fcs_data': 'fcs_data',
    'fcs_meta': 'fcs_meta',
    'current_gates': 'current_gates',
    'gate_stats': 'gate_stats',
    'selected_channels': 'selected_channels',
    'transformation_params': 'transformation_params',
    'plot_settings': 'plot_settings',
    'analysis_history': 'analysis_history'
}

# エラーメッセージ
ERROR_MESSAGES = {
    'file_not_found': 'ファイルが見つかりません。',
    'invalid_format': 'サポートされていないファイル形式です。',
    'file_too_large': f'ファイルサイズが{APP_CONFIG["max_file_size"]}MBを超えています。',
    'parsing_error': 'ファイルの解析中にエラーが発生しました。',
    'insufficient_data': 'データが不十分です。',
    'gate_error': 'ゲート設定にエラーがあります。',
    'plotting_error': 'プロット作成中にエラーが発生しました。'
}

# 成功メッセージ
SUCCESS_MESSAGES = {
    'file_uploaded': 'ファイルが正常にアップロードされました。',
    'analysis_complete': '解析が完了しました。',
    'gate_created': 'ゲートが作成されました。',
    'data_exported': 'データがエクスポートされました。',
    'settings_saved': '設定が保存されました。'
}

# FCS処理の代替設定
FCS_CONFIG = {
    'use_fallback_parser': True,  # FCSライブラリがない場合の代替処理
    'fallback_formats': ['.csv', '.xlsx'],  # 代替フォーマット
    'mock_fcs_columns': ['FSC-A', 'SSC-A', 'FITC-A', 'PE-A', 'APC-A'],  # テスト用カラム
    'sample_data_size': 10000  # サンプルデータのサイズ
}


CHANNEL_CONFIG = {
    'common_channels': {
        'FSC-A': 'Forward Scatter Area',
        'FSC-H': 'Forward Scatter Height',
        'SSC-A': 'Side Scatter Area',
        'SSC-H': 'Side Scatter Height',
        'FITC-A': 'FITC Area',
        'PE-A': 'PE Area',
        'APC-A': 'APC Area',
        'PerCP-A': 'PerCP Area'
    },
    'fluorescence_channels': ['FITC', 'PE', 'APC', 'PerCP', 'PE-Cy7', 'APC-Cy7'],
    'scatter_channels': ['FSC', 'SSC']
}
