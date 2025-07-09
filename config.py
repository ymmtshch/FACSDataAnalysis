# config.py - Configuration file for FACS Data Analysis App
# Updated to match README.md specifications

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
    'upload_help': 'FCS 2.0/3.0/3.1形式のファイルをアップロードしてください（最大100MB）',
    'supported_formats': ['FCS 2.0', 'FCS 3.0', 'FCS 3.1']
}

# FCS library configuration - prioritized according to README.md
FCS_LIBRARY_CONFIG = {
    'priority_order': ['fcsparser', 'flowio', 'flowkit'],
    'fcsparser': {
        'description': '長期間にわたり安定した実績',
        'features': ['豊富なメタデータ解析機能', '標準的なFCS形式に最適化'],
        'preferred': True
    },
    'flowio': {
        'description': '高速なデータ処理性能',
        'features': ['NumPy 2.0完全対応', 'モダンなデータ処理アーキテクチャ'],
        'fallback_for': 'fcsparser'
    },
    'flowkit': {
        'description': '高度な解析機能を提供',
        'features': ['自動的なデータフレーム変換'],
        'fallback': True
    }
}

# Plotting configuration
PLOT_CONFIG = {
    'default_bins': 100,
    'max_bins': 500,
    'default_alpha': 0.7,
    'colormap_options': ['viridis', 'plasma', 'inferno', 'magma', 'cividis'],
    'default_colormap': 'viridis',
    'figure_size': (800, 600),
    'scatter_size': 2,
    'plot_types': ['scatter', 'density', 'histogram'],
    'density_bins': 50
}

# Gating configuration - comprehensive according to README.md
GATING_CONFIG = {
    'gate_types': {
        'rectangular': {
            'label': '矩形ゲート',
            'description': '2次元での矩形領域選択',
            'parameters': ['x_min', 'x_max', 'y_min', 'y_max']
        },
        'polygon': {
            'label': 'ポリゴンゲート',
            'description': '任意の多角形領域での選択',
            'parameters': ['coordinates']
        },
        'ellipse': {
            'label': '楕円ゲート',
            'description': '楕円形領域での選択',
            'parameters': ['center_x', 'center_y', 'width', 'height']
        },
        'threshold': {
            'label': '閾値ゲート',
            'description': '単一パラメータでの閾値設定',
            'parameters': ['threshold', 'direction'],
            'directions': ['以上', '以下', 'より大きい', 'より小さい']
        }
    },
    'gate_colors': ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'brown'],
    'default_gate_color': 'red',
    'gate_line_width': 2,
    'gate_alpha': 0.3,
    'min_gate_points': 3,
    'max_gates': 10
}

# Data processing configuration - enhanced according to README.md
DATA_CONFIG = {
    'max_events_display': 100000,  # Maximum events to display
    'min_events_display': 1000,   # Minimum events to display
    'max_events_range': (1000, 100000),
    'default_max_events': 50000,
    'subsample_for_plot': True,
    'default_subsample_size': 10000,
    'transform_methods': {
        'none': {'label': 'なし', 'function': 'linear'},
        'log10': {'label': 'Log10', 'function': 'log'},
        'asinh': {'label': 'Asinh', 'function': 'asinh'},
        'biexponential': {'label': 'Biexponential', 'function': 'biexponential'}
    },
    'channel_naming_priority': ['$PnN', '$PnS', 'default'],
    'metadata_keys': {
        'standard': ['$TOT', '$PAR', '$DATE', '$BTIM', '$ETIM', '$CYT', '$CYTNUM'],
        'flowkit_compatible': ['tot', 'par', 'date', 'btim', 'etim', 'cyt', 'cytnum']
    }
}

# Page-specific configuration - matching README.md structure
PAGE_CONFIG = {
    'main_app': {
        'tabs': ['📊 基本情報', '📈 可視化', '🎯 ゲーティング', '📋 統計解析'],
        'use_fcs_processor': True,
        'export_filenames': {
            'statistics': 'facs_statistics.csv',
            'raw_data': 'facs_raw_data.csv'
        },
        'gating_type': 'basic_threshold'
    },
    'basic_analysis': {
        'use_fcsparser_direct': True,
        'auto_library_selection': True,
        'show_debug_info': True,
        'dynamic_filename': True,
        'export_format': '{filename}_stats.csv',
        'data_export_format': '{filename}_data.csv',
        'enhanced_features': [
            'auto_library_detection',
            'debug_information',
            'enhanced_metadata',
            'individual_transform_settings',
            'realtime_statistics',
            'improved_export'
        ]
    },
    'advanced_gating': {
        'gate_types': ['rectangular', 'polygon', 'ellipse', 'threshold'],
        'visualization_type': 'density_plot',
        'realtime_display': True,
        'multi_gate_support': True,
        'comprehensive_statistics': True,
        'export_gate_data': True
    }
}

# Statistics configuration - comprehensive according to README.md
STATS_CONFIG = {
    'basic_statistics': ['count', 'mean', 'median', 'std', 'min', 'max'],
    'percentiles': [5, 25, 50, 75, 95],
    'correlation_methods': ['pearson', 'spearman'],
    'density_estimation': 'gaussian_kde',
    'gate_statistics': {
        'event_count': True,
        'percentage': True,
        'channel_stats': True,
        'comparison': True
    }
}

# Export configuration - enhanced according to README.md
EXPORT_CONFIG = {
    'csv_separator': ',',
    'include_metadata': True,
    'include_statistics': True,
    'timestamp_format': '%Y%m%d_%H%M%S',
    'file_prefix': 'facs_analysis_',
    'auto_naming': True,
    'two_stage_export': True,  # Button click -> Download button display
    'export_types': {
        'statistics': 'Statistics CSV',
        'display_data': 'Display Data CSV',
        'gate_data': 'Gate Data CSV'
    }
}

# Error messages (Japanese) - comprehensive according to README.md
ERROR_MESSAGES = {
    'file_too_large': f'ファイルサイズが{UPLOAD_CONFIG["max_file_size"]}MBを超えています。',
    'invalid_file_format': 'サポートされていないファイル形式です。FCSファイルをアップロードしてください。',
    'file_read_error': 'ファイルの読み込みに失敗しました。ファイルが破損している可能性があります。',
    'insufficient_data': 'データが不十分です。より多くのイベントが必要です。',
    'channel_not_found': '指定されたチャンネルが見つかりません。',
    'gating_error': 'ゲーティング処理中にエラーが発生しました。',
    'export_error': 'データのエクスポート中にエラーが発生しました。',
    'plot_error': 'プロットの作成中にエラーが発生しました。',
    'memory_error': 'メモリ不足です。データサイズを小さくしてください。',
    'library_error': 'FCS読み込みライブラリでエラーが発生しました。',
    'numpy_compatibility': 'NumPy 2.0互換性問題が検出されました。代替ライブラリを使用します。',
    'transform_error': 'データ変換中にエラーが発生しました。',
    'channel_duplicate': 'チャンネル名が重複しています。自動的にリネームします。',
    'gate_parse_error': 'ゲート座標の解析に失敗しました。',
    'session_error': 'セッション状態でエラーが発生しました。ページを再読み込みしてください。'
}

# Success messages (Japanese)
SUCCESS_MESSAGES = {
    'file_loaded': 'ファイルが正常に読み込まれました。',
    'gate_created': 'ゲートが正常に作成されました。',
    'data_exported': 'データが正常にエクスポートされました。',
    'settings_saved': '設定が保存されました。',
    'library_selected': 'FCS読み込みライブラリが選択されました。',
    'transform_applied': 'データ変換が適用されました。',
    'gate_applied': 'ゲートが適用されました。'
}

# Warning messages (Japanese) - enhanced according to README.md
WARNING_MESSAGES = {
    'large_file': 'ファイルサイズが大きいため、処理に時間がかかる場合があります。',
    'many_events': 'イベント数が多いため、表示をサブサンプリングしています。',
    'no_compensation': '補正行列が適用されていません。',
    'performance_warning': 'パフォーマンスを向上させるため、一部のデータのみ表示しています。',
    'library_fallback': 'fcsparserでエラーが発生したため、flowioを使用します。',
    'numpy_version': 'NumPy 2.0環境でfcsparserが不安定な場合があります。',
    'channel_renamed': 'チャンネル名が重複していたため、自動的にリネームしました。',
    'session_reset': 'ブラウザの再読み込みでセッション状態がリセットされます。',
    'memory_optimization': 'メモリ最適化のため、サンプリングが適用されています。'
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
    'APC-Cy7-A': 'APC-Cy7 Area',
    'BV421-A': 'BV421 Area',
    'BV510-A': 'BV510 Area',
    'AF488-A': 'AF488 Area',
    'AF647-A': 'AF647 Area'
}

# FCSParser specific configuration - updated according to README.md
FCSPARSER_CONFIG = {
    'read_data': True,
    'reformat_meta': True,
    'data_set': 0,
    'channel_naming': '$PnN',
    'apply_compensation': False,
    'transform': 'linear',
    'auto_fallback': True,
    'error_handling': 'strict'
}

# Troubleshooting configuration - based on README.md
TROUBLESHOOTING_CONFIG = {
    'common_issues': {
        'library_selection': {
            'fcsparser_recommended': 'fcsparserが第一優先（標準FCS形式に最適化）',
            'numpy_2_issue': 'NumPy 2.0でnewbyteorderエラー時は自動的にflowioにフォールバック',
            'auto_fallback': 'ライブラリエラー時の自動代替選択機能'
        },
        'file_reading': {
            'file_size_limit': '100MB制限を確認',
            'supported_versions': 'FCS 2.0/3.0/3.1サポート',
            'corruption_check': '他のソフトウェアでファイルが開けるかテスト'
        },
        'data_processing': {
            'array_conversion': 'FlowIOでのarray.array→NumPy配列変換エラー',
            'channel_duplicate': '自動リネーム機能による解決',
            'metadata_format': '異なるライブラリ間でのキー名の違いに自動対応'
        },
        'performance': {
            'memory_insufficient': '最大イベント数を減らす、サンプリング数を調整',
            'transform_error': '変換方法を"なし"に変更してテスト',
            'display_issues': 'ブラウザのJavaScript有効化を確認'
        }
    },
    'debug_features': {
        'sidebar_debug': 'データ形状、変換プロセスの詳細表示',
        'channel_verification': '読み込まれたチャンネル名の詳細確認',
        'metadata_validation': '全メタデータ項目の確認機能',
        'library_info': '使用中のライブラリの表示'
    }
}

# System limitations - based on README.md
SYSTEM_LIMITATIONS = {
    'library_dependency': 'fcsparser推奨だが、NumPy 2.0環境ではflowioが安定',
    'numpy_compatibility': 'fcsparserはNumPy 2.0で動作不安定な場合があり、自動フォールバック機能で対応',
    'special_characters': '特殊文字を含むチャンネル名で表示問題の可能性',
    'memory_usage': '大容量FCSファイルではメモリ制限に注意',
    'concurrent_users': 'Streamlit Cloudでは同時アクセス数に制限',
    'session_persistence': 'ブラウザセッション終了でデータは消失',
    'gating_functionality': '高度なゲーティング機能は完全実装済み。メインアプリでは基本機能のみ提供'
}

# UI Theme configuration - based on README.md
UI_THEME_CONFIG = {
    'primary_color': '#FF6B35',  # Main color (Orange)
    'background_color': '#FFFFFF',
    'secondary_background_color': '#F0F2F6',
    'text_color': '#262730',
    'font_family': 'sans serif',
    'wide_layout': True,
    'expanded_sidebar': True,
    'japanese_interface': True,
    'responsive_design': True
}

def get_fcs_library_priority():
    """Get FCS library priority order according to README.md"""
    return FCS_LIBRARY_CONFIG['priority_order']

def get_library_info(library_name):
    """Get information about a specific FCS library"""
    return FCS_LIBRARY_CONFIG.get(library_name, {})

def get_gate_types():
    """Get available gate types with descriptions"""
    return GATING_CONFIG['gate_types']

def get_gate_type_info(gate_type):
    """Get information about a specific gate type"""
    return GATING_CONFIG['gate_types'].get(gate_type, {})

def get_transform_methods():
    """Get available transform methods with labels"""
    return DATA_CONFIG['transform_methods']

def get_transform_label(transform_key):
    """Get Japanese label for transform method"""
    return DATA_CONFIG['transform_methods'].get(transform_key, {}).get('label', transform_key)

def get_page_config(page_name):
    """Get page-specific configuration"""
    return PAGE_CONFIG.get(page_name, {})

def get_troubleshooting_info(issue_category):
    """Get troubleshooting information for specific issue category"""
    return TROUBLESHOOTING_CONFIG['common_issues'].get(issue_category, {})

def get_debug_features():
    """Get available debug features"""
    return TROUBLESHOOTING_CONFIG['debug_features']

def get_system_limitations():
    """Get system limitations information"""
    return SYSTEM_LIMITATIONS

def validate_max_events(num_events):
    """Validate maximum events setting"""
    min_events, max_events = DATA_CONFIG['max_events_range']
    return max(min_events, min(num_events, max_events))

def get_export_filename(base_filename, export_type):
    """Generate export filename according to README.md specifications"""
    if export_type == 'statistics':
        return f"{base_filename}_stats.csv"
    elif export_type == 'display_data':
        return f"{base_filename}_data.csv"
    elif export_type == 'gate_data':
        return f"{base_filename}_gate_data.csv"
    else:
        return f"{base_filename}_export.csv"

def get_metadata_keys():
    """Get standard and FlowKit-compatible metadata keys"""
    return DATA_CONFIG['metadata_keys']

def get_channel_naming_priority():
    """Get channel naming priority order"""
    return DATA_CONFIG['channel_naming_priority']

# Legacy Config class for backward compatibility
class Config:
    """Configuration class for backward compatibility with existing code"""
    
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
    DEFAULT_SUBSAMPLE_SIZE = DATA_CONFIG['default_subsample_size']
    SUBSAMPLE_FOR_PLOT = DATA_CONFIG['subsample_for_plot']
    
    # Channel mappings
    CHANNEL_MAPPINGS = CHANNEL_MAPPINGS
    
    # New settings according to README.md
    FCS_LIBRARY_PRIORITY = FCS_LIBRARY_CONFIG['priority_order']
    GATE_TYPES = GATING_CONFIG['gate_types']
    TRANSFORM_METHODS = DATA_CONFIG['transform_methods']
    PAGE_CONFIG = PAGE_CONFIG
    TROUBLESHOOTING_CONFIG = TROUBLESHOOTING_CONFIG
    SYSTEM_LIMITATIONS = SYSTEM_LIMITATIONS
    
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
        return list(DATA_CONFIG['transform_methods'].keys())
    
    @classmethod
    def get_colormap_options(cls):
        """Get available colormap options"""
        return cls.COLORMAP_OPTIONS
    
    @classmethod
    def get_fcs_library_priority(cls):
        """Get FCS library priority order"""
        return cls.FCS_LIBRARY_PRIORITY
    
    @classmethod
    def get_gate_type_info(cls, gate_type):
        """Get gate type information"""
        return cls.GATE_TYPES.get(gate_type, {})
    
    @classmethod
    def get_page_config(cls, page_name):
        """Get page-specific configuration"""
        return cls.PAGE_CONFIG.get(page_name, {})

def get_config_dict():
    """Return all configuration as a dictionary"""
    return {
        'app': APP_CONFIG,
        'upload': UPLOAD_CONFIG,
        'fcs_library': FCS_LIBRARY_CONFIG,
        'plot': PLOT_CONFIG,
        'gating': GATING_CONFIG,
        'data': DATA_CONFIG,
        'page': PAGE_CONFIG,
        'stats': STATS_CONFIG,
        'export': EXPORT_CONFIG,
        'fcsparser': FCSPARSER_CONFIG,
        'troubleshooting': TROUBLESHOOTING_CONFIG,
        'limitations': SYSTEM_LIMITATIONS,
        'ui_theme': UI_THEME_CONFIG
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

def get_ui_theme():
    """Get UI theme configuration"""
    return UI_THEME_CONFIG

def get_supported_fcs_formats():
    """Get supported FCS file formats"""
    return UPLOAD_CONFIG['supported_formats']

def is_debug_mode():
    """Check if debug mode is enabled"""
    return PAGE_CONFIG.get('basic_analysis', {}).get('show_debug_info', False)

def get_enhanced_features(page_name):
    """Get enhanced features for specific page"""
    return PAGE_CONFIG.get(page_name, {}).get('enhanced_features', [])

def get_gate_statistics_config():
    """Get gate statistics configuration"""
    return STATS_CONFIG['gate_statistics']

def get_export_types():
    """Get available export types"""
    return EXPORT_CONFIG['export_types']

def should_use_two_stage_export():
    """Check if two-stage export should be used"""
    return EXPORT_CONFIG['two_stage_export']

def get_auto_naming_enabled():
    """Check if auto naming is enabled for exports"""
    return EXPORT_CONFIG['auto_naming']
