# __init__.py
"""
FACS Data Analysis Platform

StreamlitベースのFACS（フローサイトメトリー）データ解析アプリケーション
"""

__version__ = "1.0.0"
__author__ = "FACS Analysis Team"
__description__ = "Streamlit-based Flow Cytometry Analysis Tool"

# パッケージレベルのインポート
from .config import (
    APP_CONFIG,
    UI_CONFIG,
    PLOT_CONFIG,
    GATING_CONFIG,
    STATS_CONFIG,
    DATA_CONFIG,
    SESSION_KEYS,
    ERROR_MESSAGES,
    SUCCESS_MESSAGES,
    CHANNEL_CONFIG
)

# バージョン情報をconfig.pyと同期
APP_CONFIG['version'] = __version__
