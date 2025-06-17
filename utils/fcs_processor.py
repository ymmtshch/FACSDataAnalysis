"""
軽量版FCS処理ユーティリティ
Streamlit Cloud対応のため、重い依存関係を避けて実装
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, Dict, Any, Optional

try:
    import fcsparser
    FCS_AVAILABLE = True
except ImportError:
    FCS_AVAILABLE = False

class FCSProcessor:
    """軽量版FCS処理クラス"""
    
    def __init__(self):
        self.meta = None
        self.data = None
        self.df = None
    
    def load_fcs_file(self, file_content: bytes) -> Tuple[bool, str]:
        """
        FCSファイルを読み込む
        
        Args:
            file_content: FCSファイルのバイト内容
            
        Returns:
            成功フラグとメッセージ
        """
        if not FCS_AVAILABLE:
            return False, "fcsparserライブラリが利用できません"
        
        try:
            # fcsparserで解析
            self.meta, self.data = fcsparser.parse(
                file_content, 
                meta_data_only=False, 
                reformat_meta=True
            )
            
            # DataFrameに変換
            self.df = pd.DataFrame(self.data)
            
            return True, f"正常に読み込みました。イベント数: {len(self.df):,}"
            
        except Exception as e:
            return False, f"ファイル読み込みエラー: {str(e)}"
    
    def get_parameters(self) -> list:
        """利用可能なパラメータ一覧を取得"""
        if self.df is not None:
            return list(self.df.columns)
        return []
    
    def get_data_info(self) -> Dict[str, Any]:
        """データの基本情報を取得"""
        if self.df is None:
            return {}
        
        return {
            'event_count': len(self.df),
            'parameter_count': len(self.df.columns),
            'parameters': list(self.df.columns),
            'memory_usage': self.df.memory_usage(deep=True).sum()
        }
    
    def apply_transformation(self, parameter: str, transform_type: str) -> pd.Series:
        """
        データ変換を適用
        
        Args:
            parameter: 対象パラメータ
            transform_type: 変換タイプ ('linear', 'log', 'asinh')
            
        Returns:
            変換後のデータ
        """
        if self.df is None or parameter not in self.df.columns:
            return pd.Series()
        
        data = self.df[parameter].copy()
        
        if transform_type == 'log':
            # 負の値を避けるため最小値を1に設定
            data = data.clip(lower=1)
            return np.log10(data)
        elif transform_type == 'asinh':
            # asinh変換（フローサイトメトリーで一般的）
            return np.arcsinh(data / 150)  # 150は一般的なコファクター
        else:  # linear
            return data
    
    def calculate_statistics(self, parameter: str) -> Dict[str, float]:
        """パラメータの統計情報を計算"""
        if self.df is None or parameter not in self.df.columns:
            return {}
        
        data = self.df[parameter]
        
        return {
            'count': len(data),
            'mean': float(data.mean()),
            'median': float(data.median()),
            'std': float(data.std()),
            'min': float(data.min()),
            'max': float(data.max()),
            'q25': float(data.quantile(0.25)),
            'q75': float(data.quantile(0.75))
        }
    
    def sample_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """データをサンプリング（可視化用）"""
        if self.df is None:
            return pd.DataFrame()
        
        if len(self.df) <= n_samples:
            return self.df.copy()
        
        return self.df.sample(n=n_samples, random_state=42)
    
    def simple_gating(self, x_param: str, y_param: str, 
                     x_range: Tuple[float, float], 
                     y_range: Tuple[float, float]) -> pd.DataFrame:
        """
        シンプルな矩形ゲーティング
        
        Args:
            x_param: X軸パラメータ
            y_param: Y軸パラメータ
            x_range: X軸の範囲 (min, max)
            y_range: Y軸の範囲 (min, max)
            
        Returns:
            ゲート内のデータ
        """
        if self.df is None:
            return pd.DataFrame()
        
        mask = (
            (self.df[x_param] >= x_range[0]) & 
            (self.df[x_param] <= x_range[1]) &
            (self.df[y_param] >= y_range[0]) & 
            (self.df[y_param] <= y_range[1])
        )
        
        return self.df[mask].copy()
    
    def export_csv(self, filename_prefix: str = "fcs_data") -> str:
        """CSV形式でデータをエクスポート"""
        if self.df is None:
            return ""
        
        return self.df.to_csv(index=False)

# グローバル関数（後方互換性のため）
def load_fcs_data(file_content: bytes) -> Tuple[Optional[pd.DataFrame], str]:
    """FCSファイルを読み込んでDataFrameを返す"""
    processor = FCSProcessor()
    success, message = processor.load_fcs_file(file_content)
    
    if success:
        return processor.df, message
    else:
        return None, message
