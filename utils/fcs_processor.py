"""
FCS (Flow Cytometry Standard) データ処理ユーティリティ
フローサイトメトリーデータの読み込み、前処理、統計計算を行う
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import io

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FCSProcessor:
    """FCSデータの処理を行うメインクラス"""
    
    def __init__(self):
        self.data = None
        self.metadata = {}
        self.channels = []
        self.original_data = None
        
    def load_fcs_file(self, file_path_or_buffer: Union[str, io.BytesIO]) -> bool:
        """
        FCSファイルを読み込む
        
        Args:
            file_path_or_buffer: ファイルパスまたはバイトストリーム
            
        Returns:
            bool: 読み込み成功の場合True
        """
        try:
            # FlowCytometryToolsまたはfcspyを使用してFCSファイルを読み込む
            # ここでは簡略化してCSV形式での読み込みも対応
            if isinstance(file_path_or_buffer, str):
                if file_path_or_buffer.endswith('.csv'):
                    self.data = pd.read_csv(file_path_or_buffer)
                else:
                    # 実際のFCSファイル読み込み処理
                    self.data = self._read_fcs_file(file_path_or_buffer)
            else:
                # バイトストリームからの読み込み
                self.data = self._read_fcs_from_buffer(file_path_or_buffer)
            
            if self.data is not None:
                self.original_data = self.data.copy()
                self.channels = list(self.data.columns)
                self._extract_metadata()
                logger.info(f"FCSファイル読み込み完了: {len(self.data)} events, {len(self.channels)} parameters")
                return True
            
        except Exception as e:
            logger.error(f"FCSファイル読み込みエラー: {str(e)}")
            return False
        
        return False
    
    def _read_fcs_file(self, file_path: str) -> pd.DataFrame:
        """
        実際のFCSファイル読み込み処理
        注意: 実際の実装では FlowCytometryTools や fcspy を使用
        """
        try:
            # FlowCytometryToolsを使用する場合の例
            # from FlowCytometryTools import FCMeasurement
            # sample = FCMeasurement(ID='Sample', datafile=file_path)
            # return sample.data
            
            # 仮実装: CSVファイルとして読み込み
            return pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"FCSファイル読み込みエラー: {str(e)}")
            return None
    
    def _read_fcs_from_buffer(self, buffer: io.BytesIO) -> pd.DataFrame:
        """
        バイトストリームからFCSファイルを読み込む
        """
        try:
            # 実際の実装では FlowCytometryTools や fcspy を使用
            # ここでは簡略化してCSVとして処理
            buffer.seek(0)
            return pd.read_csv(buffer)
        except Exception as e:
            logger.error(f"バッファからの読み込みエラー: {str(e)}")
            return None
    
    def _extract_metadata(self):
        """
        メタデータを抽出
        """
        if self.data is not None:
            self.metadata = {
                'total_events': len(self.data),
                'parameters': len(self.channels),
                'channels': self.channels,
                'data_range': {
                    channel: {
                        'min': float(self.data[channel].min()),
                        'max': float(self.data[channel].max()),
                        'mean': float(self.data[channel].mean()),
                        'std': float(self.data[channel].std())
                    } for channel in self.channels
                }
            }
    
    def get_channel_data(self, channel: str) -> Optional[pd.Series]:
        """
        指定チャンネルのデータを取得
        
        Args:
            channel: チャンネル名
            
        Returns:
            pd.Series: チャンネルデータ
        """
        if self.data is not None and channel in self.data.columns:
            return self.data[channel]
        return None
    
    def get_two_channel_data(self, x_channel: str, y_channel: str) -> Optional[pd.DataFrame]:
        """
        2チャンネルのデータを取得
        
        Args:
            x_channel: X軸チャンネル名
            y_channel: Y軸チャンネル名
            
        Returns:
            pd.DataFrame: 2チャンネルデータ
        """
        if (self.data is not None and 
            x_channel in self.data.columns and 
            y_channel in self.data.columns):
            return self.data[[x_channel, y_channel]]
        return None
    
    def apply_transform(self, channels: List[str], transform_type: str = 'log') -> bool:
        """
        データ変換を適用
        
        Args:
            channels: 変換対象チャンネル
            transform_type: 変換タイプ ('log', 'biexp', 'linear')
            
        Returns:
            bool: 変換成功の場合True
        """
        try:
            if self.data is None:
                return False
            
            for channel in channels:
                if channel not in self.data.columns:
                    continue
                
                if transform_type == 'log':
                    # 対数変換（負の値を処理）
                    self.data[channel] = np.log10(self.data[channel] + 1)
                elif transform_type == 'biexp':
                    # Biexponential変換（簡略化版）
                    self.data[channel] = self._biexp_transform(self.data[channel])
                elif transform_type == 'linear':
                    # 線形変換（元データに戻す）
                    if self.original_data is not None:
                        self.data[channel] = self.original_data[channel]
            
            # メタデータを更新
            self._extract_metadata()
            return True
            
        except Exception as e:
            logger.error(f"データ変換エラー: {str(e)}")
            return False
    
    def _biexp_transform(self, data: pd.Series) -> pd.Series:
        """
        Biexponential変換（簡略化版）
        """
        # 実際の実装では FlowCytometryTools の biexponential を使用
        # ここでは簡略化した変換を実装
        return np.sign(data) * np.log10(np.abs(data) + 1)
    
    def calculate_statistics(self, channel: str, 
                           percentiles: List[float] = [5, 25, 50, 75, 95]) -> Dict:
        """
        チャンネルの統計情報を計算
        
        Args:
            channel: チャンネル名
            percentiles: 計算するパーセンタイル
            
        Returns:
            Dict: 統計情報
        """
        if self.data is None or channel not in self.data.columns:
            return {}
        
        data = self.data[channel]
        
        stats = {
            'count': len(data),
            'mean': float(data.mean()),
            'std': float(data.std()),
            'min': float(data.min()),
            'max': float(data.max()),
            'percentiles': {}
        }
        
        # パーセンタイル計算
        for p in percentiles:
            stats['percentiles'][f'{p}%'] = float(data.quantile(p/100))
        
        return stats
    
    def filter_data(self, filters: Dict[str, Dict]) -> pd.DataFrame:
        """
        データフィルタリング
        
        Args:
            filters: フィルタ条件
                例: {'FSC-A': {'min': 1000, 'max': 50000}}
                
        Returns:
            pd.DataFrame: フィルタリング済みデータ
        """
        if self.data is None:
            return pd.DataFrame()
        
        filtered_data = self.data.copy()
        
        for channel, conditions in filters.items():
            if channel not in filtered_data.columns:
                continue
            
            if 'min' in conditions:
                filtered_data = filtered_data[filtered_data[channel] >= conditions['min']]
            if 'max' in conditions:
                filtered_data = filtered_data[filtered_data[channel] <= conditions['max']]
        
        return filtered_data
    
    def create_density_data(self, x_channel: str, y_channel: str, 
                           bins: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        2Dヒストグラム（密度データ）を作成
        
        Args:
            x_channel: X軸チャンネル
            y_channel: Y軸チャンネル
            bins: ビン数
            
        Returns:
            Tuple: (H, xedges, yedges)
        """
        if self.data is None:
            return np.array([]), np.array([]), np.array([])
        
        x_data = self.data[x_channel]
        y_data = self.data[y_channel]
        
        H, xedges, yedges = np.histogram2d(x_data, y_data, bins=bins)
        
        return H, xedges, yedges
    
    def get_data_summary(self) -> Dict:
        """
        データの概要情報を取得
        
        Returns:
            Dict: 概要情報
        """
        if self.data is None:
            return {}
        
        return {
            'total_events': len(self.data),
            'parameters': len(self.channels),
            'channels': self.channels,
            'data_types': {col: str(dtype) for col, dtype in self.data.dtypes.items()},
            'missing_values': self.data.isnull().sum().to_dict(),
            'memory_usage': f"{self.data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
        }
    
    def export_data(self, output_path: str, filtered_data: Optional[pd.DataFrame] = None) -> bool:
        """
        データをCSVファイルとしてエクスポート
        
        Args:
            output_path: 出力ファイルパス
            filtered_data: フィルタリング済みデータ（Noneの場合は全データ）
            
        Returns:
            bool: エクスポート成功の場合True
        """
        try:
            data_to_export = filtered_data if filtered_data is not None else self.data
            
            if data_to_export is None or data_to_export.empty:
                return False
            
            data_to_export.to_csv(output_path, index=False)
            logger.info(f"データエクスポート完了: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"データエクスポートエラー: {str(e)}")
            return False
    
    def reset_data(self):
        """
        データを初期状態にリセット
        """
        if self.original_data is not None:
            self.data = self.original_data.copy()
            self._extract_metadata()
    
    def get_channel_pairs(self) -> List[Tuple[str, str]]:
        """
        一般的なチャンネルペアを取得
        
        Returns:
            List[Tuple[str, str]]: チャンネルペアのリスト
        """
        if not self.channels:
            return []
        
        # 一般的なフローサイトメトリーチャンネルペア
        common_pairs = []
        
        # FSC vs SSC
        fsc_channels = [ch for ch in self.channels if 'FSC' in ch.upper()]
        ssc_channels = [ch for ch in self.channels if 'SSC' in ch.upper()]
        
        if fsc_channels and ssc_channels:
            common_pairs.append((fsc_channels[0], ssc_channels[0]))
        
        # 蛍光チャンネル同士
        fl_channels = [ch for ch in self.channels if any(x in ch.upper() for x in ['FL', 'PE', 'FITC', 'APC'])]
        
        # 最初の数個のペアを追加
        for i in range(min(3, len(fl_channels))):
            for j in range(i+1, min(3, len(fl_channels))):
                common_pairs.append((fl_channels[i], fl_channels[j]))
        
        return common_pairs if common_pairs else [(self.channels[0], self.channels[1])] if len(self.channels) >= 2 else []


def create_sample_data(n_events: int = 10000) -> pd.DataFrame:
    """
    サンプルFCSデータを作成（テスト用）
    
    Args:
        n_events: イベント数
        
    Returns:
        pd.DataFrame: サンプルデータ
    """
    np.random.seed(42)
    
    # 典型的なフローサイトメトリーデータをシミュレート
    data = {
        'FSC-A': np.random.lognormal(mean=4, sigma=0.5, size=n_events),
        'SSC-A': np.random.lognormal(mean=3.5, sigma=0.6, size=n_events),
        'FL1-H': np.random.exponential(scale=1000, size=n_events),
        'FL2-H': np.random.exponential(scale=800, size=n_events),
        'FL3-H': np.random.exponential(scale=1200, size=n_events),
        'PE-A': np.random.lognormal(mean=2.5, sigma=1.2, size=n_events),
        'FITC-A': np.random.lognormal(mean=2.8, sigma=1.0, size=n_events),
    }
    
    return pd.DataFrame(data)


# 使用例
if __name__ == "__main__":
    # サンプルデータでテスト
    processor = FCSProcessor()
    
    # サンプルデータ作成
    sample_data = create_sample_data(5000)
    processor.data = sample_data
    processor.channels = list(sample_data.columns)
    processor._extract_metadata()
    
    # 統計情報計算
    stats = processor.calculate_statistics('FSC-A')
    print("FSC-A統計情報:", stats)
    
    # 2Dヒストグラム作成
    H, xedges, yedges = processor.create_density_data('FSC-A', 'SSC-A')
    print(f"2Dヒストグラム形状: {H.shape}")
    
    # データ概要
    summary = processor.get_data_summary()
    print("データ概要:", summary)
