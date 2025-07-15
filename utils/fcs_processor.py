import pandas as pd
import numpy as np
import io
import streamlit as st
from typing import Tuple, Dict, Any, Optional

class FCSProcessor:
    """FCSファイル処理クラス - 簡素化版"""
    
    def __init__(self, file_data: bytes, filename: str):
        self.file_data = file_data
        self.filename = filename
        self.fcs_data = None
        self.metadata = None
        self.used_library = None
        
    def load_fcs_file(self) -> Tuple[Optional[pd.DataFrame], Optional[Dict], Optional[str]]:
        """FCSファイルを読み込む（fcsparserのみ使用）"""
        try:
            import fcsparser
            
            # バイトデータをファイルライクオブジェクトに変換
            file_like = io.BytesIO(self.file_data)
            
            # ファイルポインタを先頭に設定
            file_like.seek(0)
            
            # fcsparserでパース
            meta, data = fcsparser.parse(file_like, reformat_meta=True)
            
            # チャンネル名の重複処理
            if isinstance(data, pd.DataFrame):
                data.columns = self._handle_duplicate_channels(list(data.columns))
            
            self.fcs_data = data
            self.metadata = meta
            self.used_library = 'fcsparser'
            
            return data, meta, 'fcsparser'
            
        except ImportError:
            st.error("fcsparserライブラリがインストールされていません。")
            return None, None, None
            
        except Exception as e:
            st.error(f"FCSファイルの読み込みに失敗しました: {str(e)}")
            return None, None, None
    
    def _handle_duplicate_channels(self, channel_names: list) -> list:
        """重複するチャンネル名を処理"""
        seen = {}
        result = []
        
        for name in channel_names:
            if name in seen:
                seen[name] += 1
                result.append(f"{name}_{seen[name]}")
            else:
                seen[name] = 1
                result.append(name)
        
        return result
    
    def get_file_info(self) -> Dict[str, Any]:
        """ファイル基本情報を取得"""
        if self.metadata is None:
            return {}
        
        info = {
            'total_events': self.metadata.get('$TOT', 'N/A'),
            'parameters': self.metadata.get('$PAR', 'N/A'),
            'date': self.metadata.get('$DATE', 'N/A'),
            'begin_time': self.metadata.get('$BTIM', 'N/A'),
            'end_time': self.metadata.get('$ETIM', 'N/A'),
            'cytometer': self.metadata.get('$CYT', 'N/A'),
            'experiment_name': self.metadata.get('EXPERIMENT NAME', 'N/A'),
            'sample_id': self.metadata.get('SAMPLE ID', 'N/A'),
            'operator': self.metadata.get('$OP', 'N/A'),
            'software': self.metadata.get('$SRC', 'N/A')
        }
        
        return info
    
    def get_channel_info(self) -> Dict[str, Any]:
        """チャンネル詳細情報を取得"""
        if self.metadata is None or self.fcs_data is None:
            return {}
        
        channel_info = {}
        
        for i, col_name in enumerate(self.fcs_data.columns):
            param_num = i + 1
            
            # チャンネル名
            name = self.metadata.get(f'$P{param_num}N', col_name)
            short_name = self.metadata.get(f'$P{param_num}S', '')
            
            # 表示名を決定
            if name and short_name and name != short_name:
                display_name = f"{name} ({short_name})"
            elif name:
                display_name = name
            else:
                display_name = col_name
            
            channel_info[f'P{param_num}'] = {
                'name': display_name,
                'range': self.metadata.get(f'$P{param_num}R', 'N/A'),
                'bits': self.metadata.get(f'$P{param_num}B', 'N/A')
            }
        
        return channel_info
    
    def get_basic_stats(self) -> Dict[str, Any]:
        """基本統計情報を取得"""
        if self.fcs_data is None:
            return {}
        
        stats = {}
        numeric_columns = self.fcs_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            try:
                col_data = self.fcs_data[col].dropna()
                if len(col_data) > 0:
                    stats[col] = {
                        'mean': float(col_data.mean()),
                        'median': float(col_data.median()),
                        'std': float(col_data.std()),
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'count': int(len(col_data))
                    }
            except:
                continue
        
        return stats
    
    def preprocess_data(self, data: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        """データの前処理"""
        if data is None:
            return pd.DataFrame()
        
        # 数値データのみを保持してNaN値を削除
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        processed_data = data[numeric_columns].copy()
        processed_data = processed_data.dropna()
        
        return processed_data
    
    def apply_transform(self, data: pd.Series, transform_type: str) -> pd.Series:
        """データ変換を適用"""
        if transform_type == "なし":
            return data
        
        try:
            if transform_type == "Log10":
                # 負の値やゼロを小さな正の値に置換
                data_transformed = data.copy()
                data_transformed[data_transformed <= 0] = 1e-6
                return np.log10(data_transformed)
            
            elif transform_type == "Asinh":
                # Asinh変換（co-factor = 150）
                return np.arcsinh(data / 150)
            
            else:
                return data
                
        except Exception as e:
            st.warning(f"データ変換に失敗しました: {str(e)}")
            return data
    
    def export_data(self, data: pd.DataFrame, data_type: str = "data") -> str:
        """データをCSV形式でエクスポート"""
        if data.empty:
            return ""
        
        try:
            # ファイル名を生成
            base_name = self.filename.rsplit('.', 1)[0] if '.' in self.filename else self.filename
            
            if data_type == "stats":
                filename = f"{base_name}_stats.csv"
            else:
                filename = f"{base_name}_data.csv"
            
            # CSVに変換
            csv_data = data.to_csv(index=False, encoding='utf-8')
            return csv_data
            
        except Exception as e:
            st.error(f"データエクスポートエラー: {str(e)}")
            return ""
    
    def get_all_metadata(self, limit: int = 20) -> Dict[str, Any]:
        """全メタデータを取得"""
        if self.metadata is None:
            return {}
        
        items = list(self.metadata.items())
        if limit and limit > 0:
            items = items[:limit]
        return dict(items)
    
    def get_debug_info(self) -> Dict[str, Any]:
        """デバッグ情報を取得"""
        debug_info = {
            'filename': self.filename,
            'data_shape': self.fcs_data.shape if self.fcs_data is not None else None,
            'metadata_keys_count': len(self.metadata) if self.metadata else 0,
            'column_names': list(self.fcs_data.columns) if self.fcs_data is not None else []
        }
        
        return debug_info


def load_and_process_fcs(uploaded_file, transformation: str = "なし", max_events: int = 10000) -> Tuple[Optional['FCSProcessor'], Optional[pd.DataFrame], Optional[Dict], Optional[str]]:
    """FCSファイルを読み込んで処理する主要関数"""
    
    if uploaded_file is None:
        return None, None, None, "ファイルがアップロードされていません。"
    
    try:
        # ファイルを先頭に戻す
        uploaded_file.seek(0)
        
        # ファイルデータを読み込み
        file_data = uploaded_file.read()
        filename = uploaded_file.name
        
        # ファイルが空でないかチェック
        if len(file_data) == 0:
            return None, None, None, "アップロードされたファイルが空です。"
        
        # FCSProcessorを初期化
        processor = FCSProcessor(file_data, filename)
        
        # FCSファイルを読み込み
        data, metadata, used_library = processor.load_fcs_file()
        
        if data is None:
            return None, None, None, "FCSファイルの読み込みに失敗しました。"
        
        # データの前処理
        processed_data = processor.preprocess_data(data, metadata)
        
        if len(processed_data) == 0:
            return None, None, None, "有効な数値データが見つかりません。"
        
        # 最大イベント数の制限
        if len(processed_data) > max_events:
            processed_data = processed_data.sample(n=max_events, random_state=42)
            st.info(f"パフォーマンス最適化のため、{max_events:,}イベントにサンプリングしました。")
        
        # データ変換
        if transformation != "なし":
            for col in processed_data.columns:
                if pd.api.types.is_numeric_dtype(processed_data[col]):
                    processed_data[col] = processor.apply_transform(processed_data[col], transformation)
        
        # 成功メッセージ
        st.success(f"FCSファイルが正常に読み込まれました（{used_library}を使用）")
        
        return processor, processed_data, metadata, None
        
    except Exception as e:
        error_msg = f"ファイル処理中にエラーが発生しました: {str(e)}"
        st.error(error_msg)
        return None, None, None, error_msg
