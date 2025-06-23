import pandas as pd
import numpy as np
import io
import streamlit as st
from typing import Tuple, Dict, Any, Optional

class FCSProcessor:
    """FCSファイル処理クラス - FlowKit除去版"""
    
    def __init__(self, file_data: bytes, filename: str):
        self.file_data = file_data
        self.filename = filename
        self.fcs_data = None
        self.metadata = None
        self.used_library = None
        
    def load_fcs_file(self) -> Tuple[Optional[pd.DataFrame], Optional[Dict], Optional[str]]:
        """
        FCSファイルを読み込む（FlowKit除去、flowio → fcsparser の順で試行）
        
        Returns:
            Tuple[DataFrame, metadata, used_library]: データ、メタデータ、使用ライブラリ名
        """
        # FlowIOを最初に試行
        try:
            import flowio
            fcs = flowio.FlowData(self.file_data)
            
            # イベントデータを取得
            events = fcs.events
            if hasattr(events, 'dtype') and events.dtype == 'O':
                # array.arrayの場合はnumpy配列に変換
                if hasattr(events[0], 'tolist'):
                    events = np.array([evt.tolist() if hasattr(evt, 'tolist') else list(evt) for evt in events])
                else:
                    events = np.array([list(evt) for evt in events])
            
            # チャンネル名を取得
            channel_names = []
            for i in range(1, len(fcs.channels) + 1):
                # $PnN (チャンネル名) → $PnS (ショート名) の順で取得
                channel_key = f'$P{i}N'
                short_key = f'$P{i}S'
                
                if channel_key in fcs.text:
                    name = fcs.text[channel_key]
                elif short_key in fcs.text:
                    name = fcs.text[short_key]
                else:
                    name = f'Channel_{i}'
                
                channel_names.append(name)
            
            # 重複チャンネル名の処理
            channel_names = self._handle_duplicate_channels(channel_names)
            
            # DataFrameを作成
            df = pd.DataFrame(events, columns=channel_names)
            
            # メタデータを辞書形式で取得
            metadata = dict(fcs.text)
            
            self.fcs_data = df
            self.metadata = metadata
            self.used_library = 'flowio'
            
            return df, metadata, 'flowio'
            
        except Exception as e:
            st.warning(f"FlowIOでの読み込みに失敗: {str(e)}")
        
        # fcsparserをフォールバックとして試行
        try:
            import fcsparser
            
            # バイトデータをファイルライクオブジェクトに変換
            file_like = io.BytesIO(self.file_data)
            meta, data = fcsparser.parse(file_like, reformat_meta=True)
            
            # チャンネル名の重複処理
            if isinstance(data, pd.DataFrame):
                data.columns = self._handle_duplicate_channels(list(data.columns))
            
            self.fcs_data = data
            self.metadata = meta
            self.used_library = 'fcsparser'
            
            return data, meta, 'fcsparser'
            
        except Exception as e:
            st.error(f"fcsparserでの読み込みも失敗: {str(e)}")
            if "newbyteorder" in str(e):
                st.error("NumPy 2.0互換性エラーが発生しました。flowioの使用を推奨します。")
        
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
        
        info = {}
        
        # 総イベント数
        tot_keys = ['$TOT', 'tot', 'TOTAL', 'total']
        for key in tot_keys:
            if key in self.metadata:
                info['total_events'] = int(self.metadata[key])
                break
        
        # パラメータ数
        par_keys = ['$PAR', 'par', 'PARAMETERS', 'parameters']
        for key in par_keys:
            if key in self.metadata:
                info['parameters'] = int(self.metadata[key])
                break
        
        # 取得日時
        date_keys = ['$DATE', 'date', 'DATE']
        for key in date_keys:
            if key in self.metadata:
                info['date'] = self.metadata[key]
                break
        
        # 取得開始時刻
        btim_keys = ['$BTIM', 'btim', 'BTIM']
        for key in btim_keys:
            if key in self.metadata:
                info['begin_time'] = self.metadata[key]
                break
        
        # 取得終了時刻
        etim_keys = ['$ETIM', 'etim', 'ETIM']
        for key in etim_keys:
            if key in self.metadata:
                info['end_time'] = self.metadata[key]
                break
        
        # 使用機器
        cyt_keys = ['$CYT', 'cyt', 'CYTOMETER']
        for key in cyt_keys:
            if key in self.metadata:
                info['cytometer'] = self.metadata[key]
                break
        
        # 機器番号
        cytnum_keys = ['$CYTNUM', 'cytnum', 'CYTNUM']
        for key in cytnum_keys:
            if key in self.metadata:
                info['cytometer_number'] = self.metadata[key]
                break
        
        # 実験名
        exp_keys = ['EXPERIMENT NAME', '$EXP', 'exp', 'EXPERIMENT']
        for key in exp_keys:
            if key in self.metadata:
                info['experiment_name'] = self.metadata[key]
                break
        
        # サンプルID
        sample_keys = ['SAMPLE ID', '$SMNO', 'smno', 'SAMPLE']
        for key in sample_keys:
            if key in self.metadata:
                info['sample_id'] = self.metadata[key]
                break
        
        # オペレーター
        op_keys = ['$OP', 'op', 'OPERATOR']
        for key in op_keys:
            if key in self.metadata:
                info['operator'] = self.metadata[key]
                break
        
        return info
    
    def get_channel_info(self) -> Dict[str, Any]:
        """チャンネル詳細情報を取得"""
        if self.metadata is None:
            return {}
        
        channel_info = {}
        
        # パラメータ数を取得
        par_keys = ['$PAR', 'par', 'PARAMETERS']
        num_params = None
        for key in par_keys:
            if key in self.metadata:
                num_params = int(self.metadata[key])
                break
        
        if num_params is None:
            return {}
        
        for i in range(1, num_params + 1):
            param_info = {}
            
            # チャンネル名
            name_key = f'$P{i}N'
            if name_key in self.metadata:
                param_info['name'] = self.metadata[name_key]
            
            # ショート名
            short_key = f'$P{i}S'
            if short_key in self.metadata:
                param_info['short_name'] = self.metadata[short_key]
            
            # レンジ
            range_key = f'$P{i}R'
            if range_key in self.metadata:
                param_info['range'] = self.metadata[range_key]
            
            # ビット数
            bits_key = f'$P{i}B'
            if bits_key in self.metadata:
                param_info['bits'] = self.metadata[bits_key]
            
            # ゲイン
            gain_key = f'$P{i}G'
            if gain_key in self.metadata:
                param_info['gain'] = self.metadata[gain_key]
            
            # 電圧
            voltage_key = f'$P{i}V'
            if voltage_key in self.metadata:
                param_info['voltage'] = self.metadata[voltage_key]
            
            channel_info[f'P{i}'] = param_info
        
        return channel_info
    
    def get_basic_stats(self) -> Dict[str, Any]:
        """基本統計情報を取得"""
        if self.fcs_data is None:
            return {}
        
        stats = {}
        
        # 数値列のみを対象とする
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
            except Exception as e:
                # 統計計算に失敗した場合はスキップ
                continue
        
        return stats
    
    def preprocess_data(self, data: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        """データの前処理"""
        if data is None:
            return pd.DataFrame()
        
        # 数値データのみを保持
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        processed_data = data[numeric_columns].copy()
        
        # NaN値を削除
        processed_data = processed_data.dropna()
        
        return processed_data
    
    def apply_transform(self, data: pd.Series, transform_type: str) -> pd.Series:
        """データ変換を適用"""
        if transform_type == "なし" or transform_type == "None":
            return data
        
        try:
            if transform_type == "Log10":
                # 負の値やゼロをわずかに正の値に置換
                data_transformed = data.copy()
                data_transformed[data_transformed <= 0] = 1e-6
                return np.log10(data_transformed)
            
            elif transform_type == "Asinh":
                return np.arcsinh(data / 1000)  # スケール調整
            
            elif transform_type == "Biexponential":
                # 簡易的なbiexponential変換
                return np.sign(data) * np.log10(1 + np.abs(data))
            
            else:
                return data
                
        except Exception as e:
            st.warning(f"データ変換に失敗しました ({transform_type}): {str(e)}")
            return data
    
    def export_data(self, data: pd.DataFrame, data_type: str = "data") -> str:
        """データをCSV形式でエクスポート"""
        if data.empty:
            return ""
        
        try:
            # ファイル名から拡張子を除去
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


def load_and_process_fcs(uploaded_file, transformation: str = "なし", max_events: int = 10000) -> Tuple[Optional['FCSProcessor'], Optional[pd.DataFrame], Optional[Dict]]:
    """
    FCSファイルを読み込んで処理する主要関数（FlowKit除去版）
    
    Args:
        uploaded_file: Streamlitのアップロードファイル
        transformation: データ変換タイプ
        max_events: 最大イベント数
    
    Returns:
        Tuple[FCSProcessor, DataFrame, metadata]: プロセッサ、データ、メタデータ
    """
    if uploaded_file is None:
        return None, None, None
    
    try:
        # ファイルデータを読み込み
        file_data = uploaded_file.read()
        filename = uploaded_file.name
        
        # FCSProcessorを初期化
        processor = FCSProcessor(file_data, filename)
        
        # FCSファイルを読み込み
        data, metadata, used_library = processor.load_fcs_file()
        
        if data is None:
            st.error("FCSファイルの読み込みに失敗しました。")
            return None, None, None
        
        # データの前処理
        processed_data = processor.preprocess_data(data, metadata)
        
        # イベント数制限
        if len(processed_data) > max_events:
            processed_data = processed_data.sample(n=max_events, random_state=42)
        
        # データ変換（全列に適用）
        if transformation != "なし":
            for col in processed_data.columns:
                processed_data[col] = processor.apply_transform(processed_data[col], transformation)
        
        st.success(f"FCSファイルが正常に読み込まれました（使用ライブラリ: {used_library}）")
        
        return processor, processed_data, metadata
        
    except Exception as e:
        st.error(f"ファイル処理エラー: {str(e)}")
        return None, None, None
