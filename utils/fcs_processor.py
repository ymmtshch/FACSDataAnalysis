import pandas as pd
import numpy as np
import io
import streamlit as st
from typing import Tuple, Dict, Any, Optional

class FCSProcessor:
    """FCSファイル処理クラス - README.md仕様準拠版"""
    
    def __init__(self, file_data: bytes, filename: str):
        self.file_data = file_data
        self.filename = filename
        self.fcs_data = None
        self.metadata = None
        self.used_library = None
        
    def load_fcs_file(self) -> Tuple[Optional[pd.DataFrame], Optional[Dict], Optional[str]]:
        """
        FCSファイルを読み込む（README.md仕様：fcsparser → flowio → flowkit の順で試行）
        
        Returns:
            Tuple[DataFrame, metadata, used_library]: データ、メタデータ、使用ライブラリ名
        """
        # fcsparserを最初に試行（README.md仕様：推奨・第一優先）
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
            st.warning(f"fcsparserでの読み込みに失敗: {str(e)}")
            if "newbyteorder" in str(e):
                st.warning("NumPy 2.0互換性エラーが検出されました。flowioに自動フォールバックします。")
        
        # flowioを第二優先として試行
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
            
            # チャンネル名を取得（README.md仕様：$PnN → $PnS → デフォルト名の順）
            channel_names = []
            for i in range(1, len(fcs.channels) + 1):
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
            st.warning(f"flowioでの読み込みに失敗: {str(e)}")
        
        # flowkitをフォールバックとして試行（README.md仕様：第三優先）
        try:
            import flowkit
            
            # バイトデータをファイルライクオブジェクトに変換
            file_like = io.BytesIO(self.file_data)
            sample = flowkit.Sample(file_like)
            
            # データを取得
            data = sample.as_dataframe()
            
            # メタデータを取得
            metadata = {}
            if hasattr(sample, 'metadata'):
                metadata = sample.metadata
            
            # チャンネル名の重複処理
            if isinstance(data, pd.DataFrame):
                data.columns = self._handle_duplicate_channels(list(data.columns))
            
            self.fcs_data = data
            self.metadata = metadata
            self.used_library = 'flowkit'
            
            return data, metadata, 'flowkit'
            
        except Exception as e:
            st.error(f"flowkitでの読み込みも失敗: {str(e)}")
        
        st.error("すべてのライブラリでの読み込みに失敗しました。")
        return None, None, None
    
    def _handle_duplicate_channels(self, channel_names: list) -> list:
        """重複するチャンネル名を処理（README.md仕様：_2, _3等の付加）"""
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
        """ファイル基本情報を取得（README.md仕様：標準メタデータ + FlowKit互換）"""
        if self.metadata is None:
            return {}
        
        info = {}
        
        # 総イベント数（README.md仕様：標準メタデータ + FlowKit互換）
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
        
        # ソフトウェア情報（README.md仕様で言及されている項目）
        software_keys = ['$SYS', 'sys', 'SOFTWARE', 'software']
        for key in software_keys:
            if key in self.metadata:
                info['software'] = self.metadata[key]
                break
        
        return info
    
    def get_channel_info(self) -> Dict[str, Any]:
        """チャンネル詳細情報を取得（README.md仕様：各チャンネルの詳細情報と設定値）"""
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
            
            # チャンネル名（README.md仕様：$PnN → $PnS → デフォルト名の順）
            name_key = f'$P{i}N'
            short_key = f'$P{i}S'
            
            if name_key in self.metadata:
                param_info['name'] = self.metadata[name_key]
            elif short_key in self.metadata:
                param_info['name'] = self.metadata[short_key]
            else:
                param_info['name'] = f'Channel_{i}'
            
            # ショート名
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
            
            # 増幅器タイプ
            amp_key = f'$P{i}T'
            if amp_key in self.metadata:
                param_info['amplifier_type'] = self.metadata[amp_key]
            
            channel_info[f'P{i}'] = param_info
        
        return channel_info
    
    def get_basic_stats(self) -> Dict[str, Any]:
        """基本統計情報を取得（README.md仕様：平均値、中央値、標準偏差、最小値、最大値）"""
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
        """データの前処理（README.md仕様：数値データのみ保持、NaN値削除）"""
        if data is None:
            return pd.DataFrame()
        
        # 数値データのみを保持
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        processed_data = data[numeric_columns].copy()
        
        # NaN値を削除
        processed_data = processed_data.dropna()
        
        return processed_data
    
    def apply_transform(self, data: pd.Series, transform_type: str) -> pd.Series:
        """データ変換を適用（README.md仕様：なし、Log10、Asinh、Biexponential変換）"""
        if transform_type == "なし" or transform_type == "None":
            return data
        
        try:
            if transform_type == "Log10":
                # 負の値やゼロをわずかに正の値に置換
                data_transformed = data.copy()
                data_transformed[data_transformed <= 0] = 1e-6
                return np.log10(data_transformed)
            
            elif transform_type == "Asinh":
                # README.md仕様：Asinh変換
                return np.arcsinh(data / 1000)  # スケール調整
            
            elif transform_type == "Biexponential":
                # README.md仕様：Biexponential変換（簡易実装）
                return np.sign(data) * np.log10(1 + np.abs(data))
            
            else:
                return data
                
        except Exception as e:
            st.warning(f"データ変換に失敗しました ({transform_type}): {str(e)}")
            return data
    
    def export_data(self, data: pd.DataFrame, data_type: str = "data") -> str:
        """データをCSV形式でエクスポート（README.md仕様：自動命名）"""
        if data.empty:
            return ""
        
        try:
            # README.md仕様：自動ファイル命名
            base_name = self.filename.rsplit('.', 1)[0] if '.' in self.filename else self.filename
            
            if data_type == "stats":
                # 統計情報: {元ファイル名}_stats.csv
                filename = f"{base_name}_stats.csv"
            else:
                # 表示データ: {元ファイル名}_data.csv
                filename = f"{base_name}_data.csv"
            
            # CSVに変換
            csv_data = data.to_csv(index=False, encoding='utf-8')
            return csv_data
            
        except Exception as e:
            st.error(f"データエクスポートエラー: {str(e)}")
            return ""
    
    def get_all_metadata(self, limit: int = 20) -> Dict[str, Any]:
        """全メタデータを取得（README.md仕様：オプションで全メタデータ項目の表示）"""
        if self.metadata is None:
            return {}
        
        # 制限数まで表示
        items = list(self.metadata.items())[:limit]
        return dict(items)
    
    def get_debug_info(self) -> Dict[str, Any]:
        """デバッグ情報を取得（README.md仕様：使用ライブラリ、データ変換プロセスの詳細表示）"""
        debug_info = {
            'used_library': self.used_library,
            'filename': self.filename,
            'data_shape': self.fcs_data.shape if self.fcs_data is not None else None,
            'data_type': str(type(self.fcs_data)) if self.fcs_data is not None else None,
            'metadata_keys_count': len(self.metadata) if self.metadata else 0,
            'column_names': list(self.fcs_data.columns) if self.fcs_data is not None else []
        }
        
        return debug_info


def load_and_process_fcs(uploaded_file, transformation: str = "なし", max_events: int = 10000) -> Tuple[Optional['FCSProcessor'], Optional[pd.DataFrame], Optional[Dict]]:
    """
    FCSファイルを読み込んで処理する主要関数（README.md仕様準拠）
    
    Args:
        uploaded_file: Streamlitのアップロードファイル
        transformation: データ変換タイプ（なし、Log10、Asinh、Biexponential）
        max_events: 最大イベント数（README.md仕様：1,000～100,000）
    
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
        
        # FCSファイルを読み込み（README.md仕様：fcsparser → flowio → flowkit）
        data, metadata, used_library = processor.load_fcs_file()
        
        if data is None:
            st.error("FCSファイルの読み込みに失敗しました。")
            return None, None, None
        
        # データの前処理
        processed_data = processor.preprocess_data(data, metadata)
        
        # README.md仕様：パフォーマンス最適化のための最大イベント数設定
        if len(processed_data) > max_events:
            processed_data = processed_data.sample(n=max_events, random_state=42)
            st.info(f"パフォーマンス最適化のため、{max_events:,}イベントにサンプリングしました。")
        
        # データ変換（README.md仕様：なし、Log10、Asinh、Biexponential変換）
        if transformation != "なし":
            for col in processed_data.columns:
                processed_data[col] = processor.apply_transform(processed_data[col], transformation)
        
        # README.md仕様：使用ライブラリ表示
        st.success(f"FCSファイルが正常に読み込まれました（使用ライブラリ: {used_library}）")
        
        return processor, processed_data, metadata
        
    except Exception as e:
        st.error(f"ファイル処理エラー: {str(e)}")
        return None, None, None
