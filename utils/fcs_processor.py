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
            # ここに詳細なエラーログを追加
            st.warning(f"fcsparserでの読み込みに失敗しました。FlowIOを試します。詳細: {str(e)}") # <- 修正済みログメッセージ
            if "newbyteorder" in str(e) or "failed to decode" in str(e): # NumPy 2.0 / デコードエラーも考慮
                st.info("NumPy 2.0互換性またはデコードエラーが検出されました。flowioに自動フォールバックします。")
        
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
            # $PAR キーが存在するか確認し、存在しない場合はfcs.channelsの長さを使用
            num_channels = int(fcs.text.get('$PAR', len(fcs.channels)))
            for i in range(1, num_channels + 1):
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
            # ここに詳細なエラーログを追加
            st.warning(f"flowioでの読み込みに失敗しました。FlowKitを試します。詳細: {str(e)}") # <- 修正済みログメッセージ
        
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
            # ここに詳細なエラーログを追加
            st.error(f"flowkitでの読み込みにも失敗しました。詳細: {str(e)}") # <- 修正済みログメッセージ
        
        st.error("すべてのライブラリでの読み込みに失敗しました。")
        return None, None, None # データ、メタデータ、使用ライブラリ
    
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
        else: # 見つからなかった場合
            info['total_events'] = 'N/A'

        # パラメータ数
        par_keys = ['$PAR', 'par', 'PARAMETERS', 'parameters']
        for key in par_keys:
            if key in self.metadata:
                info['parameters'] = int(self.metadata[key])
                break
        else: # 見つからなかった場合
            info['parameters'] = 'N/A'
        
        # 取得日時
        date_keys = ['$DATE', 'date', 'DATE']
        for key in date_keys:
            if key in self.metadata:
                info['date'] = self.metadata[key]
                break
        else: info['date'] = 'N/A'
        
        # 取得開始時刻
        btim_keys = ['$BTIM', 'btim', 'BTIM']
        for key in btim_keys:
            if key in self.metadata:
                info['begin_time'] = self.metadata[key]
                break
        else: info['begin_time'] = 'N/A'
        
        # 取得終了時刻
        etim_keys = ['$ETIM', 'etim', 'ETIM']
        for key in etim_keys:
            if key in self.metadata:
                info['end_time'] = self.metadata[key]
                break
        else: info['end_time'] = 'N/A'
        
        # 使用機器
        cyt_keys = ['$CYT', 'cyt', 'CYTOMETER']
        for key in cyt_keys:
            if key in self.metadata:
                info['cytometer'] = self.metadata[key]
                break
        else: info['cytometer'] = 'N/A'
        
        # 機器番号
        cytnum_keys = ['$CYTNUM', 'cytnum', 'CYTNUM']
        for key in cytnum_keys:
            if key in self.metadata:
                info['cytometer_number'] = self.metadata[key]
                break
        else: info['cytometer_number'] = 'N/A'
        
        # 実験名
        exp_keys = ['EXPERIMENT NAME', '$EXP', 'exp', 'EXPERIMENT']
        for key in exp_keys:
            if key in self.metadata:
                info['experiment_name'] = self.metadata[key]
                break
        else: info['experiment_name'] = 'N/A'
        
        # サンプルID
        sample_keys = ['SAMPLE ID', '$SMNO', 'smno', 'SAMPLE']
        for key in sample_keys:
            if key in self.metadata:
                info['sample_id'] = self.metadata[key]
                break
        else: info['sample_id'] = 'N/A'
        
        # オペレーター
        op_keys = ['$OP', 'op', 'OPERATOR']
        for key in op_keys:
            if key in self.metadata:
                info['operator'] = self.metadata[key]
                break
        else: info['operator'] = 'N/A'
        
        # ソフトウェア情報（README.md仕様で言及されている項目）
        software_keys = ['$SYS', 'sys', 'SOFTWARE', 'software']
        for key in software_keys:
            if key in self.metadata:
                info['software'] = self.metadata[key]
                break
        else: info['software'] = 'N/A'
        
        return info
    
    def get_channel_info(self) -> Dict[str, Any]:
        """チャンネル詳細情報を取得（README.md仕様：各チャンネルの詳細情報と設定値）"""
        if self.metadata is None or self.fcs_data is None:
            return {}
        
        channel_info = {}
        
        # DataFrameのカラム名からチャンネル情報を取得する
        for i, col_name in enumerate(self.fcs_data.columns):
            param_info = {}
            
            # チャンネル名（$PnN を優先、$PnS があればそれも取得）
            name_key = f'$P{i+1}N'
            short_key = f'$P{i+1}S'
            
            display_name = self.metadata.get(name_key, col_name)
            detector_name = self.metadata.get(short_key, '')

            # 表示名と検出器名を結合してより分かりやすい名前を作成
            if display_name and detector_name and display_name != detector_name:
                full_channel_name = f"{display_name} ({detector_name})"
            elif display_name:
                full_channel_name = display_name
            elif detector_name:
                full_channel_name = detector_name
            else:
                full_channel_name = col_name # どちらもなければ元のカラム名
            
            param_info['name'] = full_channel_name
            if detector_name:
                param_info['short_name'] = detector_name # 検出器名をショート名として保持
            
            # レンジ
            range_key = f'$P{i+1}R'
            param_info['range'] = self.metadata.get(range_key, 'N/A')
            
            # ビット数
            bits_key = f'$P{i+1}B'
            param_info['bits'] = self.metadata.get(bits_key, 'N/A')
            
            # ゲイン
            gain_key = f'$P{i+1}G'
            param_info['gain'] = self.metadata.get(gain_key, 'N/A')
            
            # 電圧
            voltage_key = f'$P{i+1}V'
            param_info['voltage'] = self.metadata.get(voltage_key, 'N/A')
            
            # 増幅器タイプ
            amp_key = f'$P{i+1}T'
            param_info['amplifier_type'] = self.metadata.get(amp_key, 'N/A')
            
            channel_info[f'P{i+1}'] = param_info
        
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
                st.warning(f"チャンネル '{col}' の統計計算に失敗しました: {e}")
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
                # co-factor (e.g., 150, 5) はFCSファイルから取得するか、デフォルト値を使用
                # ここでは簡易的に1000をco-factorとして使用
                return np.arcsinh(data / 1000)
            
            elif transform_type == "Biexponential":
                # README.md仕様：Biexponential変換（簡易実装）
                # Biexponential変換は複雑なため、より簡易的な対数変換を適用
                # FlowJoのBiexponentialは、負の値も扱える対数変換に近い
                # ここでは np.arcsinh をより低レンジに適用する例
                # `w` パラメータは通常メタデータから取得されるが、ここでは固定値
                w = 0.5 # 負の値をどれくらい直線的にするかを制御するパラメータ
                data_transformed = data.copy()
                return np.where(data_transformed >= 0, 
                                np.log10(data_transformed + 1), 
                                -np.log10(np.abs(data_transformed) + 1) * (1 / w))
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
        items = list(self.metadata.items())
        if limit and limit > 0:
            items = items[:limit]
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


def load_and_process_fcs(uploaded_file, transformation: str = "なし", max_events: int = 10000) -> Tuple[Optional['FCSProcessor'], Optional[pd.DataFrame], Optional[Dict], Optional[str]]:
    """
    FCSファイルを読み込んで処理する主要関数（README.md仕様準拠）
    
    Args:
        uploaded_file: Streamlitのアップロードファイル
        transformation: データ変換タイプ（なし、Log10、Asinh、Biexponential）
        max_events: 最大イベント数（README.md仕様：1,000～100,000）
    
    Returns:
        Tuple[FCSProcessor, DataFrame, metadata, error_message]: プロセッサ、データ、メタデータ、エラーメッセージ
    """
    if uploaded_file is None:
        return None, None, None, "ファイルがアップロードされていません。"
    
    try:
        # ファイルデータを読み込み
        file_data = uploaded_file.read()
        filename = uploaded_file.name
        
        # FCSProcessorを初期化
        processor = FCSProcessor(file_data, filename)
        
        # FCSファイルを読み込み（README.md仕様：fcsparser → flowio → flowkit）
        # load_fcs_fileは (data_df, metadata_dict, used_library) を返す
        data, metadata, used_library = processor.load_fcs_file()
        
        if data is None:
            # load_fcs_fileがNoneを返した場合、エラーメッセージはStreamlitのst.errorで出力済みなので、ここでは汎用エラーメッセージを返す
            # FCSProcessor.load_fcs_file() の最終 return で st.error が呼ばれているため、ここでは詳細なエラーメッセージではなく、
            # 呼び出し元 (app.py) に返すための簡潔なメッセージとする
            return None, None, None, "FCSファイルの読み込みに失敗しました。"
        
        # データの前処理
        processed_data = processor.preprocess_data(data, metadata)
        
        # README.md仕様：パフォーマンス最適化のための最大イベント数設定
        if len(processed_data) > max_events:
            processed_data = processed_data.sample(n=max_events, random_state=42)
            st.info(f"パフォーマンス最適化のため、{max_events:,}イベントにサンプリングしました。")
        
        # データ変換（README.md仕様：なし、Log10、Asinh、Biexponential変換）
        if transformation != "なし":
            for col in processed_data.columns:
                # 数値型カラムのみ変換を適用
                if pd.api.types.is_numeric_dtype(processed_data[col]):
                    processed_data[col] = processor.apply_transform(processed_data[col], transformation)
        
        # README.md仕様：使用ライブラリ表示
        st.success(f"FCSファイルが正常に読み込まれました（使用ライブラリ: {used_library}）")
        
        # 成功時はエラーメッセージをNoneにする
        return processor, processed_data, metadata, None
        
    except Exception as e:
        # 予期せぬエラーが発生した場合
        st.error(f"ファイル処理中に予期せぬエラーが発生しました: {str(e)}")
        return None, None, None, f"ファイル処理中に予期せぬエラーが発生しました: {str(e)}"
