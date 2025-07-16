import pandas as pd
import numpy as np
import io
import streamlit as st
import tempfile
import os
import struct
from typing import Tuple, Dict, Any, Optional

class FCSProcessor:
    """FCSファイル処理クラス - 修正版"""
    
    def __init__(self, file_data: bytes, filename: str):
        self.file_data = file_data
        self.filename = filename
        self.fcs_data = None
        self.metadata = None
        self.used_library = None
        
    def apply_transform(self, data: pd.Series, transform_type: str, channel_name: str = None) -> pd.Series:
        """データ変換を適用（修正版）"""
        if transform_type == "なし":
            return data
        
        try:
            if transform_type == "Log10":
                # より適切な負の値とゼロの処理
                data_transformed = data.copy()
                
                # データの範囲を確認
                min_val = data_transformed.min()
                max_val = data_transformed.max()
                
                # 負の値がある場合の警告
                if min_val <= 0:
                    negative_count = (data_transformed <= 0).sum()
                    st.warning(f"チャンネル {channel_name}: {negative_count}個の負の値またはゼロがあります。最小正の値で置換します。")
                    
                    # 最小正の値を見つける
                    positive_values = data_transformed[data_transformed > 0]
                    if len(positive_values) > 0:
                        min_positive = positive_values.min()
                        replacement_value = min_positive * 0.1  # 最小正の値の10%
                    else:
                        replacement_value = 1e-6
                    
                    data_transformed[data_transformed <= 0] = replacement_value
                
                return np.log10(data_transformed)
            
            elif transform_type == "Asinh":
                # 動的co-factorの計算
                data_range = data.max() - data.min()
                
                # データ範囲に基づいてco-factorを調整
                if data_range < 1000:
                    co_factor = 5  # 低い値の範囲
                elif data_range < 10000:
                    co_factor = 150  # 標準的な蛍光データ
                else:
                    co_factor = data_range / 100  # 高い値の範囲
                
                # メタデータからco-factorを取得する試み
                if self.metadata and channel_name:
                    # チャンネル番号を取得
                    channel_num = None
                    for i, col in enumerate(self.fcs_data.columns):
                        if col == channel_name:
                            channel_num = i + 1
                            break
                    
                    if channel_num:
                        # FlowJoスタイルのco-factor
                        cofactor_key = f'$P{channel_num}E'
                        if cofactor_key in self.metadata:
                            try:
                                cofactor_values = self.metadata[cofactor_key].split(',')
                                if len(cofactor_values) >= 2:
                                    co_factor = float(cofactor_values[1])
                            except:
                                pass
                
                st.info(f"チャンネル {channel_name}: co-factor = {co_factor:.1f} でAsinh変換を適用")
                return np.arcsinh(data / co_factor)
            
            else:
                return data
                
        except Exception as e:
            st.warning(f"データ変換に失敗しました: {str(e)}")
            return data
    
    def _simple_fcs_parser(self) -> Tuple[Optional[pd.DataFrame], Optional[Dict], Optional[str]]:
        """シンプルなFCSパーサー（修正版）"""
        try:
            data_bytes = self.file_data
            
            # ヘッダーを読み取り
            if len(data_bytes) < 58:
                raise Exception("FCSファイルが短すぎます")
            
            # FCSバージョンを確認
            version = data_bytes[0:6].decode('ascii', errors='ignore')
            if not version.startswith('FCS'):
                raise Exception("有効なFCSファイルではありません")
            
            # TEXT セクションの位置を取得
            text_start = int(data_bytes[10:18].decode('ascii', errors='ignore').strip())
            text_end = int(data_bytes[18:26].decode('ascii', errors='ignore').strip())
            
            # DATAセクションの位置を取得
            data_start = int(data_bytes[26:34].decode('ascii', errors='ignore').strip())
            data_end = int(data_bytes[34:42].decode('ascii', errors='ignore').strip())
            
            # TEXTセクションを解析
            text_section = data_bytes[text_start:text_end+1].decode('ascii', errors='ignore')
            
            # 区切り文字を取得
            delimiter = text_section[0]
            
            # パラメータを解析
            params = {}
            parts = text_section[1:].split(delimiter)
            
            for i in range(0, len(parts)-1, 2):
                if i+1 < len(parts):
                    key = parts[i].strip()
                    value = parts[i+1].strip()
                    params[key] = value
            
            # 基本的なメタデータを構築
            metadata = {
                '$TOT': int(params.get('$TOT', '0')),
                '$PAR': int(params.get('$PAR', '0')),
                '$DATATYPE': params.get('$DATATYPE', 'F'),
                '$MODE': params.get('$MODE', 'L'),
                '$BYTEORD': params.get('$BYTEORD', '1,2,3,4')
            }
            
            # チャンネル情報を追加
            par_count = metadata['$PAR']
            for i in range(1, par_count + 1):
                metadata[f'$P{i}N'] = params.get(f'$P{i}N', f'Par{i}')
                metadata[f'$P{i}S'] = params.get(f'$P{i}S', '')
                metadata[f'$P{i}R'] = params.get(f'$P{i}R', '1024')
                metadata[f'$P{i}B'] = params.get(f'$P{i}B', '32')
                metadata[f'$P{i}E'] = params.get(f'$P{i}E', '0,0')  # 変換パラメータ
            
            # データセクションを解析
            data_section = data_bytes[data_start:data_end+1]
            
            # バイトオーダーを確認
            byte_order = metadata['$BYTEORD']
            if byte_order == '4,3,2,1':
                endian = '>'  # ビッグエンディアン
            else:
                endian = '<'  # リトルエンディアン
            
            # データサイズを計算
            total_events = metadata['$TOT']
            expected_size = total_events * par_count * 4  # 32bit float
            
            if len(data_section) < expected_size:
                st.warning(f"データサイズが不足しています。期待: {expected_size}, 実際: {len(data_section)}")
                # 利用可能なイベント数を再計算
                total_events = len(data_section) // (par_count * 4)
                metadata['$TOT'] = total_events
            
            # データ型に基づいてデータを解析
            if metadata['$DATATYPE'] == 'F':
                # 32bit float
                data_format = f'{endian}{total_events * par_count}f'
                try:
                    data_values = struct.unpack(data_format, data_section[:total_events * par_count * 4])
                except struct.error as e:
                    st.error(f"データ解析エラー: {e}")
                    raise
            else:
                # 他の形式はサポートしていない
                raise Exception(f"データ型 {metadata['$DATATYPE']} はサポートされていません")
            
            # DataFrameを作成
            data_array = np.array(data_values).reshape(total_events, par_count)
            
            # チャンネル名を取得
            channel_names = []
            for i in range(par_count):
                name = metadata.get(f'$P{i+1}N', f'Channel_{i+1}')
                channel_names.append(name)
            
            data_df = pd.DataFrame(data_array, columns=channel_names)
            
            # データの妥当性チェック
            self._validate_data(data_df, metadata)
            
            return self._process_fcs_data(data_df, metadata, 'Simple FCS Parser (Fixed)')
            
        except Exception as e:
            raise Exception(f"シンプルFCSパーサーでのエラー: {str(e)}")
    
    def _validate_data(self, data_df: pd.DataFrame, metadata: Dict):
        """データの妥当性をチェック"""
        # 基本的な統計情報を表示
        st.info("データ検証情報:")
        st.write(f"- イベント数: {len(data_df):,}")
        st.write(f"- パラメータ数: {len(data_df.columns)}")
        
        # 各チャンネルの範囲をチェック
        for i, col in enumerate(data_df.columns):
            param_num = i + 1
            expected_range = metadata.get(f'$P{param_num}R', '1024')
            
            actual_min = data_df[col].min()
            actual_max = data_df[col].max()
            
            # 異常値の検出
            if actual_min < -1000000 or actual_max > 1000000:
                st.warning(f"チャンネル {col}: 異常な値の範囲 ({actual_min:.2f} - {actual_max:.2f})")
            
            # 期待範囲との比較
            try:
                expected_max = float(expected_range)
                if actual_max > expected_max * 2:
                    st.warning(f"チャンネル {col}: 期待範囲を超える値 (期待最大: {expected_max}, 実際最大: {actual_max:.2f})")
            except:
                pass
    
    def get_enhanced_debug_info(self) -> Dict[str, Any]:
        """拡張デバッグ情報を取得"""
        debug_info = self.get_debug_info()
        
        if self.fcs_data is not None:
            # 各チャンネルの詳細統計
            channel_stats = {}
            for col in self.fcs_data.columns:
                try:
                    col_data = self.fcs_data[col]
                    channel_stats[col] = {
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'mean': float(col_data.mean()),
                        'negative_count': int((col_data < 0).sum()),
                        'zero_count': int((col_data == 0).sum()),
                        'data_type': str(col_data.dtype)
                    }
                except:
                    channel_stats[col] = {'error': 'statistics calculation failed'}
            
            debug_info['channel_statistics'] = channel_stats
        
        return debug_info
    
    # 他のメソッドは元のコードと同じ
    def load_fcs_file(self) -> Tuple[Optional[pd.DataFrame], Optional[Dict], Optional[str]]:
        """FCSファイルを読み込む（複数の方法を試行）"""
        
        # 方法1: fcsparserを一時ファイルで試行
        try:
            result = self._load_with_tempfile()
            if result[0] is not None:
                return result
        except Exception as e:
            st.warning(f"一時ファイル方式でのエラー: {str(e)}")
        
        # 方法2: fcsparserをBytesIOで試行（新しいバージョン用）
        try:
            result = self._load_with_bytesio()
            if result[0] is not None:
                return result
        except Exception as e:
            st.warning(f"BytesIO方式でのエラー: {str(e)}")
        
        # 方法3: 代替FCSパーサーを試行
        try:
            result = self._load_with_alternative_parser()
            if result[0] is not None:
                return result
        except Exception as e:
            st.warning(f"代替パーサーでのエラー: {str(e)}")
        
        return None, None, None
