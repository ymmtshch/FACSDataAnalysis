"""
FACS解析アプリ - ゲーティング機能ユーティリティ
フローサイトメトリーデータのゲーティング（解析領域設定）機能を提供
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from shapely.geometry import Polygon, Point
from shapely.geometry.polygon import LinearRing
import uuid
import streamlit as st
from datetime import datetime

class Gate:
    """
    ゲート（解析領域）を表現するクラス
    """
    
    def __init__(self, gate_id: str = None, gate_type: str = "polygon", 
                 coordinates: List[Tuple[float, float]] = None, 
                 axes: Tuple[str, str] = None, name: str = None):
        """
        ゲートの初期化
        
        Args:
            gate_id: ゲートの一意ID
            gate_type: ゲートタイプ（polygon, rectangle, ellipse）
            coordinates: ゲート境界の座標リスト [(x1,y1), (x2,y2), ...]
            axes: 使用する軸のペア (x_axis, y_axis)
            name: ゲート名
        """
        self.gate_id = gate_id or str(uuid.uuid4())
        self.gate_type = gate_type
        self.coordinates = coordinates or []
        self.axes = axes
        self.name = name or f"Gate_{self.gate_id[:8]}"
        self.created_at = datetime.now()
        self.statistics = {}
        self._polygon = None
        
    def add_point(self, x: float, y: float):
        """座標点を追加"""
        self.coordinates.append((x, y))
        self._polygon = None  # キャッシュをクリア
        
    def close_gate(self):
        """ゲートを閉じる（最初の点と最後の点を接続）"""
        if len(self.coordinates) >= 3 and self.coordinates[0] != self.coordinates[-1]:
            self.coordinates.append(self.coordinates[0])
        self._update_polygon()
        
    def _update_polygon(self):
        """Shapely Polygonオブジェクトを更新"""
        if len(self.coordinates) >= 3:
            try:
                self._polygon = Polygon(self.coordinates)
            except Exception as e:
                st.warning(f"ゲート作成エラー: {e}")
                self._polygon = None
                
    def contains_points(self, data: pd.DataFrame) -> np.ndarray:
        """
        データポイントがゲート内にあるかチェック
        
        Args:
            data: 解析対象データ
            
        Returns:
            ゲート内にあるポイントのブール配列
        """
        if not self.axes or not self._polygon:
            return np.array([False] * len(data))
            
        x_col, y_col = self.axes
        if x_col not in data.columns or y_col not in data.columns:
            return np.array([False] * len(data))
            
        points = data[[x_col, y_col]].values
        mask = np.array([self._polygon.contains(Point(p)) for p in points])
        return mask
        
    def calculate_statistics(self, data: pd.DataFrame) -> Dict:
        """
        ゲート内データの統計を計算
        
        Args:
            data: 解析対象データ
            
        Returns:
            統計情報辞書
        """
        mask = self.contains_points(data)
        gated_data = data[mask]
        
        total_events = len(data)
        gated_events = len(gated_data)
        percentage = (gated_events / total_events * 100) if total_events > 0 else 0
        
        stats = {
            'total_events': total_events,
            'gated_events': gated_events,
            'percentage': percentage,
            'gate_name': self.name,
            'gate_type': self.gate_type,
            'axes': self.axes
        }
        
        # 各パラメータの統計
        if len(gated_data) > 0:
            numeric_cols = gated_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                stats[f'{col}_mean'] = float(gated_data[col].mean())
                stats[f'{col}_median'] = float(gated_data[col].median())
                stats[f'{col}_std'] = float(gated_data[col].std())
                
        self.statistics = stats
        return stats
        
    def to_dict(self) -> Dict:
        """ゲート情報を辞書形式で返す"""
        return {
            'gate_id': self.gate_id,
            'gate_type': self.gate_type,
            'coordinates': self.coordinates,
            'axes': self.axes,
            'name': self.name,
            'created_at': self.created_at.isoformat(),
            'statistics': self.statistics
        }
        
    @classmethod
    def from_dict(cls, data: Dict):
        """辞書からゲートオブジェクトを作成"""
        gate = cls(
            gate_id=data['gate_id'],
            gate_type=data['gate_type'],
            coordinates=data['coordinates'],
            axes=data['axes'],
            name=data['name']
        )
        gate.created_at = datetime.fromisoformat(data['created_at'])
        gate.statistics = data.get('statistics', {})
        gate._update_polygon()
        return gate

class GatingManager:
    """
    ゲーティング機能を管理するクラス
    """
    
    def __init__(self):
        self.gates: List[Gate] = []
        self.active_gate: Optional[Gate] = None
        
    def create_gate(self, gate_type: str = "polygon", axes: Tuple[str, str] = None, 
                   name: str = None) -> Gate:
        """
        新しいゲートを作成
        
        Args:
            gate_type: ゲートタイプ
            axes: 使用する軸
            name: ゲート名
            
        Returns:
            作成されたGateオブジェクト
        """
        gate = Gate(gate_type=gate_type, axes=axes, name=name)
        self.active_gate = gate
        return gate
        
    def add_gate(self, gate: Gate):
        """ゲートをリストに追加"""
        if gate not in self.gates:
            self.gates.append(gate)
            
    def remove_gate(self, gate_id: str) -> bool:
        """ゲートを削除"""
        for i, gate in enumerate(self.gates):
            if gate.gate_id == gate_id:
                del self.gates[i]
                if self.active_gate and self.active_gate.gate_id == gate_id:
                    self.active_gate = None
                return True
        return False
        
    def get_gate(self, gate_id: str) -> Optional[Gate]:
        """ゲートIDでゲートを取得"""
        for gate in self.gates:
            if gate.gate_id == gate_id:
                return gate
        return None
        
    def apply_gates(self, data: pd.DataFrame, gate_ids: List[str] = None) -> pd.DataFrame:
        """
        指定されたゲートを適用してデータをフィルタリング
        
        Args:
            data: 解析対象データ
            gate_ids: 適用するゲートIDのリスト（Noneの場合は全ゲート）
            
        Returns:
            ゲーティング後のデータ
        """
        if gate_ids is None:
            gates_to_apply = self.gates
        else:
            gates_to_apply = [self.get_gate(gid) for gid in gate_ids if self.get_gate(gid)]
            
        if not gates_to_apply:
            return data
            
        # 全てのゲートの条件を満たすデータを抽出
        combined_mask = np.ones(len(data), dtype=bool)
        for gate in gates_to_apply:
            gate_mask = gate.contains_points(data)
            combined_mask = combined_mask & gate_mask
            
        return data[combined_mask]
        
    def calculate_gate_hierarchy(self, data: pd.DataFrame) -> Dict:
        """
        ゲート階層統計を計算
        ゲートの親子関係と重複を分析
        """
        hierarchy = {}
        
        for gate in self.gates:
            gate_stats = gate.calculate_statistics(data)
            hierarchy[gate.gate_id] = {
                'gate': gate,
                'stats': gate_stats,
                'children': [],
                'parents': []
            }
            
        # 親子関係の計算（簡易版：座標重複チェック）
        for i, gate1 in enumerate(self.gates):
            for j, gate2 in enumerate(self.gates):
                if i != j and gate1._polygon and gate2._polygon:
                    if gate1._polygon.contains(gate2._polygon):
                        hierarchy[gate1.gate_id]['children'].append(gate2.gate_id)
                        hierarchy[gate2.gate_id]['parents'].append(gate1.gate_id)
                        
        return hierarchy
        
    def export_gates(self) -> List[Dict]:
        """ゲート情報をエクスポート用辞書リストで返す"""
        return [gate.to_dict() for gate in self.gates]
        
    def import_gates(self, gates_data: List[Dict]):
        """ゲート情報をインポート"""
        self.gates = []
        for gate_data in gates_data:
            gate = Gate.from_dict(gate_data)
            self.gates.append(gate)

def create_rectangular_gate(x_min: float, x_max: float, y_min: float, y_max: float,
                          axes: Tuple[str, str], name: str = None) -> Gate:
    """
    矩形ゲートを作成
    
    Args:
        x_min, x_max: X軸の範囲
        y_min, y_max: Y軸の範囲
        axes: 使用する軸
        name: ゲート名
        
    Returns:
        矩形ゲート
    """
    coordinates = [
        (x_min, y_min),
        (x_max, y_min),
        (x_max, y_max),
        (x_min, y_max),
        (x_min, y_min)
    ]
    
    gate = Gate(gate_type="rectangle", coordinates=coordinates, axes=axes, name=name)
    gate._update_polygon()
    return gate

def create_elliptical_gate(center_x: float, center_y: float, 
                          width: float, height: float,
                          axes: Tuple[str, str], name: str = None, 
                          n_points: int = 50) -> Gate:
    """
    楕円ゲートを作成
    
    Args:
        center_x, center_y: 中心座標
        width, height: 幅と高さ
        axes: 使用する軸
        name: ゲート名
        n_points: 楕円を近似する点の数
        
    Returns:
        楕円ゲート
    """
    angles = np.linspace(0, 2*np.pi, n_points)
    coordinates = []
    
    for angle in angles:
        x = center_x + (width/2) * np.cos(angle)
        y = center_y + (height/2) * np.sin(angle)
        coordinates.append((x, y))
        
    coordinates.append(coordinates[0])  # 閉じる
    
    gate = Gate(gate_type="ellipse", coordinates=coordinates, axes=axes, name=name)
    gate._update_polygon()
    return gate

# セッション状態管理用ヘルパー関数
def get_gating_manager() -> GatingManager:
    """セッション状態からGatingManagerを取得または作成"""
    if 'gating_manager' not in st.session_state:
        st.session_state.gating_manager = GatingManager()
    return st.session_state.gating_manager

def save_gates_to_session():
    """ゲート情報をセッション状態に保存"""
    manager = get_gating_manager()
    st.session_state.current_gates = manager.export_gates()

def load_gates_from_session():
    """セッション状態からゲート情報を読み込み"""
    if 'current_gates' in st.session_state:
        manager = get_gating_manager()
        manager.import_gates(st.session_state.current_gates)

# デバッグ・テスト用関数
def generate_test_data(n_points: int = 1000) -> pd.DataFrame:
    """テスト用のサンプルデータを生成"""
    np.random.seed(42)
    data = pd.DataFrame({
        'FSC-A': np.random.normal(50000, 15000, n_points),
        'SSC-A': np.random.normal(30000, 10000, n_points),
        'FITC-A': np.random.exponential(1000, n_points),
        'PE-A': np.random.exponential(800, n_points)
    })
    # 負の値を除去
    data = data[data > 0].dropna()
    return data

if __name__ == "__main__":
    # テスト実行例
    print("FACS Gating Utils Test")
    
    # テストデータ生成
    test_data = generate_test_data()
    print(f"Generated {len(test_data)} test events")
    
    # ゲーティングマネージャー作成
    manager = GatingManager()
    
    # 矩形ゲート作成
    rect_gate = create_rectangular_gate(
        x_min=20000, x_max=80000,
        y_min=10000, y_max=50000,
        axes=('FSC-A', 'SSC-A'),
        name="Lymphocytes"
    )
    manager.add_gate(rect_gate)
    
    # 統計計算
    stats = rect_gate.calculate_statistics(test_data)
    print(f"Rectangle gate stats: {stats['gated_events']}/{stats['total_events']} ({stats['percentage']:.1f}%)")
    
    # ゲーティング適用
    gated_data = manager.apply_gates(test_data, [rect_gate.gate_id])
    print(f"Gated data: {len(gated_data)} events")
