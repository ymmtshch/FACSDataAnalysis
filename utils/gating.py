# utils/gating.py - Simplified Gating utilities for FACS Data Analysis

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import re
from typing import List, Dict, Optional, Tuple, Union

class Gate:
    """Base class for all gate types"""
    
    def __init__(self, name: str, gate_type: str, channels: Union[str, List[str]], color: str = 'red'):
        self.name = name
        self.gate_type = gate_type
        self.channels = channels if isinstance(channels, list) else [channels]
        self.color = color
        self.created_at = pd.Timestamp.now()
    
    def apply(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Apply gate to data - to be implemented by subclasses"""
        raise NotImplementedError
    
    def get_statistics(self, data: pd.DataFrame) -> Dict:
        """Get statistics for gated data"""
        try:
            gated_data = self.apply(data)
            if gated_data is None or len(gated_data) == 0:
                return {
                    'total_events': len(data),
                    'gated_events': 0,
                    'percentage': 0.0,
                    'gate_name': self.name,
                    'gate_type': self.gate_type,
                    'channels': self.channels,
                    'statistics': {}
                }
            
            total_events = len(data)
            gated_events = len(gated_data)
            percentage = (gated_events / total_events) * 100 if total_events > 0 else 0
            
            # Calculate statistics for gated data
            statistics = {}
            for channel in self.channels:
                if channel in gated_data.columns:
                    channel_data = gated_data[channel].dropna()
                    if len(channel_data) > 0:
                        statistics[channel] = {
                            'mean': float(channel_data.mean()),
                            'median': float(channel_data.median()),
                            'std': float(channel_data.std()),
                            'min': float(channel_data.min()),
                            'max': float(channel_data.max()),
                            'count': int(len(channel_data))
                        }
            
            return {
                'total_events': total_events,
                'gated_events': gated_events,
                'percentage': percentage,
                'gate_name': self.name,
                'gate_type': self.gate_type,
                'channels': self.channels,
                'statistics': statistics
            }
        except Exception as e:
            st.error(f"Error calculating statistics for gate '{self.name}': {e}")
            return {
                'total_events': len(data),
                'gated_events': 0,
                'percentage': 0.0,
                'gate_name': self.name,
                'gate_type': self.gate_type,
                'channels': self.channels,
                'statistics': {}
            }

class ThresholdGate(Gate):
    """Threshold gate for single channel data"""
    
    def __init__(self, name: str, channel: str, threshold: float, direction: str = 'above', color: str = 'orange'):
        super().__init__(name, 'threshold', [channel], color)
        self.channel = channel
        self.threshold = float(threshold)
        self.direction = direction  # 'above' or 'below'
    
    def apply(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Apply threshold gate to data"""
        try:
            if self.channel not in data.columns:
                st.error(f"Channel {self.channel} not found in data")
                return None
            
            channel_data = pd.to_numeric(data[self.channel], errors='coerce')
            valid_mask = ~channel_data.isna()
            
            if self.direction == 'above':
                threshold_mask = channel_data >= self.threshold
            else:  # below
                threshold_mask = channel_data <= self.threshold
            
            final_mask = valid_mask & threshold_mask
            return data[final_mask].copy()
        except Exception as e:
            st.error(f"Error applying threshold gate '{self.name}': {e}")
            return None

class RectangleGate(Gate):
    """Rectangle gate for 2D data"""
    
    def __init__(self, name: str, x_channel: str, y_channel: str, 
                 x_min: float, x_max: float, y_min: float, y_max: float, color: str = 'blue'):
        super().__init__(name, 'rectangular', [x_channel, y_channel], color)
        self.x_channel = x_channel
        self.y_channel = y_channel
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
    
    def apply(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Apply rectangle gate to data"""
        try:
            if self.x_channel not in data.columns or self.y_channel not in data.columns:
                st.error(f"Channels {self.x_channel} or {self.y_channel} not found in data")
                return None
            
            x_data = pd.to_numeric(data[self.x_channel], errors='coerce')
            y_data = pd.to_numeric(data[self.y_channel], errors='coerce')
            
            mask = (
                (x_data >= self.x_min) &
                (x_data <= self.x_max) &
                (y_data >= self.y_min) &
                (y_data <= self.y_max) &
                (~x_data.isna()) &
                (~y_data.isna())
            )
            
            return data[mask].copy()
        except Exception as e:
            st.error(f"Error applying rectangle gate '{self.name}': {e}")
            return None
    
    def get_rectangle_coords(self) -> Dict[str, List[float]]:
        """Get rectangle coordinates for plotting"""
        return {
            'x': [self.x_min, self.x_max, self.x_max, self.x_min, self.x_min],
            'y': [self.y_min, self.y_min, self.y_max, self.y_max, self.y_min]
        }

class PolygonGate(Gate):
    """Polygon gate for 2D data"""
    
    def __init__(self, name: str, x_channel: str, y_channel: str, 
                 vertices: List[Tuple[float, float]], color: str = 'red'):
        super().__init__(name, 'polygon', [x_channel, y_channel], color)
        self.x_channel = x_channel
        self.y_channel = y_channel
        self.vertices = [(float(x), float(y)) for x, y in vertices]
    
    def apply(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Apply polygon gate to data using ray casting algorithm"""
        try:
            if self.x_channel not in data.columns or self.y_channel not in data.columns:
                st.error(f"Channels {self.x_channel} or {self.y_channel} not found in data")
                return None
            
            if len(self.vertices) < 3:
                st.error("Need at least 3 vertices for polygon gate")
                return None
            
            x_data = pd.to_numeric(data[self.x_channel], errors='coerce')
            y_data = pd.to_numeric(data[self.y_channel], errors='coerce')
            
            valid_mask = ~(x_data.isna() | y_data.isna())
            
            # Apply ray casting algorithm
            inside_mask = np.zeros(len(data), dtype=bool)
            valid_indices = np.where(valid_mask)[0]
            
            for i in valid_indices:
                x, y = x_data.iloc[i], y_data.iloc[i]
                inside_mask[i] = self._point_in_polygon(x, y)
            
            return data[inside_mask].copy()
        except Exception as e:
            st.error(f"Error applying polygon gate '{self.name}': {e}")
            return None
    
    def _point_in_polygon(self, x: float, y: float) -> bool:
        """Check if point is inside polygon using ray casting"""
        n = len(self.vertices)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = self.vertices[i]
            xj, yj = self.vertices[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        
        return inside
    
    def get_vertices_array(self) -> np.ndarray:
        """Get vertices as numpy array"""
        return np.array(self.vertices)

class EllipseGate(Gate):
    """Ellipse gate for 2D data"""
    
    def __init__(self, name: str, x_channel: str, y_channel: str, 
                 center_x: float, center_y: float, width: float, height: float, 
                 angle: float = 0, color: str = 'green'):
        super().__init__(name, 'ellipse', [x_channel, y_channel], color)
        self.x_channel = x_channel
        self.y_channel = y_channel
        self.center_x = float(center_x)
        self.center_y = float(center_y)
        self.width = float(width)
        self.height = float(height)
        self.angle = float(angle)  # rotation angle in degrees
    
    def apply(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Apply ellipse gate to data"""
        try:
            if self.x_channel not in data.columns or self.y_channel not in data.columns:
                st.error(f"Channels {self.x_channel} or {self.y_channel} not found in data")
                return None
            
            x_data = pd.to_numeric(data[self.x_channel], errors='coerce')
            y_data = pd.to_numeric(data[self.y_channel], errors='coerce')
            
            valid_mask = ~(x_data.isna() | y_data.isna())
            
            # Translate points to ellipse center
            x_translated = x_data - self.center_x
            y_translated = y_data - self.center_y
            
            # Rotate points if angle is specified
            if self.angle != 0:
                angle_rad = np.radians(self.angle)
                cos_angle = np.cos(angle_rad)
                sin_angle = np.sin(angle_rad)
                
                x_rotated = x_translated * cos_angle + y_translated * sin_angle
                y_rotated = -x_translated * sin_angle + y_translated * cos_angle
            else:
                x_rotated = x_translated
                y_rotated = y_translated
            
            # Check if points are inside ellipse
            ellipse_eq = (x_rotated / (self.width / 2)) ** 2 + (y_rotated / (self.height / 2)) ** 2
            inside_ellipse = ellipse_eq <= 1
            
            final_mask = valid_mask & inside_ellipse
            return data[final_mask].copy()
        except Exception as e:
            st.error(f"Error applying ellipse gate '{self.name}': {e}")
            return None
    
    def get_ellipse_points(self, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Get ellipse boundary points for plotting"""
        theta = np.linspace(0, 2 * np.pi, n_points)
        
        # Ellipse points in local coordinates
        x_local = (self.width / 2) * np.cos(theta)
        y_local = (self.height / 2) * np.sin(theta)
        
        # Rotate if necessary
        if self.angle != 0:
            angle_rad = np.radians(self.angle)
            cos_angle = np.cos(angle_rad)
            sin_angle = np.sin(angle_rad)
            
            x_rotated = x_local * cos_angle - y_local * sin_angle
            y_rotated = x_local * sin_angle + y_local * cos_angle
        else:
            x_rotated = x_local
            y_rotated = y_local
        
        # Translate to center
        x_final = x_rotated + self.center_x
        y_final = y_rotated + self.center_y
        
        return x_final, y_final

class GateManager:
    """Manager class for handling multiple gates"""
    
    def __init__(self):
        self.gates = {}
    
    def create_threshold_gate(self, name: str, channel: str, threshold: float, direction: str) -> Optional[ThresholdGate]:
        """Create threshold gate"""
        try:
            gate = ThresholdGate(name, channel, threshold, direction)
            self.add_gate(gate)
            return gate
        except Exception as e:
            st.error(f"Error creating threshold gate '{name}': {e}")
            return None
    
    def create_rectangular_gate(self, name: str, x_channel: str, y_channel: str, 
                               x_min: float, x_max: float, y_min: float, y_max: float) -> Optional[RectangleGate]:
        """Create rectangular gate"""
        try:
            gate = RectangleGate(name, x_channel, y_channel, x_min, x_max, y_min, y_max)
            self.add_gate(gate)
            return gate
        except Exception as e:
            st.error(f"Error creating rectangular gate '{name}': {e}")
            return None
    
    def create_polygon_gate(self, name: str, x_channel: str, y_channel: str, 
                           coordinates: Union[str, List[Tuple[float, float]]]) -> Optional[PolygonGate]:
        """Create polygon gate"""
        try:
            if isinstance(coordinates, str):
                vertices = self.parse_polygon_coordinates(coordinates)
            else:
                vertices = coordinates
            
            if vertices is None or len(vertices) < 3:
                st.error("Need at least 3 vertices for polygon gate")
                return None
            
            gate = PolygonGate(name, x_channel, y_channel, vertices)
            self.add_gate(gate)
            return gate
        except Exception as e:
            st.error(f"Error creating polygon gate '{name}': {e}")
            return None
    
    def create_ellipse_gate(self, name: str, x_channel: str, y_channel: str, 
                           center_x: float, center_y: float, width: float, height: float) -> Optional[EllipseGate]:
        """Create ellipse gate"""
        try:
            gate = EllipseGate(name, x_channel, y_channel, center_x, center_y, width, height)
            self.add_gate(gate)
            return gate
        except Exception as e:
            st.error(f"Error creating ellipse gate '{name}': {e}")
            return None
    
    def add_gate(self, gate: Gate) -> Gate:
        """Add a gate to the manager"""
        if not isinstance(gate, Gate):
            raise ValueError("Object must be a Gate instance")
        
        self.gates[gate.name] = gate
        return gate
    
    def remove_gate(self, gate_name: str) -> bool:
        """Remove a gate from the manager"""
        if gate_name in self.gates:
            del self.gates[gate_name]
            return True
        else:
            st.error(f"Gate '{gate_name}' not found")
            return False
    
    def get_gate(self, gate_name: str) -> Optional[Gate]:
        """Get a gate by name"""
        return self.gates.get(gate_name)
    
    def list_gates(self) -> List[str]:
        """List all gates"""
        return list(self.gates.keys())
    
    def apply_gate(self, data: pd.DataFrame, gate: Union[str, Gate]) -> Optional[pd.DataFrame]:
        """Apply gate to data"""
        if isinstance(gate, str):
            gate_obj = self.get_gate(gate)
            if gate_obj is None:
                st.error(f"Gate '{gate}' not found")
                return None
        elif isinstance(gate, Gate):
            gate_obj = gate
        else:
            st.error("Invalid gate parameter")
            return None
        
        return gate_obj.apply(data)
    
    def add_gate_to_plot(self, fig, gate: Union[str, Gate], gate_index: int = 0):
        """Add gate visualization to plot"""
        try:
            if isinstance(gate, str):
                gate_obj = self.get_gate(gate)
            else:
                gate_obj = gate
            
            if gate_obj is None:
                return fig
            
            # Color cycling for multiple gates
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            color = colors[gate_index % len(colors)]
            
            if isinstance(gate_obj, RectangleGate):
                coords = gate_obj.get_rectangle_coords()
                fig.add_trace(go.Scatter(
                    x=coords['x'],
                    y=coords['y'],
                    mode='lines',
                    line=dict(color=color, width=2),
                    name=f"Gate: {gate_obj.name}",
                    showlegend=True
                ))
            
            elif isinstance(gate_obj, PolygonGate):
                vertices = gate_obj.get_vertices_array()
                # Close the polygon
                x_coords = list(vertices[:, 0]) + [vertices[0, 0]]
                y_coords = list(vertices[:, 1]) + [vertices[0, 1]]
                
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines',
                    line=dict(color=color, width=2),
                    name=f"Gate: {gate_obj.name}",
                    showlegend=True
                ))
            
            elif isinstance(gate_obj, EllipseGate):
                x_ellipse, y_ellipse = gate_obj.get_ellipse_points()
                fig.add_trace(go.Scatter(
                    x=x_ellipse,
                    y=y_ellipse,
                    mode='lines',
                    line=dict(color=color, width=2),
                    name=f"Gate: {gate_obj.name}",
                    showlegend=True
                ))
            
            return fig
        except Exception as e:
            st.error(f"Error adding gate to plot: {e}")
            return fig
    
    def parse_polygon_coordinates(self, coordinates_str: str) -> Optional[List[Tuple[float, float]]]:
        """Parse polygon coordinates from string"""
        try:
            coords_str = coordinates_str.strip()
            
            # Format: "x1,y1 x2,y2 x3,y3"
            if ' ' in coords_str and ',' in coords_str:
                pairs = coords_str.split()
                vertices = []
                for pair in pairs:
                    if ',' in pair:
                        parts = pair.split(',')
                        if len(parts) >= 2:
                            x, y = parts[0].strip(), parts[1].strip()
                            vertices.append((float(x), float(y)))
                return vertices
            
            # Format: "x1,y1;x2,y2;x3,y3"
            elif ';' in coords_str:
                pairs = coords_str.split(';')
                vertices = []
                for pair in pairs:
                    if ',' in pair:
                        parts = pair.split(',')
                        if len(parts) >= 2:
                            x, y = parts[0].strip(), parts[1].strip()
                            vertices.append((float(x), float(y)))
                return vertices
            
            # Format: "(x1,y1),(x2,y2),(x3,y3)"
            elif '(' in coords_str and ')' in coords_str:
                pattern = r'\(([^)]+)\)'
                matches = re.findall(pattern, coords_str)
                vertices = []
                for match in matches:
                    if ',' in match:
                        parts = match.split(',')
                        if len(parts) >= 2:
                            x, y = parts[0].strip(), parts[1].strip()
                            vertices.append((float(x), float(y)))
                return vertices
            
            else:
                st.error("Unsupported coordinate format. Use: '1,2 3,4 5,6' or '1,2;3,4;5,6' or '(1,2),(3,4),(5,6)'")
                return None
                
        except Exception as e:
            st.error(f"Error parsing coordinates: {e}")
            return None
    
    def get_gate_statistics(self, gate_name: str, data: pd.DataFrame) -> Optional[Dict]:
        """Get statistics for a specific gate"""
        gate = self.get_gate(gate_name)
        if gate is None:
            return None
        
        return gate.get_statistics(data)
    
    def get_all_gate_statistics(self, data: pd.DataFrame) -> Dict:
        """Get statistics for all gates"""
        stats = {}
        for gate_name, gate in self.gates.items():
            stats[gate_name] = gate.get_statistics(data)
        
        return stats
    
    def clear_gates(self):
        """Clear all gates"""
        self.gates.clear()
    
    def export_gates(self) -> Dict:
        """Export gate definitions"""
        gate_definitions = {}
        
        for name, gate in self.gates.items():
            gate_def = {
                'name': gate.name,
                'type': gate.gate_type,
                'channels': gate.channels,
                'color': gate.color,
                'created_at': gate.created_at.isoformat()
            }
            
            # Add type-specific parameters
            if isinstance(gate, PolygonGate):
                gate_def['vertices'] = gate.vertices
            elif isinstance(gate, RectangleGate):
                gate_def.update({
                    'x_min': gate.x_min,
                    'x_max': gate.x_max,
                    'y_min': gate.y_min,
                    'y_max': gate.y_max
                })
            elif isinstance(gate, EllipseGate):
                gate_def.update({
                    'center_x': gate.center_x,
                    'center_y': gate.center_y,
                    'width': gate.width,
                    'height': gate.height,
                    'angle': gate.angle
                })
            elif isinstance(gate, ThresholdGate):
                gate_def.update({
                    'threshold': gate.threshold,
                    'direction': gate.direction
                })
            
            gate_definitions[name] = gate_def
        
        return gate_definitions
