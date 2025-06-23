# utils/gating.py - Fixed Gating utilities for FACS Data Analysis
# Updated to resolve conflicts with README specifications

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.geometry.polygon import LinearRing
from scipy.spatial import ConvexHull
import streamlit as st
import plotly.graph_objects as go
import re

class Gate:
    """Base class for all gate types"""
    
    def __init__(self, name, gate_type, channels, color='red'):
        self.name = name
        self.gate_type = gate_type
        self.channels = channels if isinstance(channels, list) else [channels]
        self.color = color
        self.created_at = pd.Timestamp.now()
    
    def apply(self, data):
        """Apply gate to data - to be implemented by subclasses"""
        raise NotImplementedError
    
    def get_statistics(self, data):
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
                    channel_data = gated_data[channel]
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
            return None

class RectangleGate(Gate):
    """Rectangle gate for 2D data - matches README specification"""
    
    def __init__(self, name, x_channel, y_channel, x_min, x_max, y_min, y_max, color='blue'):
        super().__init__(name, 'rectangular', [x_channel, y_channel], color)
        self.x_channel = x_channel
        self.y_channel = y_channel
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
    
    def apply(self, data):
        """Apply rectangle gate to data"""
        try:
            if self.x_channel not in data.columns or self.y_channel not in data.columns:
                st.error(f"Channels {self.x_channel} or {self.y_channel} not found in data")
                return None
            
            mask = (
                (data[self.x_channel] >= self.x_min) &
                (data[self.x_channel] <= self.x_max) &
                (data[self.y_channel] >= self.y_min) &
                (data[self.y_channel] <= self.y_max)
            )
            
            return data[mask].copy()
        except Exception as e:
            st.error(f"Error applying rectangle gate '{self.name}': {e}")
            return None
    
    def get_rectangle_coords(self):
        """Get rectangle coordinates for plotting"""
        return {
            'x': [self.x_min, self.x_max, self.x_max, self.x_min, self.x_min],
            'y': [self.y_min, self.y_min, self.y_max, self.y_max, self.y_min]
        }

class PolygonGate(Gate):
    """Polygon gate for 2D data"""
    
    def __init__(self, name, x_channel, y_channel, vertices, color='red'):
        super().__init__(name, 'polygon', [x_channel, y_channel], color)
        self.x_channel = x_channel
        self.y_channel = y_channel
        self.vertices = vertices
        self.polygon = None
        self._create_polygon()
    
    def _create_polygon(self):
        """Create Shapely polygon from vertices"""
        if len(self.vertices) >= 3:
            try:
                # Ensure vertices are numeric
                numeric_vertices = [(float(x), float(y)) for x, y in self.vertices]
                self.polygon = Polygon(numeric_vertices)
                if not self.polygon.is_valid:
                    # Try to fix invalid polygon
                    self.polygon = self.polygon.buffer(0)
            except Exception as e:
                st.error(f"Error creating polygon for gate '{self.name}': {e}")
                self.polygon = None
    
    def apply(self, data):
        """Apply polygon gate to data"""
        try:
            if self.polygon is None:
                return None
            
            if self.x_channel not in data.columns or self.y_channel not in data.columns:
                st.error(f"Channels {self.x_channel} or {self.y_channel} not found in data")
                return None
            
            # Get coordinates and handle NaN values
            x_data = pd.to_numeric(data[self.x_channel], errors='coerce')
            y_data = pd.to_numeric(data[self.y_channel], errors='coerce')
            
            # Create mask for valid data points
            valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
            valid_indices = data.index[valid_mask]
            
            if len(valid_indices) == 0:
                return data.iloc[0:0].copy()  # Return empty dataframe with same structure
            
            # Check which points are inside the polygon
            points = list(zip(x_data[valid_mask], y_data[valid_mask]))
            inside_mask = np.array([self.polygon.contains(Point(point)) for point in points])
            
            # Map back to original dataframe indices
            final_mask = np.zeros(len(data), dtype=bool)
            final_mask[valid_indices[inside_mask]] = True
            
            return data[final_mask].copy()
        except Exception as e:
            st.error(f"Error applying polygon gate '{self.name}': {e}")
            return None
    
    def get_vertices_array(self):
        """Get vertices as numpy array"""
        return np.array(self.vertices)

class EllipseGate(Gate):
    """Ellipse gate for 2D data"""
    
    def __init__(self, name, x_channel, y_channel, center_x, center_y, 
                 width, height, angle=0, color='green'):
        super().__init__(name, 'ellipse', [x_channel, y_channel], color)
        self.x_channel = x_channel
        self.y_channel = y_channel
        self.center_x = float(center_x)
        self.center_y = float(center_y)
        self.width = float(width)
        self.height = float(height)
        self.angle = float(angle)  # rotation angle in degrees
    
    def apply(self, data):
        """Apply ellipse gate to data"""
        try:
            if self.x_channel not in data.columns or self.y_channel not in data.columns:
                st.error(f"Channels {self.x_channel} or {self.y_channel} not found in data")
                return None
            
            # Get coordinates and handle NaN values
            x_data = pd.to_numeric(data[self.x_channel], errors='coerce')
            y_data = pd.to_numeric(data[self.y_channel], errors='coerce')
            
            # Create mask for valid data points
            valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
            
            if not valid_mask.any():
                return data.iloc[0:0].copy()  # Return empty dataframe
            
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
            
            # Combine with valid mask
            final_mask = valid_mask & inside_ellipse
            
            return data[final_mask].copy()
        except Exception as e:
            st.error(f"Error applying ellipse gate '{self.name}': {e}")
            return None
    
    def get_ellipse_points(self, n_points=100):
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

class ThresholdGate(Gate):
    """Threshold gate for 1D data - matches README specification"""
    
    def __init__(self, name, channel, threshold, direction='above', color='orange'):
        super().__init__(name, 'threshold', [channel], color)
        self.channel = channel
        self.threshold = float(threshold)
        self.direction = direction  # 'above', 'below', 'greater', 'less'
    
    def apply(self, data):
        """Apply threshold gate to data"""
        try:
            if self.channel not in data.columns:
                st.error(f"Channel {self.channel} not found in data")
                return None
            
            channel_data = pd.to_numeric(data[self.channel], errors='coerce')
            valid_mask = ~np.isnan(channel_data)
            
            if self.direction in ['above', 'greater']:
                threshold_mask = channel_data >= self.threshold
            elif self.direction in ['below', 'less']:
                threshold_mask = channel_data <= self.threshold
            else:
                st.error(f"Invalid direction '{self.direction}' for threshold gate")
                return None
            
            final_mask = valid_mask & threshold_mask
            return data[final_mask].copy()
        except Exception as e:
            st.error(f"Error applying threshold gate '{self.name}': {e}")
            return None

class GateManager:
    """Manager class for handling multiple gates - matches README specification"""
    
    def __init__(self):
        self.gates = {}
        self.gate_hierarchy = {}  # For storing parent-child relationships
    
    def create_rectangular_gate(self, name, x_channel, y_channel, x_min, x_max, y_min, y_max):
        """Create rectangular gate - matches README API"""
        try:
            gate = RectangleGate(name, x_channel, y_channel, x_min, x_max, y_min, y_max)
            self.add_gate(gate)
            return gate
        except Exception as e:
            st.error(f"Error creating rectangular gate '{name}': {e}")
            return None
    
    def create_polygon_gate(self, name, x_channel, y_channel, coordinates):
        """Create polygon gate - matches README API"""
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
    
    def create_ellipse_gate(self, name, x_channel, y_channel, center_x, center_y, width, height):
        """Create ellipse gate - matches README API"""
        try:
            gate = EllipseGate(name, x_channel, y_channel, center_x, center_y, width, height)
            self.add_gate(gate)
            return gate
        except Exception as e:
            st.error(f"Error creating ellipse gate '{name}': {e}")
            return None
    
    def create_threshold_gate(self, name, channel, threshold, direction):
        """Create threshold gate - matches README API"""
        try:
            gate = ThresholdGate(name, channel, threshold, direction)
            self.add_gate(gate)
            return gate
        except Exception as e:
            st.error(f"Error creating threshold gate '{name}': {e}")
            return None
    
    def add_gate(self, gate):
        """Add a gate to the manager"""
        if not isinstance(gate, Gate):
            raise ValueError("Object must be a Gate instance")
        
        self.gates[gate.name] = gate
        return gate
    
    def remove_gate(self, gate_name):
        """Remove a gate from the manager"""
        if gate_name in self.gates:
            del self.gates[gate_name]
            # Remove from hierarchy if exists
            if gate_name in self.gate_hierarchy:
                del self.gate_hierarchy[gate_name]
            return True
        else:
            st.error(f"Gate '{gate_name}' not found")
            return False
    
    def get_gate(self, gate_name):
        """Get a gate by name"""
        return self.gates.get(gate_name)
    
    def list_gates(self):
        """List all gates"""
        return list(self.gates.keys())
    
    def apply_gate(self, data, gate):
        """Apply gate to data - matches README API"""
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
    
    def add_gate_to_plot(self, fig, gate, gate_index):
        """Add gate visualization to plot - matches README API"""
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
    
    def parse_polygon_coordinates(self, coordinates_str):
        """Parse polygon coordinates from string - matches README API"""
        try:
            # Remove extra whitespace and normalize
            coords_str = coordinates_str.strip()
            
            # Try different formats
            # Format 1: "x1,y1 x2,y2 x3,y3"
            if ' ' in coords_str and ',' in coords_str:
                pairs = coords_str.split()
                vertices = []
                for pair in pairs:
                    if ',' in pair:
                        x, y = pair.split(',', 1)
                        vertices.append((float(x.strip()), float(y.strip())))
                return vertices
            
            # Format 2: "x1,y1;x2,y2;x3,y3"
            elif ';' in coords_str:
                pairs = coords_str.split(';')
                vertices = []
                for pair in pairs:
                    if ',' in pair:
                        x, y = pair.split(',', 1)
                        vertices.append((float(x.strip()), float(y.strip())))
                return vertices
            
            # Format 3: "(x1,y1),(x2,y2),(x3,y3)"
            elif '(' in coords_str and ')' in coords_str:
                # Extract coordinate pairs using regex
                pattern = r'\(([^)]+)\)'
                matches = re.findall(pattern, coords_str)
                vertices = []
                for match in matches:
                    if ',' in match:
                        x, y = match.split(',', 1)
                        vertices.append((float(x.strip()), float(y.strip())))
                return vertices
            
            else:
                st.error("Unsupported coordinate format. Use formats like: '1,2 3,4 5,6' or '1,2;3,4;5,6' or '(1,2),(3,4),(5,6)'")
                return None
                
        except Exception as e:
            st.error(f"Error parsing coordinates: {e}")
            return None
    
    def get_gate_statistics(self, gate_name, data):
        """Get statistics for a specific gate"""
        gate = self.get_gate(gate_name)
        if gate is None:
            return None
        
        return gate.get_statistics(data)
    
    def get_all_gate_statistics(self, data):
        """Get statistics for all gates"""
        stats = {}
        for gate_name, gate in self.gates.items():
            stats[gate_name] = gate.get_statistics(data)
        
        return stats
    
    def clear_gates(self):
        """Clear all gates"""
        self.gates.clear()
        self.gate_hierarchy.clear()
    
    def export_gates(self):
        """Export gate definitions for later use"""
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

# Utility functions for compatibility

def create_polygon_from_clicks(click_coordinates, min_points=3):
    """Create a polygon gate from mouse click coordinates"""
    if len(click_coordinates) < min_points:
        st.warning(f"Need at least {min_points} points to create a polygon gate")
        return None
    
    # Close the polygon by adding the first point at the end if not already closed
    vertices = click_coordinates.copy()
    if vertices[0] != vertices[-1]:
        vertices.append(vertices[0])
    
    return vertices

def calculate_convex_hull_gate(data, x_channel, y_channel, fraction=0.95):
    """Create a convex hull gate around a fraction of the data"""
    try:
        if x_channel not in data.columns or y_channel not in data.columns:
            st.error("Channels not found in data")
            return None
        
        # Get valid numeric data
        x_data = pd.to_numeric(data[x_channel], errors='coerce')
        y_data = pd.to_numeric(data[y_channel], errors='coerce')
        
        valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
        
        if not valid_mask.any():
            st.error("No valid data points found")
            return None
        
        points = np.column_stack([x_data[valid_mask], y_data[valid_mask]])
        
        if len(points) < 3:
            st.error("Not enough valid points for convex hull")
            return None
        
        # Calculate density and select points within fraction
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(points.T)
            density = kde(points.T)
            
            # Select points above threshold
            threshold = np.percentile(density, (1 - fraction) * 100)
            selected_points = points[density >= threshold]
        except:
            # Fallback: use random sampling if density calculation fails
            n_select = max(int(len(points) * fraction), 10)
            selected_indices = np.random.choice(len(points), min(n_select, len(points)), replace=False)
            selected_points = points[selected_indices]
        
        if len(selected_points) < 3:
            st.error("Not enough points for convex hull after filtering")
            return None
        
        # Calculate convex hull
        hull = ConvexHull(selected_points)
        vertices = selected_points[hull.vertices].tolist()
        return vertices
        
    except Exception as e:
        st.error(f"Error calculating convex hull: {e}")
        return None
