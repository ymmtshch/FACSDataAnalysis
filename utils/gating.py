# utils/gating.py - Gating utilities for FACS Data Analysis
# Updated for fcsparser migration from flowkit

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.geometry.polygon import LinearRing
from scipy.spatial import ConvexHull
import streamlit as st
from config import GATING_CONFIG

class Gate:
    """Base class for all gate types"""
    
    def __init__(self, name, gate_type, channels, color='red'):
        self.name = name
        self.gate_type = gate_type
        self.channels = channels
        self.color = color
        self.created_at = pd.Timestamp.now()
    
    def apply(self, data):
        """Apply gate to data - to be implemented by subclasses"""
        raise NotImplementedError
    
    def get_statistics(self, data):
        """Get statistics for gated data"""
        gated_data = self.apply(data)
        if gated_data is None or len(gated_data) == 0:
            return None
        
        total_events = len(data)
        gated_events = len(gated_data)
        percentage = (gated_events / total_events) * 100 if total_events > 0 else 0
        
        return {
            'total_events': total_events,
            'gated_events': gated_events,
            'percentage': percentage,
            'gate_name': self.name,
            'gate_type': self.gate_type,
            'channels': self.channels
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
                self.polygon = Polygon(self.vertices)
            except Exception as e:
                st.error(f"Error creating polygon: {e}")
                self.polygon = None
    
    def apply(self, data):
        """Apply polygon gate to data"""
        if self.polygon is None:
            return None
        
        if self.x_channel not in data.columns or self.y_channel not in data.columns:
            st.error(f"Channels {self.x_channel} or {self.y_channel} not found in data")
            return None
        
        # Create points from data
        points = list(zip(data[self.x_channel], data[self.y_channel]))
        
        # Check which points are inside the polygon
        mask = [self.polygon.contains(Point(point)) for point in points]
        
        return data[mask]
    
    def get_vertices_array(self):
        """Get vertices as numpy array"""
        return np.array(self.vertices)

class RectangleGate(Gate):
    """Rectangle gate for 2D data"""
    
    def __init__(self, name, x_channel, y_channel, x_min, x_max, y_min, y_max, color='blue'):
        super().__init__(name, 'rectangle', [x_channel, y_channel], color)
        self.x_channel = x_channel
        self.y_channel = y_channel
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
    
    def apply(self, data):
        """Apply rectangle gate to data"""
        if self.x_channel not in data.columns or self.y_channel not in data.columns:
            st.error(f"Channels {self.x_channel} or {self.y_channel} not found in data")
            return None
        
        mask = (
            (data[self.x_channel] >= self.x_min) &
            (data[self.x_channel] <= self.x_max) &
            (data[self.y_channel] >= self.y_min) &
            (data[self.y_channel] <= self.y_max)
        )
        
        return data[mask]
    
    def get_rectangle_coords(self):
        """Get rectangle coordinates for plotting"""
        return {
            'x': [self.x_min, self.x_max, self.x_max, self.x_min, self.x_min],
            'y': [self.y_min, self.y_min, self.y_max, self.y_max, self.y_min]
        }

class EllipseGate(Gate):
    """Ellipse gate for 2D data"""
    
    def __init__(self, name, x_channel, y_channel, center_x, center_y, 
                 width, height, angle=0, color='green'):
        super().__init__(name, 'ellipse', [x_channel, y_channel], color)
        self.x_channel = x_channel
        self.y_channel = y_channel
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height
        self.angle = angle  # rotation angle in degrees
    
    def apply(self, data):
        """Apply ellipse gate to data"""
        if self.x_channel not in data.columns or self.y_channel not in data.columns:
            st.error(f"Channels {self.x_channel} or {self.y_channel} not found in data")
            return None
        
        # Translate points to ellipse center
        x_translated = data[self.x_channel] - self.center_x
        y_translated = data[self.y_channel] - self.center_y
        
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
        mask = ellipse_eq <= 1
        
        return data[mask]
    
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
    """Threshold gate for 1D data"""
    
    def __init__(self, name, channel, threshold, direction='above', color='orange'):
        super().__init__(name, 'threshold', [channel], color)
        self.channel = channel
        self.threshold = threshold
        self.direction = direction  # 'above' or 'below'
    
    def apply(self, data):
        """Apply threshold gate to data"""
        if self.channel not in data.columns:
            st.error(f"Channel {self.channel} not found in data")
            return None
        
        if self.direction == 'above':
            mask = data[self.channel] >= self.threshold
        else:  # below
            mask = data[self.channel] <= self.threshold
        
        return data[mask]

class QuadrantGate(Gate):
    """Quadrant gate for 2D data"""
    
    def __init__(self, name, x_channel, y_channel, x_threshold, y_threshold, color='purple'):
        super().__init__(name, 'quadrant', [x_channel, y_channel], color)
        self.x_channel = x_channel
        self.y_channel = y_channel
        self.x_threshold = x_threshold
        self.y_threshold = y_threshold
    
    def apply(self, data, quadrant='all'):
        """Apply quadrant gate to data
        
        Parameters:
        - quadrant: 'all', 'Q1' (upper right), 'Q2' (upper left), 
                   'Q3' (lower left), 'Q4' (lower right)
        """
        if self.x_channel not in data.columns or self.y_channel not in data.columns:
            st.error(f"Channels {self.x_channel} or {self.y_channel} not found in data")
            return None
        
        x_data = data[self.x_channel]
        y_data = data[self.y_channel]
        
        if quadrant == 'all':
            return data
        elif quadrant == 'Q1':  # Upper right
            mask = (x_data >= self.x_threshold) & (y_data >= self.y_threshold)
        elif quadrant == 'Q2':  # Upper left
            mask = (x_data < self.x_threshold) & (y_data >= self.y_threshold)
        elif quadrant == 'Q3':  # Lower left
            mask = (x_data < self.x_threshold) & (y_data < self.y_threshold)
        elif quadrant == 'Q4':  # Lower right
            mask = (x_data >= self.x_threshold) & (y_data < self.y_threshold)
        else:
            st.error(f"Invalid quadrant: {quadrant}")
            return None
        
        return data[mask]
    
    def get_quadrant_statistics(self, data):
        """Get statistics for all quadrants"""
        total_events = len(data)
        stats = {}
        
        for quad in ['Q1', 'Q2', 'Q3', 'Q4']:
            quad_data = self.apply(data, quad)
            if quad_data is not None:
                count = len(quad_data)
                percentage = (count / total_events) * 100 if total_events > 0 else 0
                stats[quad] = {
                    'count': count,
                    'percentage': percentage
                }
        
        return stats

class GateManager:
    """Manager class for handling multiple gates"""
    
    def __init__(self):
        self.gates = {}
        self.gate_hierarchy = {}  # For storing parent-child relationships
    
    def add_gate(self, gate):
        """Add a gate to the manager"""
        if not isinstance(gate, Gate):
            raise ValueError("Object must be a Gate instance")
        
        self.gates[gate.name] = gate
        st.success(f"Gate '{gate.name}' added successfully")
    
    def remove_gate(self, gate_name):
        """Remove a gate from the manager"""
        if gate_name in self.gates:
            del self.gates[gate_name]
            # Remove from hierarchy if exists
            if gate_name in self.gate_hierarchy:
                del self.gate_hierarchy[gate_name]
            st.success(f"Gate '{gate_name}' removed successfully")
        else:
            st.error(f"Gate '{gate_name}' not found")
    
    def get_gate(self, gate_name):
        """Get a gate by name"""
        return self.gates.get(gate_name)
    
    def list_gates(self):
        """List all gates"""
        return list(self.gates.keys())
    
    def apply_gate(self, gate_name, data):
        """Apply a specific gate to data"""
        gate = self.get_gate(gate_name)
        if gate is None:
            st.error(f"Gate '{gate_name}' not found")
            return None
        
        return gate.apply(data)
    
    def apply_gate_sequence(self, gate_names, data):
        """Apply a sequence of gates to data"""
        current_data = data
        
        for gate_name in gate_names:
            gate = self.get_gate(gate_name)
            if gate is None:
                st.error(f"Gate '{gate_name}' not found")
                return None
            
            current_data = gate.apply(current_data)
            if current_data is None or len(current_data) == 0:
                st.warning(f"No events remaining after applying gate '{gate_name}'")
                break
        
        return current_data
    
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
            elif isinstance(gate, QuadrantGate):
                gate_def.update({
                    'x_threshold': gate.x_threshold,
                    'y_threshold': gate.y_threshold
                })
            
            gate_definitions[name] = gate_def
        
        return gate_definitions
    
    def import_gates(self, gate_definitions):
        """Import gate definitions"""
        for name, gate_def in gate_definitions.items():
            try:
                if gate_def['type'] == 'polygon':
                    gate = PolygonGate(
                        name=gate_def['name'],
                        x_channel=gate_def['channels'][0],
                        y_channel=gate_def['channels'][1],
                        vertices=gate_def['vertices'],
                        color=gate_def.get('color', 'red')
                    )
                elif gate_def['type'] == 'rectangle':
                    gate = RectangleGate(
                        name=gate_def['name'],
                        x_channel=gate_def['channels'][0],
                        y_channel=gate_def['channels'][1],
                        x_min=gate_def['x_min'],
                        x_max=gate_def['x_max'],
                        y_min=gate_def['y_min'],
                        y_max=gate_def['y_max'],
                        color=gate_def.get('color', 'blue')
                    )
                elif gate_def['type'] == 'ellipse':
                    gate = EllipseGate(
                        name=gate_def['name'],
                        x_channel=gate_def['channels'][0],
                        y_channel=gate_def['channels'][1],
                        center_x=gate_def['center_x'],
                        center_y=gate_def['center_y'],
                        width=gate_def['width'],
                        height=gate_def['height'],
                        angle=gate_def.get('angle', 0),
                        color=gate_def.get('color', 'green')
                    )
                elif gate_def['type'] == 'threshold':
                    gate = ThresholdGate(
                        name=gate_def['name'],
                        channel=gate_def['channels'][0],
                        threshold=gate_def['threshold'],
                        direction=gate_def.get('direction', 'above'),
                        color=gate_def.get('color', 'orange')
                    )
                elif gate_def['type'] == 'quadrant':
                    gate = QuadrantGate(
                        name=gate_def['name'],
                        x_channel=gate_def['channels'][0],
                        y_channel=gate_def['channels'][1],
                        x_threshold=gate_def['x_threshold'],
                        y_threshold=gate_def['y_threshold'],
                        color=gate_def.get('color', 'purple')
                    )
                else:
                    st.error(f"Unknown gate type: {gate_def['type']}")
                    continue
                
                self.add_gate(gate)
                
            except Exception as e:
                st.error(f"Error importing gate '{name}': {e}")

def create_polygon_from_clicks(click_coordinates, min_points=3):
    """Create a polygon gate from mouse click coordinates"""
    if len(click_coordinates) < min_points:
        st.warning(f"Need at least {min_points} points to create a polygon gate")
        return None
    
    # Close the polygon by adding the first point at the end
    vertices = click_coordinates + [click_coordinates[0]]
    
    return vertices

def calculate_convex_hull_gate(data, x_channel, y_channel, fraction=0.95):
    """Create a convex hull gate around a fraction of the data"""
    if x_channel not in data.columns or y_channel not in data.columns:
        st.error("Channels not found in data")
        return None
    
    # Sample data points
    points = np.column_stack([data[x_channel], data[y_channel]])
    
    # Calculate density and select points within fraction
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(points.T)
    density = kde(points.T)
    
    # Select points above threshold
    threshold = np.percentile(density, (1 - fraction) * 100)
    selected_points = points[density >= threshold]
    
    if len(selected_points) < 3:
        st.error("Not enough points for convex hull")
        return None
    
    # Calculate convex hull
    try:
        hull = ConvexHull(selected_points)
        vertices = selected_points[hull.vertices].tolist()
        return vertices
    except Exception as e:
        st.error(f"Error calculating convex hull: {e}")
        return None
