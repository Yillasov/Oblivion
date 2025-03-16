"""
Simulation Visualization Tools

Provides real-time and post-simulation visualization capabilities.
"""

import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pygame
from typing import Dict, Any

from src.simulation.core.recorder import PlaybackSystem, SimulationState
from src.simulation.core.scheduler import SimulationScheduler

class DataVisualizer:
    """Visualizes recorded simulation data using Matplotlib."""
    
    def __init__(self, playback: PlaybackSystem):
        self.playback = playback
        self.frames = sorted(playback.recorded_data.keys())
        
    def plot_altitude(self):
        """Plot altitude data from recorded simulation."""
        times = []
        agl = []
        msl = []
        
        for frame in self.frames:
            state = self.playback.recorded_data[frame]
            if 'altimeter' in state.state_data.get('sensors', {}):
                times.append(state.sim_time)
                agl.append(state.state_data['sensors']['altimeter']['altitude_agl'])
                msl.append(state.state_data['sensors']['altimeter']['altitude_msl'])
        
        plt.figure(figsize=(10, 5))
        plt.plot(times, agl, label='AGL')
        plt.plot(times, msl, label='MSL')
        plt.xlabel('Simulation Time (s)')
        plt.ylabel('Altitude (m)')
        plt.title('Altitude Profile')
        plt.legend()
        plt.grid(True)
        plt.show()

class RealTimeMonitor(QtWidgets.QMainWindow):
    """GUI-based real-time simulation monitor using PyQt."""
    
    def __init__(self, scheduler: SimulationScheduler):
        super().__init__()
        self.scheduler = scheduler
        self.init_ui()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(100)  # Update every 100ms
        
    def init_ui(self):
        self.setWindowTitle('Simulation Monitor')
        self.setGeometry(100, 100, 400, 300)
        
        # Create central widget
        self.central = QtWidgets.QWidget()
        self.setCentralWidget(self.central)
        
        # Layout
        layout = QtWidgets.QVBoxLayout()
        
        # System stats
        self.stats_label = QtWidgets.QLabel()
        layout.addWidget(self.stats_label)
        
        # Sensor data
        self.sensor_list = QtWidgets.QListWidget()
        layout.addWidget(self.sensor_list)
        
        self.central.setLayout(layout)
    
    def update_display(self):
        stats = self.scheduler.get_statistics()
        self.stats_label.setText(
            f"Time: {stats['sim_time']:.1f}s | FPS: {stats['fps']:.1f}\n"
            f"Tasks: {stats['task_count']} | State: {stats['state']}"
        )
        
        # Update sensor list
        self.sensor_list.clear()
        sensors = self.scheduler.get_task_statistics()
        for name, data in sensors.items():
            if data['group'] == 'sensors':
                self.sensor_list.addItem(
                    f"{name}: {data['actual_rate']:.1f}Hz | "
                    f"Avg Exec: {data['average_execution_time']*1000:.2f}ms"
                )

class Simulation3DView:
    """Basic 3D visualization using Pygame."""
    
    def __init__(self, playback: PlaybackSystem):
        self.playback = playback
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        self.clock = pygame.time.Clock()
        
    def render_frame(self, frame: int):
        state = self.playback.recorded_data[frame]
        pos = state.state_data.get('platform', {}).get('position', [0, 0, 0])
        
        self.screen.fill((0, 0, 0))
        
        # Convert 3D position to 2D view (simple projection)
        x = int(pos[0] * 0.1 + 400)
        y = int(-pos[1] * 0.1 + 300)
        size = max(5, int(pos[2] * 0.01))
        
        # Draw platform
        pygame.draw.circle(self.screen, (0, 255, 0), (x, y), size)
        
        pygame.display.flip()
        self.clock.tick(30)
    
    def run_visualization(self):
        for frame in sorted(self.playback.recorded_data.keys()):
            self.render_frame(frame)