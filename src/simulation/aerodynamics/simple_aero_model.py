#!/usr/bin/env python3
"""
Simple Aerodynamic Model

A lightweight aerodynamic model for quick UCAV simulations.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from typing import Dict, Tuple, Any, Optional
import logging

from src.core.utils.logging_framework import get_logger
from src.simulation.aerodynamics.ucav_model import UCAVGeometry

logger = get_logger("simple_aero")


class SimpleAerodynamicModel:
    """
    Simplified aerodynamic model using lookup tables and basic equations.
    """
    
    def __init__(self, geometry: UCAVGeometry):
        """
        Initialize the simple aerodynamic model.
        
        Args:
            geometry: UCAV geometric parameters
        """
        self.geometry = geometry
        
        # Create simple lookup tables for aerodynamic coefficients
        self._create_lookup_tables()
        
        logger.info(f"Simple aerodynamic model initialized for UCAV with wingspan {geometry.wingspan}m")
    
    def _create_lookup_tables(self):
        """Create simplified lookup tables for aerodynamic coefficients."""
        # Angle of attack range (degrees)
        self.alpha_range = np.linspace(-10, 20, 16)
        
        # Mach number range
        self.mach_range = np.linspace(0.1, 1.5, 15)
        
        # Basic lift coefficient table (alpha, mach)
        self.cl_table = np.zeros((len(self.alpha_range), len(self.mach_range)))
        for i, alpha in enumerate(self.alpha_range):
            for j, mach in enumerate(self.mach_range):
                # Simple linear model with compressibility effects
                cl_base = 0.1 * alpha
                if mach < 1.0:
                    cl_factor = 1.0 / np.sqrt(1.0 - min(0.95, mach)**2)
                else:
                    cl_factor = 0.7  # Supersonic lift reduction
                
                self.cl_table[i, j] = cl_base * cl_factor
        
        # Basic drag coefficient table (alpha, mach)
        self.cd_table = np.zeros((len(self.alpha_range), len(self.mach_range)))
        for i, alpha in enumerate(self.alpha_range):
            for j, mach in enumerate(self.mach_range):
                # Parasitic drag + induced drag + wave drag
                cd0 = 0.015
                cdi = 0.05 * alpha**2 / self.geometry.aspect_ratio
                
                # Wave drag
                if mach > 0.8:
                    cdw = 0.1 * (mach - 0.8)**2
                else:
                    cdw = 0.0
                
                self.cd_table[i, j] = cd0 + cdi + cdw
    
    def _interpolate_coefficient(self, table, alpha, mach):
        """
        Interpolate a coefficient from a lookup table.
        
        Args:
            table: Coefficient table
            alpha: Angle of attack (degrees)
            mach: Mach number
            
        Returns:
            float: Interpolated coefficient
        """
        # Clamp values to table ranges
        alpha = max(self.alpha_range[0], min(alpha, self.alpha_range[-1]))
        mach = max(self.mach_range[0], min(mach, self.mach_range[-1]))
        
        # Find indices for interpolation
        alpha_idx = np.searchsorted(self.alpha_range, alpha) - 1
        mach_idx = np.searchsorted(self.mach_range, mach) - 1
        
        # Ensure valid indices
        alpha_idx = max(0, min(alpha_idx, len(self.alpha_range) - 2))
        mach_idx = max(0, min(mach_idx, len(self.mach_range) - 2))
        
        # Calculate interpolation weights
        alpha_weight = (alpha - self.alpha_range[alpha_idx]) / (self.alpha_range[alpha_idx + 1] - self.alpha_range[alpha_idx])
        mach_weight = (mach - self.mach_range[mach_idx]) / (self.mach_range[mach_idx + 1] - self.mach_range[mach_idx])
        
        # Bilinear interpolation
        c00 = table[alpha_idx, mach_idx]
        c01 = table[alpha_idx, mach_idx + 1]
        c10 = table[alpha_idx + 1, mach_idx]
        c11 = table[alpha_idx + 1, mach_idx + 1]
        
        c0 = c00 * (1 - mach_weight) + c01 * mach_weight
        c1 = c10 * (1 - mach_weight) + c11 * mach_weight
        
        return c0 * (1 - alpha_weight) + c1 * alpha_weight
    
    def calculate_coefficients(self, alpha: float, beta: float, mach: float) -> Dict[str, float]:
        """
        Calculate aerodynamic coefficients using lookup tables.
        
        Args:
            alpha: Angle of attack (degrees)
            beta: Sideslip angle (degrees)
            mach: Mach number
            
        Returns:
            Dict[str, float]: Dictionary of aerodynamic coefficients
        """
        # Get basic coefficients from tables
        cl = self._interpolate_coefficient(self.cl_table, alpha, mach)
        cd = self._interpolate_coefficient(self.cd_table, alpha, mach)
        
        # Side force (simplified linear model)
        cy = -0.02 * beta
        
        # Pitching moment (simplified)
        cm = -0.01 - 0.05 * alpha
        
        # Rolling moment due to sideslip
        cl_roll = -0.01 * beta
        
        # Yawing moment due to sideslip
        cn = 0.005 * beta
        
        return {
            'CL': cl,
            'CD': cd,
            'CY': cy,
            'Cm': cm,
            'Cl': cl_roll,
            'Cn': cn
        }
    
    def calculate_forces(self, velocity: float, altitude: float, 
                        alpha: float, beta: float) -> Dict[str, float]:
        """
        Calculate aerodynamic forces and moments.
        
        Args:
            velocity: Airspeed (m/s)
            altitude: Altitude (m)
            alpha: Angle of attack (degrees)
            beta: Sideslip angle (degrees)
            
        Returns:
            Dict[str, float]: Dictionary of forces and moments
        """
        # Simple atmospheric model
        if altitude < 11000:  # Troposphere
            temperature = 288.15 - 0.0065 * altitude
            pressure = 101325 * (temperature / 288.15) ** 5.255
        else:  # Stratosphere (simplified)
            temperature = 216.65
            pressure = 22632 * np.exp(-0.00015769 * (altitude - 11000))
        
        # Air density
        rho = pressure / (287.05 * temperature)
        
        # Dynamic pressure
        q = 0.5 * rho * velocity ** 2
        
        # Mach number
        speed_of_sound = np.sqrt(1.4 * 287.05 * temperature)
        mach = velocity / speed_of_sound
        
        # Get coefficients
        coeffs = self.calculate_coefficients(alpha, beta, mach)
        
        # Calculate forces and moments
        forces = {
            'lift': coeffs['CL'] * q * self.geometry.wing_area,
            'drag': coeffs['CD'] * q * self.geometry.wing_area,
            'side_force': coeffs['CY'] * q * self.geometry.wing_area,
            'pitch_moment': coeffs['Cm'] * q * self.geometry.wing_area * self.geometry.mean_chord,
            'roll_moment': coeffs['Cl'] * q * self.geometry.wing_area * self.geometry.wingspan,
            'yaw_moment': coeffs['Cn'] * q * self.geometry.wing_area * self.geometry.wingspan
        }
        
        # Add stability derivatives (simplified)
        # Pitch damping
        pitch_rate = 0.0  # Would come from flight dynamics
        forces['pitch_moment'] += -0.5 * q * self.geometry.wing_area * self.geometry.mean_chord * pitch_rate
        
        return forces
    
    def estimate_performance(self, weight: float, altitude: float) -> Dict[str, float]:
        """
        Estimate basic aircraft performance metrics.
        
        Args:
            weight: Aircraft weight (N)
            altitude: Altitude (m)
            
        Returns:
            Dict[str, float]: Performance metrics
        """
        # Simple atmospheric model for density
        rho = 1.225 * np.exp(-altitude / 10000)
        
        # Estimate stall speed
        cl_max = 1.2
        v_stall = np.sqrt((2 * weight) / (rho * self.geometry.wing_area * cl_max))
        
        # Estimate minimum drag speed
        cd0 = 0.015
        k = 1 / (np.pi * self.geometry.aspect_ratio * 0.85)
        v_min_drag = np.sqrt((2 * weight) / (rho * self.geometry.wing_area) * np.sqrt(cd0 / k))
        
        # Estimate maximum L/D
        max_ld = 0.5 * np.sqrt(1 / (cd0 * k))
        
        # Estimate ceiling (simplified)
        ceiling = -10000 * np.log(0.7 * cd0 * self.geometry.wing_area * 1.225 / weight)
        ceiling = min(ceiling, 15000)  # Reasonable limit
        
        return {
            'stall_speed': v_stall,
            'min_drag_speed': v_min_drag,
            'max_ld': max_ld,
            'ceiling': ceiling
        }