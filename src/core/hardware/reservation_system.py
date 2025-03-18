"""
Hardware Reservation System

Provides mechanisms for reserving neuromorphic hardware resources in multi-user environments.
"""

from typing import Dict, Any, List, Optional, Set, Tuple
import threading
import time
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from src.core.utils.logging_framework import get_logger
from src.core.hardware.resource_sharing import ResourceType, get_resource_manager
from src.core.hardware.exceptions import ResourceSharingError

logger = get_logger("reservation_system")


class ReservationStatus(Enum):
    """Status of a hardware reservation."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class HardwareReservation:
    """Hardware reservation details."""
    reservation_id: str
    user_id: str
    hardware_type: str
    resources: Dict[ResourceType, int]
    start_time: float
    end_time: float
    status: ReservationStatus = ReservationStatus.PENDING
    allocation_ids: List[str] = []


class ReservationSystem:
    """Manages hardware reservations for multi-user environments."""
    
    _instance = None
    _lock = threading.RLock()
    
    @classmethod
    def get_instance(cls) -> 'ReservationSystem':
        """Get singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = ReservationSystem()
            return cls._instance
    
    def __init__(self):
        """Initialize reservation system."""
        self.reservations: Dict[str, HardwareReservation] = {}
        self.user_reservations: Dict[str, Set[str]] = {}
        self.hardware_schedule: Dict[str, List[HardwareReservation]] = {}
        self.lock = threading.RLock()
        
        # Start background thread for reservation management
        self.running = True
        self.manager_thread = threading.Thread(target=self._manage_reservations, daemon=True)
        self.manager_thread.start()
    
    def reserve_hardware(self, user_id: str, hardware_type: str, 
                        resources: Dict[str, int], duration_minutes: int,
                        start_time: Optional[float] = None) -> str:
        """
        Reserve hardware resources.
        
        Args:
            user_id: User ID
            hardware_type: Hardware type
            resources: Resource requirements (type -> quantity)
            duration_minutes: Duration in minutes
            start_time: Optional start time (timestamp), defaults to now
            
        Returns:
            str: Reservation ID
            
        Raises:
            ResourceSharingError: If reservation fails
        """
        with self.lock:
            # Convert string resource types to enum
            resource_dict = {}
            for res_type_str, quantity in resources.items():
                try:
                    res_type = ResourceType(res_type_str)
                    resource_dict[res_type] = quantity
                except ValueError:
                    raise ValueError(f"Invalid resource type: {res_type_str}")
            
            # Set start and end times
            if start_time is None:
                start_time = time.time()
            end_time = start_time + (duration_minutes * 60)
            
            # Check if resources are available for the requested time period
            if not self._check_availability(hardware_type, resource_dict, start_time, end_time):
                raise ResourceSharingError(
                    f"Resources not available for {hardware_type} during requested time period")
            
            # Create reservation
            reservation_id = str(uuid.uuid4())
            reservation = HardwareReservation(
                reservation_id=reservation_id,
                user_id=user_id,
                hardware_type=hardware_type,
                resources=resource_dict,
                start_time=start_time,
                end_time=end_time,
                status=ReservationStatus.PENDING,
                allocation_ids=[]
            )
            
            # Store reservation
            self.reservations[reservation_id] = reservation
            
            # Track user reservations
            if user_id not in self.user_reservations:
                self.user_reservations[user_id] = set()
            self.user_reservations[user_id].add(reservation_id)
            
            # Add to hardware schedule
            if hardware_type not in self.hardware_schedule:
                self.hardware_schedule[hardware_type] = []
            self.hardware_schedule[hardware_type].append(reservation)
            
            logger.info(f"Created reservation {reservation_id} for user {user_id} on {hardware_type}")
            
            return reservation_id
    
    def cancel_reservation(self, reservation_id: str, user_id: str) -> bool:
        """
        Cancel a hardware reservation.
        
        Args:
            reservation_id: Reservation ID
            user_id: User ID (for authorization)
            
        Returns:
            bool: Success status
        """
        with self.lock:
            if reservation_id not in self.reservations:
                return False
            
            reservation = self.reservations[reservation_id]
            
            # Check authorization
            if reservation.user_id != user_id:
                logger.warning(f"User {user_id} attempted to cancel reservation {reservation_id} owned by {reservation.user_id}")
                return False
            
            # Only pending or active reservations can be cancelled
            if reservation.status not in [ReservationStatus.PENDING, ReservationStatus.ACTIVE]:
                return False
            
            # Release resources if active
            if reservation.status == ReservationStatus.ACTIVE:
                self._release_reservation_resources(reservation)
            
            # Update status
            reservation.status = ReservationStatus.CANCELLED
            
            logger.info(f"Cancelled reservation {reservation_id} for user {user_id}")
            
            return True
    
    def get_reservation(self, reservation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get reservation details.
        
        Args:
            reservation_id: Reservation ID
            
        Returns:
            Optional[Dict[str, Any]]: Reservation details
        """
        with self.lock:
            if reservation_id not in self.reservations:
                return None
            
            reservation = self.reservations[reservation_id]
            
            # Convert to dictionary
            return {
                "reservation_id": reservation.reservation_id,
                "user_id": reservation.user_id,
                "hardware_type": reservation.hardware_type,
                "resources": {rt.value: qty for rt, qty in reservation.resources.items()},
                "start_time": reservation.start_time,
                "end_time": reservation.end_time,
                "status": reservation.status.value,
                "start_time_str": datetime.fromtimestamp(reservation.start_time).strftime("%Y-%m-%d %H:%M:%S"),
                "end_time_str": datetime.fromtimestamp(reservation.end_time).strftime("%Y-%m-%d %H:%M:%S"),
                "duration_minutes": int((reservation.end_time - reservation.start_time) / 60)
            }
    
    def list_user_reservations(self, user_id: str) -> List[Dict[str, Any]]:
        """
        List reservations for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List[Dict[str, Any]]: List of reservation details
        """
        with self.lock:
            if user_id not in self.user_reservations:
                return []
            
            result = []
            for res_id in self.user_reservations[user_id]:
                reservation_data = self.get_reservation(res_id)
                if reservation_data is not None:
                    result.append(reservation_data)
            
            return result
    
    def list_hardware_reservations(self, hardware_type: str) -> List[Dict[str, Any]]:
        """
        List reservations for a hardware type.
        
        Args:
            hardware_type: Hardware type
            
        Returns:
            List[Dict[str, Any]]: List of reservation details
        """
        with self.lock:
            if hardware_type not in self.hardware_schedule:
                return []
            
            result = []
            for res in self.hardware_schedule[hardware_type]:
                reservation_data = self.get_reservation(res.reservation_id)
                if reservation_data is not None:
                    result.append(reservation_data)
            
            return result
    
    def _check_availability(self, hardware_type: str, resources: Dict[ResourceType, int],
                          start_time: float, end_time: float) -> bool:
        """Check if resources are available for the requested time period."""
        if hardware_type not in self.hardware_schedule:
            return True
        
        # Get overlapping reservations
        overlapping = [
            r for r in self.hardware_schedule[hardware_type]
            if r.status in [ReservationStatus.PENDING, ReservationStatus.ACTIVE]
            and r.start_time < end_time and r.end_time > start_time
        ]
        
        # Check resource availability
        resource_manager = get_resource_manager()
        pool = resource_manager.get_pool(hardware_type)
        
        if not pool:
            # No pool means no limits defined yet, assume available
            return True
        
        # Check each resource type
        for res_type, quantity in resources.items():
            # Calculate total reserved during the period
            reserved = sum(
                r.resources.get(res_type, 0)
                for r in overlapping
            )
            
            # Check against limits
            limit = pool.resource_limits.get(res_type, 0)
            if reserved + quantity > limit:
                return False
        
        return True
    
    def _manage_reservations(self):
        """Background thread to manage reservations."""
        while self.running:
            try:
                now = time.time()
                
                with self.lock:
                    # Activate pending reservations
                    pending = [
                        r for r in self.reservations.values()
                        if r.status == ReservationStatus.PENDING and r.start_time <= now
                    ]
                    
                    for reservation in pending:
                        self._activate_reservation(reservation)
                    
                    # Complete active reservations
                    active = [
                        r for r in self.reservations.values()
                        if r.status == ReservationStatus.ACTIVE and r.end_time <= now
                    ]
                    
                    for reservation in active:
                        self._complete_reservation(reservation)
                
                # Sleep for a short time
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in reservation manager: {str(e)}")
                time.sleep(5)
    
    def _activate_reservation(self, reservation: HardwareReservation):
        """Activate a pending reservation."""
        try:
            # Allocate resources
            resource_manager = get_resource_manager()
            
            allocation_ids = []
            for res_type, quantity in reservation.resources.items():
                allocation_id = resource_manager.allocate_resource(
                    reservation.user_id,
                    reservation.hardware_type,
                    res_type,
                    quantity,
                    reservation.end_time - time.time()
                )
                allocation_ids.append(allocation_id)
            
            # Update reservation
            reservation.allocation_ids = allocation_ids
            reservation.status = ReservationStatus.ACTIVE
            
            logger.info(f"Activated reservation {reservation.reservation_id} for user {reservation.user_id}")
        except Exception as e:
            logger.error(f"Failed to activate reservation {reservation.reservation_id}: {str(e)}")
            reservation.status = ReservationStatus.FAILED
    
    def _complete_reservation(self, reservation: HardwareReservation):
        """Complete an active reservation."""
        # Release resources
        self._release_reservation_resources(reservation)
        
        # Update status
        reservation.status = ReservationStatus.COMPLETED
        
        logger.info(f"Completed reservation {reservation.reservation_id} for user {reservation.user_id}")
    
    def _release_reservation_resources(self, reservation: HardwareReservation):
        """Release resources for a reservation."""
        if not reservation.allocation_ids:
            return
        
        resource_manager = get_resource_manager()
        
        for allocation_id in reservation.allocation_ids:
            try:
                resource_manager.release_resource(allocation_id, reservation.hardware_type)
            except Exception as e:
                logger.error(f"Error releasing allocation {allocation_id}: {str(e)}")
        
        reservation.allocation_ids = []


# Create global instance
reservation_system = ReservationSystem.get_instance()


def get_reservation_system() -> ReservationSystem:
    """Get the global reservation system."""
    return reservation_system