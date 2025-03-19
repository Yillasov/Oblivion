"""
Memory Management System for Neuromorphic Hardware

This module provides memory management capabilities for neuromorphic processors,
handling allocation, deallocation, and optimization of memory resources.
"""

from typing import Dict, List, Tuple, Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


class MemoryBlock:
    """Represents a block of memory on a neuromorphic processor."""
    
    def __init__(self, start_address: int, size: int, block_type: str):
        """
        Initialize a memory block.
        
        Args:
            start_address: Starting address of the memory block
            size: Size of the memory block in bytes
            block_type: Type of memory block ('neuron', 'synapse', 'weight', etc.)
        """
        self.start_address = start_address
        self.size = size
        self.block_type = block_type
        self.in_use = False
        self.owner_id = None  # ID of the component using this block


class MemoryRegion:
    """Represents a memory region on a neuromorphic processor."""
    
    def __init__(self, region_id: int, start_address: int, size: int, region_type: str):
        """
        Initialize a memory region.
        
        Args:
            region_id: Unique identifier for the region
            start_address: Starting address of the region
            size: Size of the region in bytes
            region_type: Type of memory region ('core', 'shared', 'external', etc.)
        """
        self.region_id = region_id
        self.start_address = start_address
        self.size = size
        self.region_type = region_type
        self.blocks: List[MemoryBlock] = []
        self.free_space = size


class NeuromorphicMemoryManager:
    """
    Memory manager for neuromorphic hardware.
    
    Handles allocation, deallocation, and optimization of memory resources
    on neuromorphic processors.
    """
    
    def __init__(self):
        """Initialize the memory manager."""
        self.regions: Dict[int, MemoryRegion] = {}
        self.allocated_blocks: Dict[int, MemoryBlock] = {}
        self.next_block_id = 0
        self.power_mode = "balanced"  # Default power mode
    
    def add_memory_region(self, start_address: int, size: int, 
                          region_type: str) -> int:
        """
        Add a memory region to the manager.
        
        Args:
            start_address: Starting address of the region
            size: Size of the region in bytes
            region_type: Type of memory region ('core', 'shared', 'external', etc.)
            
        Returns:
            int: Region ID
        """
        region_id = len(self.regions)
        region = MemoryRegion(region_id, start_address, size, region_type)
        self.regions[region_id] = region
        logger.info(f"Added memory region {region_id}: {region_type}, {size} bytes")
        return region_id
    
    def allocate_memory(self, size: int, block_type: str, 
                        region_id: Optional[int] = None) -> Optional[int]:
        """
        Allocate a block of memory.
        
        Args:
            size: Size of the block to allocate in bytes
            block_type: Type of memory block ('neuron', 'synapse', 'weight', etc.)
            region_id: Optional region ID to allocate from (if None, will search all regions)
            
        Returns:
            Optional[int]: Block ID if allocation successful, None otherwise
        """
        # If region_id is specified, only search that region
        if region_id is not None:
            if region_id not in self.regions:
                logger.error(f"Region {region_id} does not exist")
                return None
            
            return self._allocate_in_region(self.regions[region_id], size, block_type)
        
        # Otherwise, search all regions for space
        for region in self.regions.values():
            block_id = self._allocate_in_region(region, size, block_type)
            if block_id is not None:
                return block_id
        
        logger.error(f"Failed to allocate {size} bytes for {block_type}")
        return None
    
    def _allocate_in_region(self, region: MemoryRegion, size: int, 
                           block_type: str) -> Optional[int]:
        """
        Allocate memory within a specific region.
        
        Args:
            region: Memory region to allocate from
            size: Size of the block to allocate in bytes
            block_type: Type of memory block
            
        Returns:
            Optional[int]: Block ID if allocation successful, None otherwise
        """
        if region.free_space < size:
            return None
        
        # Find a suitable location in the region
        current_address = region.start_address
        
        # Sort blocks by start address
        sorted_blocks = sorted(region.blocks, key=lambda b: b.start_address)
        
        for block in sorted_blocks:
            # Check if there's enough space before this block
            if block.start_address - current_address >= size:
                # Found a gap big enough
                break
            current_address = block.start_address + block.size
        
        # Check if there's enough space at the end of the region
        if (region.start_address + region.size) - current_address < size:
            return None
        
        # Create new block
        block = MemoryBlock(current_address, size, block_type)
        block.in_use = True
        
        # Generate block ID
        block_id = self.next_block_id
        self.next_block_id += 1
        
        # Add block to region and update free space
        region.blocks.append(block)
        region.free_space -= size
        
        # Store in allocated blocks
        self.allocated_blocks[block_id] = block
        
        logger.info(f"Allocated block {block_id}: {block_type}, {size} bytes at {current_address}")
        return block_id
    
    def optimize_allocation(self) -> None:
        """
        Optimize memory allocation across all regions.
        
        This method attempts to minimize fragmentation and maximize utilization
        by rearranging memory blocks.
        """
        logger.info("Starting memory allocation optimization")
        
        for region in self.regions.values():
            # Sort blocks by start address
            region.blocks.sort(key=lambda b: b.start_address)
            
            # Compact blocks to minimize fragmentation
            current_address = region.start_address
            for block in region.blocks:
                if block.in_use and block.start_address != current_address:
                    logger.info(f"Moving block from {block.start_address} to {current_address}")
                    block.start_address = current_address
                current_address += block.size
            
            # Update free space
            region.free_space = (region.start_address + region.size) - current_address
        
        logger.info("Memory allocation optimization complete")

    def free_memory(self, block_id: int) -> bool:
        """
        Free an allocated memory block.
        
        Args:
            block_id: ID of the block to free
            
        Returns:
            bool: True if successful, False otherwise
        """
        if block_id not in self.allocated_blocks:
            logger.error(f"Block {block_id} does not exist or is already freed")
            return False
        
        block = self.allocated_blocks[block_id]
        
        # Find the region containing this block
        for region in self.regions.values():
            if block in region.blocks:
                # Update block status
                block.in_use = False
                block.owner_id = None
                
                # Update region free space
                region.free_space += block.size
                
                # Remove from allocated blocks
                del self.allocated_blocks[block_id]
                
                logger.info(f"Freed block {block_id}: {block.size} bytes")
                return True
        
        logger.error(f"Block {block_id} not found in any region")
        return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Dict[str, Any]: Dictionary containing memory statistics
        """
        stats = {
            'total_regions': len(self.regions),
            'total_memory': sum(r.size for r in self.regions.values()),
            'allocated_memory': sum(b.size for b in self.allocated_blocks.values()),
            'free_memory': sum(r.free_space for r in self.regions.values()),
            'allocation_count': len(self.allocated_blocks),
            'regions': []
        }
        
        for region_id, region in self.regions.items():
            region_stats = {
                'region_id': region_id,
                'region_type': region.region_type,
                'size': region.size,
                'free_space': region.free_space,
                'utilization': (region.size - region.free_space) / region.size if region.size > 0 else 0,
                'block_count': len(region.blocks)
            }
            stats['regions'].append(region_stats)
        
        return stats
    
    def defragment(self, region_id: Optional[int] = None) -> int:
        """
        Defragment memory to consolidate free space.
        
        Args:
            region_id: Optional region ID to defragment (if None, will defragment all regions)
            
        Returns:
            int: Number of blocks moved during defragmentation
        """
        blocks_moved = 0
        
        regions_to_defrag = [self.regions[region_id]] if region_id is not None else self.regions.values()
        
        for region in regions_to_defrag:
            # Only consider blocks that are in use
            active_blocks = [b for b in region.blocks if b.in_use]
            
            if not active_blocks:
                continue
            
            # Sort blocks by address
            active_blocks.sort(key=lambda b: b.start_address)
            
            # Compact blocks starting from the beginning of the region
            current_address = region.start_address
            
            for block in active_blocks:
                if block.start_address != current_address:
                    # Need to move this block
                    old_address = block.start_address
                    block.start_address = current_address
                    blocks_moved += 1
                    logger.info(f"Moved block from {old_address} to {current_address}")
                
                current_address += block.size
            
            # Update free space at the end
            region.free_space = (region.start_address + region.size) - current_address
        
        logger.info(f"Defragmentation complete: {blocks_moved} blocks moved")
        return blocks_moved
    
    def find_best_fit_region(self, size: int, block_type: str) -> Optional[int]:
        """
        Find the region with the best fit for a memory allocation.
        
        Args:
            size: Size of the block to allocate in bytes
            block_type: Type of memory block
            
        Returns:
            Optional[int]: Region ID with the best fit, or None if no suitable region
        """
        best_region_id = None
        smallest_sufficient_space = float('inf')
        
        for region_id, region in self.regions.items():
            if region.free_space >= size:
                # This region has enough space
                
                # Check if it's a better fit than the current best
                if region.free_space < smallest_sufficient_space:
                    smallest_sufficient_space = region.free_space
                    best_region_id = region_id
        
        return best_region_id

    def set_power_mode(self, mode: str) -> None:
        """
        Set the power mode for the memory manager.
        
        Args:
            mode: Power mode to set ('high_performance', 'balanced', 'power_saving')
        """
        if mode not in ["high_performance", "balanced", "power_saving"]:
            logger.error(f"Invalid power mode: {mode}")
            return
        
        self.power_mode = mode
        logger.info(f"Power mode set to {mode}")
        self._apply_power_mode()

    def _apply_power_mode(self) -> None:
        """
        Apply the current power mode settings.
        """
        if self.power_mode == "high_performance":
            # Maximize performance, potentially at the cost of higher power usage
            self._optimize_for_performance()
        elif self.power_mode == "balanced":
            # Balance between performance and power usage
            self._optimize_for_balance()
        elif self.power_mode == "power_saving":
            # Minimize power usage, potentially at the cost of performance
            self._optimize_for_power_saving()

    def _optimize_for_performance(self) -> None:
        """
        Optimize settings for high performance.
        """
        logger.info("Optimizing for high performance")
        # Implement performance optimization logic here
        # Example: Increase memory access speed, prioritize resource allocation

    def _optimize_for_balance(self) -> None:
        """
        Optimize settings for balanced performance and power usage.
        """
        logger.info("Optimizing for balanced performance and power usage")
        # Implement balanced optimization logic here
        # Example: Moderate memory access speed, balanced resource allocation

    def _optimize_for_power_saving(self) -> None:
        """
        Optimize settings for power saving.
        """
        logger.info("Optimizing for power saving")
        # Implement power-saving optimization logic here
        # Example: Reduce memory access speed, limit resource allocation