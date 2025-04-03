#!/usr/bin/env python3
"""
Workload Optimization Tool

Command-line tool to optimize hardware configurations for specific workloads.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

#!/usr/bin/env python3


import argparse
import json
import sys
import os
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.core.hardware.workload_optimizer import WorkloadOptimizer, WorkloadType
from src.core.hardware.unified_config_manager import UnifiedConfigManager

def optimize_config(args):
    """Optimize configuration for a specific workload."""
    # Load configuration
    config_manager = UnifiedConfigManager.get_instance()
    
    if args.file:
        # Load from file
        try:
            with open(args.file, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error loading configuration file: {str(e)}")
            return False
    else:
        # Load from config manager
        config = config_manager.load_config(args.hardware, args.name)
        if not config:
            print(f"Configuration '{args.name}' not found for {args.hardware}")
            return False
    
    # Create workload optimizer
    optimizer = WorkloadOptimizer(args.hardware)
    
    # Get workload type
    try:
        workload_type = WorkloadType(args.workload)
    except ValueError:
        print(f"Invalid workload type: {args.workload}")
        print(f"Available workload types: {[wt.value for wt in WorkloadType]}")
        return False
    
    # Check if workload type is supported for this hardware
    available_workloads = optimizer.get_available_workload_types()
    if workload_type not in available_workloads:
        print(f"Workload type '{workload_type.value}' not supported for {args.hardware}")
        print(f"Supported workload types: {[wt.value for wt in available_workloads]}")
        return False
    
    # Optimize configuration
    optimized_config = optimizer.optimize_config(config, workload_type)
    
    # Save optimized configuration
    if args.output:
        # Save to file
        try:
            with open(args.output, 'w') as f:
                json.dump(optimized_config, f, indent=2)
            print(f"Optimized configuration saved to {args.output}")
        except Exception as e:
            print(f"Error saving optimized configuration: {str(e)}")
            return False
    else:
        # Save to config manager
        output_name = f"{args.name}_optimized_{workload_type.value}" if args.name else f"optimized_{workload_type.value}"
        success = config_manager.save_config(args.hardware, output_name, optimized_config)
        
        if success:
            print(f"Optimized configuration saved as '{output_name}' for {args.hardware}")
        else:
            print("Failed to save optimized configuration")
            return False
    
    # Print optimization summary
    print("\nOptimization Summary:")
    print(f"  Hardware: {args.hardware}")
    print(f"  Workload: {workload_type.value}")
    print(f"  Priority: {optimized_config.get('priority', 'unknown')}")
    print(f"  Neuron Model: {optimized_config.get('neuron_params', {}).get('type', 'unknown')}")
    print(f"  Weight Precision: {optimized_config.get('neuron_params', {}).get('weight_precision', 'unknown')}")
    print(f"  Learning: {optimized_config.get('learning', {}).get('enabled', False)}")
    if optimized_config.get('learning', {}).get('enabled', False):
        print(f"  Learning Rule: {optimized_config.get('learning', {}).get('rule', 'unknown')}")
    print(f"  Encoding: {optimized_config.get('encoding', {}).get('scheme', 'unknown')}")
    
    return True

def analyze_workload(args):
    """Analyze metrics to determine workload type."""
    # Load metrics
    try:
        with open(args.metrics, 'r') as f:
            metrics = json.load(f)
    except Exception as e:
        print(f"Error loading metrics file: {str(e)}")
        return False
    
    # Create workload optimizer
    optimizer = WorkloadOptimizer(args.hardware)
    
    # Analyze workload
    workload_type, confidence = optimizer.analyze_workload(metrics)
    
    print("\nWorkload Analysis:")
    print(f"  Hardware: {args.hardware}")
    print(f"  Detected Workload: {workload_type.value}")
    print(f"  Confidence: {confidence:.2f}")
    
    # Suggest optimization if confidence is high enough
    if confidence > 0.6:
        print("\nRecommendation:")
        print(f"  Optimize for {workload_type.value} workload")
        print(f"  Command: optimize_for_workload.py optimize --hardware {args.hardware} --workload {workload_type.value}")
    
    return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Workload Optimization Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize configuration for workload")
    optimize_parser.add_argument("--hardware", "-hw", required=True, help="Hardware type")
    optimize_parser.add_argument("--workload", "-w", required=True, help="Workload type")
    optimize_parser.add_argument("--name", "-n", help="Configuration name")
    optimize_parser.add_argument("--file", "-f", help="Configuration file path")
    optimize_parser.add_argument("--output", "-o", help="Output file path")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze metrics to determine workload type")
    analyze_parser.add_argument("--hardware", "-hw", required=True, help="Hardware type")
    analyze_parser.add_argument("--metrics", "-m", required=True, help="Metrics file path")
    
    args = parser.parse_args()
    
    if args.command == "optimize":
        if not args.name and not args.file:
            optimize_parser.error("Either --name or --file is required")
        optimize_config(args)
    elif args.command == "analyze":
        analyze_workload(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()