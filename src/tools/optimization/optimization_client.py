#!/usr/bin/env python3
"""
Optimization API Client

Command-line tool to interact with the optimization API server.
"""

import argparse
import json
import sys
import os
import requests
from typing import Dict, Any, List, Optional

def register_hardware(args):
    """Register hardware instance."""
    url = f"http://{args.host}:{args.port}/api/register"
    
    data = {
        "hardware_type": args.hardware_type,
        "hardware_id": args.hardware_id
    }
    
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        result = response.json()
        if result.get("success"):
            print(f"Successfully registered {args.hardware_id} of type {args.hardware_type}")
        else:
            print(f"Failed to register hardware: {result.get('error', 'Unknown error')}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def start_optimization(args):
    """Start optimization."""
    url = f"http://{args.host}:{args.port}/api/start"
    
    data = {
        "hardware_type": args.hardware_type,
        "hardware_id": args.hardware_id,
        "interval": args.interval
    }
    
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        result = response.json()
        if result.get("success"):
            print(f"Successfully started optimization for {args.hardware_id}")
        else:
            print(f"Failed to start optimization: {result.get('error', 'Unknown error')}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def stop_optimization(args):
    """Stop optimization."""
    url = f"http://{args.host}:{args.port}/api/stop"
    
    data = {
        "hardware_type": args.hardware_type,
        "hardware_id": args.hardware_id
    }
    
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        result = response.json()
        if result.get("success"):
            print(f"Successfully stopped optimization for {args.hardware_id}")
        else:
            print(f"Failed to stop optimization: {result.get('error', 'Unknown error')}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def reset_optimization(args):
    """Reset optimization."""
    url = f"http://{args.host}:{args.port}/api/reset"
    
    data = {
        "hardware_type": args.hardware_type,
        "hardware_id": args.hardware_id,
        "keep_learning": args.keep_learning
    }
    
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        result = response.json()
        if result.get("success"):
            print(f"Successfully reset optimization for {args.hardware_id}")
        else:
            print(f"Failed to reset optimization: {result.get('error', 'Unknown error')}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def update_config(args):
    """Update optimizer configuration."""
    url = f"http://{args.host}:{args.port}/api/config"
    
    # Parse config updates
    config_updates = {}
    for update in args.updates:
        key, value = update.split("=", 1)
        
        # Try to convert value to appropriate type
        try:
            # Try as number
            if "." in value:
                config_updates[key] = float(value)
            else:
                config_updates[key] = int(value)
        except ValueError:
            # Keep as string
            config_updates[key] = value
    
    data = {
        "hardware_type": args.hardware_type,
        "config": config_updates
    }
    
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        result = response.json()
        if result.get("success"):
            print(f"Successfully updated configuration for {args.hardware_type}")
        else:
            print(f"Failed to update configuration: {result.get('error', 'Unknown error')}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def get_status(args):
    """Get optimization status."""
    url = f"http://{args.host}:{args.port}/api/status"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        status = response.json()
        
        print("Optimization Status:")
        print(f"  Hardware Types: {', '.join(status.get('hardware_types', []))}")
        
        print("\nRegistered Hardware:")
        for hw_type, instances in status.get('registered_hardware', {}).items():
            print(f"  {hw_type}: {', '.join(instances)}")
        
        print("\nActive Optimizations:")
        for optimization in status.get('active_optimizations', []):
            print(f"  {optimization}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def get_stats(args):
    """Get optimization statistics."""
    url = f"http://{args.host}:{args.port}/api/stats/{args.hardware_type}/{args.hardware_id}"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        stats = response.json()
        
        if stats.get("success"):
            print(f"Optimization Statistics for {args.hardware_id} of type {args.hardware_type}:")
            
            # Print metrics
            metrics_stats = stats.get("stats", {}).get("metrics", {})
            if metrics_stats:
                print("\nMetrics:")
                for metric, metric_stats in metrics_stats.items():
                    print(f"  {metric}:")
                    print(f"    Current: {metric_stats.get('current', 0.0):.3f}")
                    print(f"    Average: {metric_stats.get('average', 0.0):.3f}")
                    print(f"    Min: {metric_stats.get('min', 0.0):.3f}")
                    print(f"    Max: {metric_stats.get('max', 0.0):.3f}")
                    print(f"    Improvement: {metric_stats.get('improvement', 0.0):.3f}")
            
            # Print overall score
            overall_score = stats.get("stats", {}).get("overall_score", {})
            if overall_score:
                print("\nOverall Score:")
                print(f"  Current: {overall_score.get('current', 0.0):.3f}")
                print(f"  Average: {overall_score.get('average', 0.0):.3f}")
                print(f"  Improvement: {overall_score.get('improvement', 0.0):.3f}")
            
            # Print parameters
            parameters = stats.get("parameters", {})
            if parameters:
                print("\nCurrent Parameters:")
                for param, value in parameters.items():
                    print(f"  {param}: {value:.4f}")
            
            # Print optimization count
            optimization_count = stats.get("stats", {}).get("optimization_count", 0)
            print(f"\nOptimization Count: {optimization_count}")
            
            # Print update frequency
            update_frequency = stats.get("update_frequency", 0.0)
            print(f"Update Frequency: {update_frequency:.2f} Hz")
        else:
            print(f"Failed to get statistics: {stats.get('error', 'Unknown error')}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def get_hardware(args):
    """Get registered hardware."""
    url = f"http://{args.host}:{args.port}/api/hardware"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        hardware = response.json()
        
        print("Registered Hardware:")
        for hw_type, instances in hardware.items():
            print(f"  {hw_type}: {', '.join(instances)}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Optimization API Client")
    parser.add_argument("--host", default="localhost", help="API server host")
    parser.add_argument("--port", type=int, default=8080, help="API server port")
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Register command
    register_parser = subparsers.add_parser("register", help="Register hardware instance")
    register_parser.add_argument("--hardware-type", "-t", required=True, help="Hardware type")
    register_parser.add_argument("--hardware-id", "-i", required=True, help="Hardware instance identifier")
    register_parser.set_defaults(func=register_hardware)
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start optimization")
    start_parser.add_argument("--hardware-type", "-t", required=True, help="Hardware type")
    start_parser.add_argument("--hardware-id", "-i", required=True, help="Hardware instance identifier")
    start_parser.add_argument("--interval", type=float, default=5.0, help="Update interval in seconds")
    start_parser.set_defaults(func=start_optimization)
    
    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop optimization")
    stop_parser.add_argument("--hardware-type", "-t", required=True, help="Hardware type")
    stop_parser.add_argument("--hardware-id", "-i", required=True, help="Hardware instance identifier")
    stop_parser.set_defaults(func=stop_optimization)
    
    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset optimization")
    reset_parser.add_argument("--hardware-type", "-t", required=True, help="Hardware type")
    reset_parser.add_argument("--hardware-id", "-i", required=True, help="Hardware instance identifier")
    reset_parser.add_argument("--keep-learning", "-k", action="store_true", help="Keep learned parameters")
    reset_parser.set_defaults(func=reset_optimization)
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Update optimizer configuration")
    config_parser.add_argument("--hardware-type", "-t", required=True, help="Hardware type")
    config_parser.add_argument("--updates", "-u", nargs="+", required=True, help="Configuration updates (key=value)")
    config_parser.set_defaults(func=update_config)
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Get optimization status")
    status_parser.set_defaults(func=get_status)
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Get optimization statistics")
    stats_parser.add_argument("--hardware-type", "-t", required=True, help="Hardware type")
    stats_parser.add_argument("--hardware-id", "-i", required=True, help="Hardware instance identifier")
    stats_parser.set_defaults(func=get_stats)
    
    # Hardware command
    hardware_parser = subparsers.add_parser("hardware", help="Get registered hardware")
    hardware_parser.set_defaults(func=get_hardware)
    
    args = parser.parse_args()
    
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()