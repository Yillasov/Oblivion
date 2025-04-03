#!/usr/bin/env python3
"""
Hardware Health Monitoring CLI

Simple command-line interface for hardware health monitoring.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

#!/usr/bin/env python3


import sys
import time
import json
import argparse
import os
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.core.utils.logging_framework import get_logger
from src.core.hardware.health_monitor import (
    start_health_monitoring, 
    stop_health_monitoring,
    get_hardware_health,
    register_health_alert_callback
)

logger = get_logger("health_cli")


def print_alert(alert: Dict[str, Any]) -> None:
    """Print health alert."""
    print("\n" + "!" * 80)
    print(f"HARDWARE ALERT: {alert['level']} - {alert['message']}")
    print(f"Hardware: {alert['hardware_type']}")
    print(f"Issues: {', '.join(alert['issues'])}")
    print(f"Time: {alert['timestamp']}")
    print("!" * 80 + "\n")


def monitor_command(args: argparse.Namespace) -> None:
    """Run monitoring command."""
    print(f"Starting hardware health monitoring (interval: {args.interval}s)")
    
    # Register alert callback
    register_health_alert_callback(print_alert)
    
    # Start monitoring
    start_health_monitoring(
        check_interval=args.interval,
        auto_recovery=not args.no_recovery
    )
    
    try:
        while True:
            time.sleep(args.refresh)
            if args.continuous:
                # Clear screen
                print("\033c", end="")
                
                # Print current health
                health = get_hardware_health()
                print(f"Hardware Health Status (updated: {health['timestamp']})")
                print("-" * 80)
                
                for hw in health.get('hardware_health', []):
                    print(f"Hardware: {hw['hardware_type']}")
                    print(f"Overall: {hw['overall_health']}")
                    
                    # Print metrics
                    for metric, value in hw.get('metrics', {}).items():
                        status = hw.get('health_status', {}).get(metric, "UNKNOWN")
                        print(f"  {metric}: {value} ({status})")
                    
                    print("-" * 40)
                
                print(f"System Overall Health: {health.get('overall_health', 'UNKNOWN')}")
                print("-" * 80)
                print("Press Ctrl+C to exit")
    except KeyboardInterrupt:
        print("\nStopping monitoring...")
        stop_health_monitoring()
        print("Monitoring stopped")


def status_command(args: argparse.Namespace) -> None:
    """Run status command."""
    health = get_hardware_health(args.hardware)
    
    if args.json:
        print(json.dumps(health, indent=2))
    else:
        print(f"Hardware Health Status (updated: {health['timestamp']})")
        print("-" * 80)
        
        for hw in health.get('hardware_health', []):
            print(f"Hardware: {hw['hardware_type']}")
            print(f"Overall: {hw['overall_health']}")
            
            # Print metrics
            for metric, value in hw.get('metrics', {}).items():
                status = hw.get('health_status', {}).get(metric, "UNKNOWN")
                print(f"  {metric}: {value} ({status})")
            
            print("-" * 40)
        
        print(f"System Overall Health: {health.get('overall_health', 'UNKNOWN')}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Hardware Health Monitoring CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Start health monitoring")
    monitor_parser.add_argument("--interval", type=float, default=60.0,
                               help="Interval between health checks (seconds)")
    monitor_parser.add_argument("--refresh", type=float, default=5.0,
                               help="Refresh interval for display (seconds)")
    monitor_parser.add_argument("--continuous", action="store_true",
                               help="Continuously update display")
    monitor_parser.add_argument("--no-recovery", action="store_true",
                               help="Disable automatic recovery")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show current health status")
    status_parser.add_argument("--hardware", type=str,
                              help="Hardware type to show status for")
    status_parser.add_argument("--json", action="store_true",
                              help="Output in JSON format")
    
    args = parser.parse_args()
    
    if args.command == "monitor":
        monitor_command(args)
    elif args.command == "status":
        status_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()