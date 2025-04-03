#!/usr/bin/env python3
"""
Hardware Diagnostic Tool

Simple command-line tool for running hardware diagnostics.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

#!/usr/bin/env python3


import os
import sys
import argparse
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.core.utils.logging_framework import get_logger
from src.core.hardware.diagnostics import run_diagnostics, generate_report

logger = get_logger("diagnostic_tool")


def main():
    """Main entry point for hardware diagnostic tool."""
    parser = argparse.ArgumentParser(description="Hardware Diagnostic Tool")
    parser.add_argument("--hardware", help="Hardware type to diagnose")
    parser.add_argument("--output", help="Output file for diagnostic report")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    
    args = parser.parse_args()
    
    # Run diagnostics
    results = run_diagnostics(args.hardware)
    
    # Generate output
    if args.json:
        # JSON output
        print(json.dumps(results, indent=2))
    else:
        # Generate report
        output_file = args.output
        if not output_file:
            # Default output file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.expanduser("~/Oblivion/reports/diagnostics")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"diagnostic_report_{timestamp}.md")
        
        report = generate_report(results, output_file)
        
        # Print summary
        print("\nDiagnostic Summary:")
        print("-" * 40)
        
        if "error" in results:
            print(f"Error: {results['error']}")
            return 1
        
        for hw_id, hw_results in results.get("results", {}).items():
            all_passed = hw_results.get("all_passed", False)
            tests = hw_results.get("tests", [])
            passed_count = sum(1 for t in tests if t.get("result"))
            
            status = "PASSED" if all_passed else "FAILED"
            print(f"Hardware {hw_id}: {status} ({passed_count}/{len(tests)} tests passed)")
        
        print(f"\nDetailed report saved to: {output_file}")
    
    # Return appropriate exit code
    all_passed = all(
        hw_results.get("all_passed", False) 
        for hw_results in results.get("results", {}).values()
    )
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())