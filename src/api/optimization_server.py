#!/usr/bin/env python3
"""
Optimization API Server

Provides a REST API for interacting with the neuromorphic hardware optimizer.
"""

import argparse
import json
import sys
import os
from typing import Dict, Any, List, Optional
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.hardware.optimization_api import OptimizationAPI
from src.core.utils.logging_framework import get_logger

logger = get_logger("optimization_server")

class OptimizationHandler(BaseHTTPRequestHandler):
    """HTTP request handler for optimization API."""
    
    def _set_headers(self, content_type="application/json"):
        """Set response headers."""
        self.send_response(200)
        self.send_header("Content-type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    
    def _send_json_response(self, data):
        """Send JSON response."""
        self._set_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def _send_error(self, error_message, status_code=400):
        """Send error response."""
        self.send_response(status_code)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"error": error_message}).encode())
    
    def _parse_json_body(self):
        """Parse JSON request body."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 0:
            body = self.rfile.read(content_length)
            return json.loads(body.decode())
        return {}
    
    def _parse_query_params(self):
        """Parse query parameters."""
        parsed_url = urllib.parse.urlparse(self.path)
        return urllib.parse.parse_qs(parsed_url.query)
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests."""
        self._set_headers()
    
    def do_GET(self):
        """Handle GET requests."""
        api = OptimizationAPI.get_instance()
        
        # Parse URL
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        
        # Handle different endpoints
        if path == "/api/status":
            # Get optimization status
            status = api.get_optimization_status()
            self._send_json_response(status)
        
        elif path == "/api/hardware":
            # Get registered hardware
            hardware = api.get_registered_hardware()
            self._send_json_response(hardware)
        
        elif path.startswith("/api/stats/"):
            # Get optimization stats for hardware instance
            # Format: /api/stats/<hardware_type>/<hardware_id>
            parts = path.split("/")
            if len(parts) >= 5:
                hardware_type = parts[3]
                hardware_id = parts[4]
                
                stats = api.get_optimization_stats(hardware_type, hardware_id)
                self._send_json_response(stats)
            else:
                self._send_error("Invalid path format")
        
        else:
            self._send_error("Unknown endpoint", 404)
    
    def do_POST(self):
        """Handle POST requests."""
        api = OptimizationAPI.get_instance()
        
        # Parse URL
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        
        # Parse request body
        try:
            body = self._parse_json_body()
        except json.JSONDecodeError:
            self._send_error("Invalid JSON")
            return
        
        # Handle different endpoints
        if path == "/api/register":
            # Register hardware instance
            hardware_type = body.get("hardware_type")
            hardware_id = body.get("hardware_id")
            
            if not hardware_type or not hardware_id:
                self._send_error("Missing hardware_type or hardware_id")
                return
            
            # In a real implementation, we would get the hardware instance and monitor
            # from a hardware registry or create them based on the request
            # For now, we'll use mock objects
            from src.tools.optimization.monitor_realtime_optimization import MockHardware, MockHardwareMonitor
            
            hardware_instance = MockHardware(hardware_type, hardware_id)
            hardware_monitor = MockHardwareMonitor(hardware_type, hardware_id)
            
            success = api.register_hardware(hardware_type, hardware_id, hardware_instance, hardware_monitor)
            
            self._send_json_response({"success": success})
        
        elif path == "/api/start":
            # Start optimization
            hardware_type = body.get("hardware_type")
            hardware_id = body.get("hardware_id")
            interval = body.get("interval", 5.0)
            
            if not hardware_type or not hardware_id:
                self._send_error("Missing hardware_type or hardware_id")
                return
            
            success = api.start_optimization(hardware_type, hardware_id, float(interval))
            
            self._send_json_response({"success": success})
        
        elif path == "/api/stop":
            # Stop optimization
            hardware_type = body.get("hardware_type")
            hardware_id = body.get("hardware_id")
            
            if not hardware_type or not hardware_id:
                self._send_error("Missing hardware_type or hardware_id")
                return
            
            success = api.stop_optimization(hardware_type, hardware_id)
            
            self._send_json_response({"success": success})
        
        elif path == "/api/reset":
            # Reset optimization
            hardware_type = body.get("hardware_type")
            hardware_id = body.get("hardware_id")
            keep_learning = body.get("keep_learning", False)
            
            if not hardware_type or not hardware_id:
                self._send_error("Missing hardware_type or hardware_id")
                return
            
            success = api.reset_optimization(hardware_type, hardware_id, keep_learning)
            
            self._send_json_response({"success": success})
        
        elif path == "/api/config":
            # Update optimizer configuration
            hardware_type = body.get("hardware_type")
            config_updates = body.get("config", {})
            
            if not hardware_type:
                self._send_error("Missing hardware_type")
                return
            
            success = api.update_optimizer_config(hardware_type, config_updates)
            
            self._send_json_response({"success": success})
        
        else:
            self._send_error("Unknown endpoint", 404)

def run_server(port=8080):
    """Run the optimization API server."""
    server_address = ("", port)
    httpd = HTTPServer(server_address, OptimizationHandler)
    
    logger.info(f"Starting optimization API server on port {port}")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    
    httpd.server_close()
    logger.info("Server stopped")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Optimization API Server")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    args = parser.parse_args()
    
    run_server(args.port)

if __name__ == "__main__":
    main()