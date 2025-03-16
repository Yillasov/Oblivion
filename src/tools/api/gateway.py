"""
Simple API Gateway for Subsystem Communication

Provides a lightweight gateway for communication between different subsystems.
"""

import os
import sys
import json
import time
import uuid
import threading
import http.server
import socketserver
from typing import Dict, List, Any, Optional, Callable
from urllib.parse import urlparse, parse_qs
from datetime import datetime

# Add project root to path
sys.path.append("/Users/yessine/Oblivion")

from src.core.utils.logging_framework import get_logger

logger = get_logger("api_gateway")


class APIGateway:
    """Simple API gateway for subsystem communication."""
    
    def __init__(self, port: int = 8080):
        """
        Initialize API gateway.
        
        Args:
            port: Port to listen on
        """
        self.port = port
        self.routes = {}
        self.middleware = []
        self.server = None
        self.server_thread = None
        self.is_running = False
        
        # Register default routes
        self.register_route("GET", "/health", self._health_check)
        self.register_route("GET", "/routes", self._list_routes)
        
        logger.info(f"Initialized API gateway on port {port}")
    
    def register_route(self, method: str, path: str, handler: Callable) -> None:
        """
        Register a route handler.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            path: URL path
            handler: Handler function
        """
        key = f"{method.upper()}:{path}"
        self.routes[key] = handler
        logger.info(f"Registered route: {method} {path}")
    
    def add_middleware(self, middleware: Callable) -> None:
        """
        Add middleware function.
        
        Args:
            middleware: Middleware function
        """
        self.middleware.append(middleware)
        logger.info(f"Added middleware: {middleware.__name__}")
    
    def start(self) -> bool:
        """
        Start the API gateway server.
        
        Returns:
            bool: True if started successfully
        """
        if self.is_running:
            logger.warning("API gateway already running")
            return False
        
        # Create request handler
        gateway = self
        
        class GatewayHandler(http.server.BaseHTTPRequestHandler):
            def _handle_request(self, method):
                # Parse URL
                url = urlparse(self.path)
                path = url.path
                query = parse_qs(url.query)
                
                # Get route handler
                route_key = f"{method}:{path}"
                handler = gateway.routes.get(route_key)
                
                if not handler:
                    self.send_response(404)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    response = {"error": "Not Found", "path": path}
                    self.wfile.write(json.dumps(response).encode())
                    return
                
                # Get request body for POST/PUT
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else ""
                
                try:
                    body_json = json.loads(body) if body else {}
                except json.JSONDecodeError:
                    body_json = {}
                
                # Create request context
                context = {
                    "method": method,
                    "path": path,
                    "query": query,
                    "headers": dict(self.headers),
                    "body": body_json,
                    "raw_body": body,
                    "client_address": self.client_address,
                    "request_time": datetime.now().isoformat()
                }
                
                # Apply middleware
                for middleware in gateway.middleware:
                    try:
                        middleware_result = middleware(context)
                        if middleware_result is False:
                            self.send_response(403)
                            self.send_header("Content-Type", "application/json")
                            self.end_headers()
                            response = {"error": "Forbidden by middleware"}
                            self.wfile.write(json.dumps(response).encode())
                            return
                    except Exception as e:
                        logger.error(f"Middleware error: {str(e)}")
                        self.send_response(500)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        response = {"error": "Internal Server Error", "details": str(e)}
                        self.wfile.write(json.dumps(response).encode())
                        return
                
                # Call handler
                try:
                    start_time = time.time()
                    result = handler(context)
                    elapsed_time = time.time() - start_time
                    
                    # Log request
                    logger.info(f"{method} {path} - {elapsed_time:.3f}s")
                    
                    # Send response
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    
                    if isinstance(result, dict) or isinstance(result, list):
                        self.wfile.write(json.dumps(result).encode())
                    elif isinstance(result, str):
                        self.wfile.write(result.encode())
                    else:
                        self.wfile.write(json.dumps({"result": str(result)}).encode())
                        
                except Exception as e:
                    logger.error(f"Handler error: {str(e)}")
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    response = {"error": "Internal Server Error", "details": str(e)}
                    self.wfile.write(json.dumps(response).encode())
            
            def do_GET(self):
                self._handle_request("GET")
            
            def do_POST(self):
                self._handle_request("POST")
            
            def do_PUT(self):
                self._handle_request("PUT")
            
            def do_DELETE(self):
                self._handle_request("DELETE")
            
            def log_message(self, format, *args):
                # Suppress default logging
                return
        
        try:
            # Create server
            self.server = socketserver.ThreadingTCPServer(("", self.port), GatewayHandler)
            
            # Start server in a separate thread
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            self.is_running = True
            logger.info(f"API gateway started on port {self.port}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting API gateway: {str(e)}")
            return False
    
    def stop(self) -> bool:
        """
        Stop the API gateway server.
        
        Returns:
            bool: True if stopped successfully
        """
        if not self.is_running:
            logger.warning("API gateway not running")
            return False
        
        try:
            # Stop server
            if self.server:
                self.server.shutdown()
                self.server.server_close()
            
            self.is_running = False
            logger.info("API gateway stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping API gateway: {str(e)}")
            return False
    
    def _health_check(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Default health check endpoint."""
        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "uptime": time.time() - getattr(self, "_start_time", time.time()),
            "routes": len(self.routes)
        }
    
    def _list_routes(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """List available routes."""
        routes = []
        
        for route_key in self.routes:
            method, path = route_key.split(":", 1)
            routes.append({
                "method": method,
                "path": path,
                "handler": self.routes[route_key].__name__
            })
        
        return {
            "routes": routes
        }


class SubsystemClient:
    """Client for communicating with the API gateway."""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        """
        Initialize subsystem client.
        
        Args:
            base_url: Base URL of the API gateway
        """
        self.base_url = base_url
        logger.info(f"Initialized subsystem client for {base_url}")
    
    def request(self, 
               method: str, 
               path: str, 
               data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Send a request to the API gateway.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            path: URL path
            data: Request data (for POST/PUT)
            
        Returns:
            Dict[str, Any]: Response data
        """
        import requests
        
        url = f"{self.base_url}{path}"
        
        try:
            if method.upper() == "GET":
                response = requests.get(url)
            elif method.upper() == "POST":
                response = requests.post(url, json=data)
            elif method.upper() == "PUT":
                response = requests.put(url, json=data)
            elif method.upper() == "DELETE":
                response = requests.delete(url)
            else:
                logger.error(f"Unsupported method: {method}")
                return {"error": "Unsupported method"}
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error sending request: {str(e)}")
            return {"error": str(e)}


# Example subsystem handlers
def hardware_status_handler(context: Dict[str, Any]) -> Dict[str, Any]:
    """Handle hardware status requests."""
    # In a real implementation, this would query actual hardware
    return {
        "status": "online",
        "temperature": 45.2,
        "power": 15.7,
        "active_neurons": 1024,
        "timestamp": datetime.now().isoformat()
    }

def simulation_control_handler(context: Dict[str, Any]) -> Dict[str, Any]:
    """Handle simulation control requests."""
    command = context.get("body", {}).get("command")
    
    if not command:
        return {"error": "Missing command parameter"}
    
    if command == "start":
        # Start simulation
        return {"status": "started", "simulation_id": str(uuid.uuid4())}
    elif command == "stop":
        # Stop simulation
        return {"status": "stopped"}
    elif command == "pause":
        # Pause simulation
        return {"status": "paused"}
    elif command == "resume":
        # Resume simulation
        return {"status": "resumed"}
    else:
        return {"error": f"Unknown command: {command}"}

def logging_middleware(context: Dict[str, Any]) -> bool:
    """Simple logging middleware."""
    logger.debug(f"Request: {context['method']} {context['path']} from {context['client_address']}")
    return True


def main():
    """Main entry point for API gateway."""
    import argparse
    
    parser = argparse.ArgumentParser(description="API Gateway for Subsystem Communication")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--example", action="store_true", help="Register example handlers")
    
    args = parser.parse_args()
    
    # Create API gateway
    gateway = APIGateway(port=args.port)
    
    # Add middleware
    gateway.add_middleware(logging_middleware)
    
    # Register example handlers
    if args.example:
        gateway.register_route("GET", "/hardware/status", hardware_status_handler)
        gateway.register_route("POST", "/simulation/control", simulation_control_handler)
        logger.info("Registered example handlers")
    
    # Start gateway
    if gateway.start():
        print(f"API gateway started on port {args.port}")
        print("Press Ctrl+C to stop")
        
        try:
            # Keep running until interrupted
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping API gateway...")
            gateway.stop()
    else:
        print("Failed to start API gateway")
        sys.exit(1)


if __name__ == "__main__":
    main()