#!/usr/bin/env python3
"""
Test API gateway initialization.
"""

import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.tools.api.gateway import APIGateway

def test_api_gateway_initialization():
    
    gateway = APIGateway(port=8080)
    assert gateway.port == 8080
    assert len(gateway.routes) == 2  # Default health and routes endpoints
    assert not gateway.is_running

def test_route_registration():
    """Test route registration."""
    gateway = APIGateway()
    
    def test_handler(context):
        return {"status": "ok"}
    
    gateway.register_route("GET", "/test", test_handler)
    assert "GET:/test" in gateway.routes
    assert gateway.routes["GET:/test"] == test_handler

def test_middleware_registration():
    """Test middleware registration."""
    gateway = APIGateway()
    
    def test_middleware(context):
        return True
    
    gateway.add_middleware(test_middleware)
    assert len(gateway.middleware) == 1
    assert gateway.middleware[0] == test_middleware

def test_health_check():
    """Test health check endpoint."""
    gateway = APIGateway()
    response = gateway._health_check({})
    assert response["status"] == "ok"
    assert "timestamp" in response
    assert "uptime" in response
    assert response["routes"] == 2