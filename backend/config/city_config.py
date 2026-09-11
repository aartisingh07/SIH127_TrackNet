"""
================================================================================
File: config/city_config.py
Project: TrackNet AI - City-Wide Multi-Camera ANPR & Urban Traffic Analytics Engine
Purpose: Provides configuration for geographic regions (Mumbai, Pune, Gujarat cities),
         Overpass API endpoints, rate-limiting parameters, and database connection strings.
Why this file was made:
  To eliminate hardcoded coordinates throughout the application and establish a
  centralized, extensible configuration for geospatial camera discovery via OpenStreetMap.
================================================================================
"""

import os
import requests

# Default region configuration
DEFAULT_CITY = "Mumbai"
DEFAULT_STATE = "Maharashtra"

# List of robust public Overpass API endpoints (used with failover fallback)
OVERPASS_ENDPOINTS = [
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter"
]

# Query configuration
OVERPASS_TIMEOUT_SECONDS = 60
NOMINATIM_USER_AGENT = "TrackNet-SIH-ANPR-Engine/2.1"

# Database Connection String
# Prefers environment variable DATABASE_URL (e.g. postgresql://user:pass@localhost:5432/tracknet_db)
# Falls back to local SQLite database with GIS support for seamless zero-dependency deployment
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///tracknet.db")

# Supported Geographic Coverage (BBOX format: south, west, north, east)
CITIES_CONFIG = {
    "Mumbai": {
        "state": "Maharashtra",
        "bbox": "18.8500,72.7500,19.3500,73.1000",
        "center_lat": 19.0760,
        "center_lng": 72.8777,
        "zoom": 12,
        "description": "Mumbai Metropolitan Region (MMR) ANPR surveillance grid"
    },
    "Pune": {
        "state": "Maharashtra",
        "bbox": "18.4000,73.7500,18.6500,73.9800",
        "center_lat": 18.5204,
        "center_lng": 73.8567,
        "zoom": 12,
        "description": "Pune Municipal Corporation & Pimpri-Chinchwad CCTV camera grid"
    },
    "Ahmedabad": {
        "state": "Gujarat",
        "bbox": "22.9500,72.4800,23.1200,72.6800",
        "center_lat": 23.0225,
        "center_lng": 72.5714,
        "zoom": 12,
        "description": "Ahmedabad Smart City ANPR surveillance network"
    },
    "Gandhinagar": {
        "state": "Gujarat",
        "bbox": "23.1500,72.5800,23.2800,72.7000",
        "center_lat": 23.2156,
        "center_lng": 72.6369,
        "zoom": 13,
        "description": "Gandhinagar Capital Sector surveillance network"
    },
    "Surat": {
        "state": "Gujarat",
        "bbox": "21.1000,72.7200,21.2800,72.9200",
        "center_lat": 21.1702,
        "center_lng": 72.8311,
        "zoom": 12,
        "description": "Surat Diamond & Textile Belt traffic CCTV network"
    },
    "Vadodara": {
        "state": "Gujarat",
        "bbox": "22.2200,73.1000,22.3800,73.2600",
        "center_lat": 22.3072,
        "center_lng": 73.1812,
        "zoom": 12,
        "description": "Vadodara Urban Development Corridor ANPR network"
    },
    "Rajkot": {
        "state": "Gujarat",
        "bbox": "22.2200,70.7300,22.3500,70.8800",
        "center_lat": 22.3039,
        "center_lng": 70.8022,
        "zoom": 13,
        "description": "Rajkot Municipal Ring Road surveillance network"
    }
}

def get_city_bbox(city_name):
    """Returns the bounding box string for a configured city or None."""
    city_info = CITIES_CONFIG.get(city_name)
    if city_info:
        return city_info["bbox"]
    return None

def fetch_bounding_box_nominatim(city_name, state_name=None):
    """
    Dynamically queries Nominatim API for a city's bounding box if not preset in config.
    Returns: 'south,west,north,east' string or None.
    """
    query = f"{city_name}, {state_name}" if state_name else city_name
    url = "https://nominatim.openstreetmap.org/search"
    headers = {"User-Agent": NOMINATIM_USER_AGENT}
    params = {"q": query, "format": "json", "limit": 1}

    try:
        res = requests.get(url, headers=headers, params=params, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if data:
                bbox_list = data[0].get("boundingbox") # [south, north, west, east]
                if bbox_list and len(bbox_list) == 4:
                    s, n, w, e = bbox_list
                    return f"{s},{w},{n},{e}"
    except Exception as err:
        print(f"[CityConfig] Nominatim lookup failed for {query}: {err}")
    return None
