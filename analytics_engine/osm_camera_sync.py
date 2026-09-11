"""
================================================================================
File: analytics_engine/osm_camera_sync.py
Project: TrackNet AI - City-Wide Multi-Camera ANPR & Urban Traffic Analytics Engine
Purpose: Overpass API client, camera classification engine, deduplication, and PostGIS/DB synchronizer.
Why this file was made:
  To query live OpenStreetMap infrastructure data via Overpass QL, discover physical surveillance/ANPR/CCTV
  nodes, classify them without fabrication, and synchronize them into the local PostgreSQL/SQLite database.
================================================================================
"""

import json
import time
import datetime
import requests
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from config.city_config import (
    CITIES_CONFIG,
    OVERPASS_ENDPOINTS,
    OVERPASS_TIMEOUT_SECONDS,
    get_city_bbox,
    DEFAULT_CITY
)
from database.db_engine import get_db_session, init_db
from database.models import Camera, CameraConnection, ANPREvent


def classify_osm_camera(tags):
    """
    Classifies camera into one of four strictly defined categories:
    - 'ANPR / ALPR'
    - 'traffic camera'
    - 'CCTV / surveillance'
    - 'unknown' (Strict rule: Never infer unknown is ANPR)
    """
    if not tags:
        return "unknown", "surveillance"

    surv_type = str(tags.get("surveillance:type", "")).lower()
    cam_type = str(tags.get("camera:type", "")).lower()
    surv_zone = str(tags.get("surveillance:zone", "")).lower()
    surv_kind = str(tags.get("surveillance", "")).lower()
    name = str(tags.get("name", "")).lower()

    # 1. ANPR / ALPR Check
    if any(k in surv_type or k in cam_type or k in name for k in ["alpr", "anpr", "plate", "license_plate"]):
        return "ANPR / ALPR", surv_type or cam_type or "ALPR"

    # 2. Traffic Camera Check
    if any(k in surv_zone or k in cam_type or k in surv_kind for k in ["traffic", "speed", "average_speed", "red_light"]):
        return "traffic camera", surv_type or cam_type or "traffic"

    # 3. CCTV / General Surveillance Check
    if surv_type in ["camera", "cctv"] or cam_type in ["fixed", "dome", "pan_tilt"] or "surveillance" in tags:
        return "CCTV / surveillance", surv_type or cam_type or "camera"

    # 4. Fallback: Unknown
    return "unknown", surv_type or "surveillance"


class OSMCameraSynchronizer:
    def __init__(self, session=None):
        self.session = session or get_db_session()

    def build_overpass_query(self, bbox_str):
        """Constructs Overpass QL query for surveillance nodes and ways within bbox."""
        return f"""
        [out:json][timeout:{OVERPASS_TIMEOUT_SECONDS}];
        (
          node["man_made"="surveillance"]({bbox_str});
          way["man_made"="surveillance"]({bbox_str});
        );
        out center tags;
        """

    def fetch_overpass_data(self, bbox_str):
        """Queries Overpass API endpoints with automatic failover and retry handling."""
        query = self.build_overpass_query(bbox_str)
        
        for endpoint in OVERPASS_ENDPOINTS:
            try:
                print(f"[OSMSync] Querying Overpass endpoint: {endpoint}")
                resp = requests.post(endpoint, data={"data": query}, timeout=OVERPASS_TIMEOUT_SECONDS + 5)
                if resp.status_code == 200:
                    data = resp.json()
                    elements = data.get("elements", [])
                    print(f"[OSMSync] Successfully retrieved {len(elements)} OSM objects from {endpoint}")
                    return elements
                else:
                    print(f"[OSMSync] Endpoint {endpoint} returned HTTP status {resp.status_code}")
            except Exception as err:
                print(f"[OSMSync] Error connecting to {endpoint}: {err}")
            time.sleep(1)

        print("[OSMSync] WARNING: All Overpass API endpoints timed out or failed.")
        return []

    def parse_and_normalize_element(self, element, city_name):
        """Converts raw OSM element JSON into internal Camera record dictionary."""
        osm_id_val = str(element.get("id"))
        osm_type_val = element.get("type", "node").lower()
        
        # Deterministic Camera ID generation (e.g. OSM_NODE_123456)
        camera_id = f"OSM_{osm_type_val.upper()}_{osm_id_val}"

        # Latitude & Longitude extraction (handling node vs way center coordinates)
        if "lat" in element and "lon" in element:
            lat = float(element["lat"])
            lng = float(element["lon"])
        elif "center" in element:
            lat = float(element["center"]["lat"])
            lng = float(element["center"]["lon"])
        else:
            return None

        tags = element.get("tags", {})
        state_name = CITIES_CONFIG.get(city_name, {}).get("state", "Maharashtra")

        # Classify camera based on OSM tags
        cam_type, surv_type = classify_osm_camera(tags)

        # Additional metadata tags
        direction = tags.get("camera:direction") or tags.get("direction") or "N/A"
        mount = tags.get("camera:mount") or tags.get("mount") or "N/A"
        
        loc_desc = tags.get("name") or tags.get("description") or f"OSM {cam_type} camera at {city_name}"

        source_url = f"https://www.openstreetmap.org/{osm_type_val}/{osm_id_val}"

        return {
            "camera_id": camera_id,
            "osm_type": osm_type_val,
            "osm_id": osm_id_val,
            "city": city_name,
            "state": state_name,
            "camera_type": cam_type,
            "surveillance_type": surv_type,
            "latitude": lat,
            "longitude": lng,
            "direction": direction,
            "mount": mount,
            "location_description": loc_desc,
            "source": "OpenStreetMap",
            "source_url": source_url,
            "verification_status": "osm_mapped",
            "last_synced_at": datetime.datetime.utcnow(),
            "raw_osm_tags": json.dumps(tags)
        }

    def sync_city_cameras(self, city_name):
        """
        Queries Overpass for a specific city, deduplicates, and UPSERTS records into Database.
        Returns: Count of synchronized cameras.
        """
        bbox_str = get_city_bbox(city_name)
        if not bbox_str:
            print(f"[OSMSync] ERROR: City '{city_name}' not found in CITIES_CONFIG.")
            return 0

        elements = self.fetch_overpass_data(bbox_str)
        if not elements:
            print(f"[OSMSync] No OSM camera elements returned for {city_name}.")
            return 0

        synced_records = []
        seen_keys = set()

        for el in elements:
            rec = self.parse_and_normalize_element(el, city_name)
            if rec:
                key = (rec["osm_type"], rec["osm_id"])
                if key not in seen_keys:
                    seen_keys.add(key)
                    synced_records.append(rec)

        if not synced_records:
            return 0

        # Upsert into DB
        count = 0
        for rec in synced_records:
            existing = self.session.query(Camera).filter(
                Camera.osm_type == rec["osm_type"],
                Camera.osm_id == rec["osm_id"]
            ).first()

            if existing:
                for k, v in rec.items():
                    setattr(existing, k, v)
            else:
                cam = Camera(**rec)
                self.session.add(cam)
            count += 1

        self.session.commit()
        print(f"[OSMSync] Successfully synced {count} OpenStreetMap cameras for city: {city_name}")
        return count

    def sync_if_cache_empty(self, city_name=DEFAULT_CITY):
        """
        Performs initial startup sync only when the local database camera cache is empty.
        If cache already contains cameras, skips Overpass network call.
        """
        existing_count = self.session.query(Camera).filter(Camera.city == city_name).count()
        if existing_count > 0:
            print(f"[OSMSync] Camera cache already populated ({existing_count} nodes for {city_name}). Skipping startup Overpass query.")
            return existing_count

        print(f"[OSMSync] Camera cache is empty for {city_name}. Initiating startup Overpass synchronization...")
        count = self.sync_city_cameras(city_name)
        
        # If OSM has sparse camera mappings in this bounding box, populate demo nodes clearly tagged as 'Demo Network'
        if count == 0:
            print(f"[OSMSync] OSM returned 0 cameras for {city_name}. Seeding baseline Demo Network nodes for demonstration.")
            self.seed_demo_cameras(city_name)
            
        return count

    def seed_demo_cameras(self, city_name=DEFAULT_CITY):
        """
        Seeds clearly demarcated proposed/demo cameras when OSM has sparse coverage in a region.
        STRICT DATA RULE: Tagged with source='Demo Network' and verification_status='demo_proposed'.
        """
        city_info = CITIES_CONFIG.get(city_name, CITIES_CONFIG["Mumbai"])
        base_lat = city_info["center_lat"]
        base_lng = city_info["center_lng"]
        state_name = city_info["state"]
        city_prefix = city_name.upper()[:3]

        demo_nodes = [
            (f"CAM_{city_prefix}_01", f"{city_name} Central Intersection Node", base_lat + 0.015, base_lng - 0.010, "ANPR / ALPR"),
            (f"CAM_{city_prefix}_02", f"{city_name} Ring Road Highway Node",     base_lat - 0.020, base_lng + 0.015, "ANPR / ALPR"),
            (f"CAM_{city_prefix}_03", f"{city_name} Flyover Junction",          base_lat - 0.010, base_lng - 0.020, "traffic camera"),
            (f"CAM_{city_prefix}_04", f"{city_name} Expressway Expressway Toll",base_lat + 0.025, base_lng + 0.025, "ANPR / ALPR"),
            (f"CAM_{city_prefix}_05", f"{city_name} Airport Terminal Access",   base_lat - 0.030, base_lng - 0.030, "CCTV / surveillance"),
            (f"CAM_{city_prefix}_06", f"{city_name} Business District Corridor",base_lat + 0.005, base_lng + 0.035, "ANPR / ALPR"),
            (f"CAM_{city_prefix}_07", f"{city_name} Outer Ring Sector Gate",   base_lat + 0.035, base_lng - 0.015, "traffic camera"),
            (f"CAM_{city_prefix}_08", f"{city_name} Bus Terminal Express Gate", base_lat - 0.005, base_lng + 0.005, "ANPR / ALPR")
        ]

        count = 0
        for cam_id, name, lat, lng, c_type in demo_nodes:
            existing = self.session.query(Camera).filter(Camera.camera_id == cam_id).first()
            if not existing:
                cam = Camera(
                    camera_id=cam_id,
                    osm_type="demo",
                    osm_id=cam_id,
                    city=city_name,
                    state=state_name,
                    camera_type=c_type,
                    surveillance_type="demo_anpr",
                    latitude=lat,
                    longitude=lng,
                    direction="N/A",
                    mount="pole",
                    location_description=name,
                    source="Demo Network",
                    source_url="",
                    verification_status="demo_proposed",
                    last_synced_at=datetime.datetime.utcnow(),
                    raw_osm_tags=json.dumps({"demo": True})
                )
                self.session.add(cam)
                count += 1

        self.session.commit()
        print(f"[OSMSync] Seeded {count} Demo Network camera nodes for {city_name}.")
        return count


if __name__ == "__main__":
    init_db()
    synchronizer = OSMCameraSynchronizer()
    synced_count = synchronizer.sync_city_cameras("Mumbai")
    print(f"Total Mumbai cameras synced: {synced_count}")
