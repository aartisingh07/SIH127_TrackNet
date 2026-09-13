"""
================================================================================
File: database/models.py
Project: TrackNet AI - City-Wide Multi-Camera ANPR & Urban Traffic Analytics Engine
Purpose: Defines database ORM schemas for Cameras, Camera Connections, and ANPR Events.
Why this file was made:
  To store synchronized OpenStreetMap camera metadata, logical camera network connections,
  and ANPR detection logs in a structured PostgreSQL/PostGIS (or SQLite) relational database.
================================================================================
"""

import datetime
import json
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, UniqueConstraint, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from database.db_engine import Base

class Camera(Base):
    """
    Represents physical or simulated ANPR/CCTV/traffic camera nodes.
    Supports OpenStreetMap sync metadata and visual verification badges.
    """
    __tablename__ = "cameras"

    camera_id = Column(String(64), primary_key=True, index=True)
    osm_type = Column(String(16), nullable=False, default="demo")  # node, way, relation, demo
    osm_id = Column(String(64), nullable=False, default="0")
    city = Column(String(64), nullable=False, index=True)
    state = Column(String(64), nullable=False)
    
    # Camera Classification (ANPR / ALPR, CCTV / surveillance, traffic camera, unknown)
    camera_type = Column(String(32), nullable=False, default="unknown")
    surveillance_type = Column(String(64), nullable=True)
    
    # Spatial Coordinates (EPSG:4326 WGS84)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    direction = Column(String(32), nullable=True)  # Direction vector/degrees (e.g. 60, 240, N, SW)
    mount = Column(String(32), nullable=True)      # Mounting (e.g. pole, wall, gantry)
    location_description = Column(String(256), nullable=True)
    
    # Data Provenance & Verification
    source = Column(String(64), nullable=False, default="OpenStreetMap") # OpenStreetMap, Demo Network
    source_url = Column(String(256), nullable=True)
    verification_status = Column(String(32), nullable=False, default="osm_mapped") # osm_mapped, demo_proposed, government_verified
    last_synced_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # JSON String storing all raw OSM tags for future auditing
    raw_osm_tags = Column(Text, nullable=True)

    __table_args__ = (
        UniqueConstraint("osm_type", "osm_id", name="uq_osm_type_id"),
    )

    def to_dict(self):
        """Converts camera object to API serializable dictionary."""
        tags = {}
        if self.raw_osm_tags:
            try:
                tags = json.loads(self.raw_osm_tags)
            except Exception:
                tags = {}

        return {
            "camera_id": self.camera_id,
            "osm_type": self.osm_type,
            "osm_id": str(self.osm_id),
            "city": self.city,
            "state": self.state,
            "camera_type": self.camera_type,
            "surveillance_type": self.surveillance_type or "surveillance",
            "lat": self.latitude,
            "lng": self.longitude,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "direction": self.direction or "N/A",
            "mount": self.mount or "N/A",
            "location_description": self.location_description or f"Camera Node {self.camera_id} in {self.city}",
            "source": self.source,
            "source_url": self.source_url or (f"https://www.openstreetmap.org/{self.osm_type}/{self.osm_id}" if self.osm_type in ['node', 'way'] else ""),
            "verification_status": self.verification_status,
            "last_synced_at": self.last_synced_at.strftime("%Y-%m-%d %H:%M:%S") if self.last_synced_at else "",
            "raw_osm_tags": tags
        }


class CameraConnection(Base):
    """
    Represents logical directed connections between camera nodes along road segments.
    """
    __tablename__ = "camera_connections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    from_camera_id = Column(String(64), ForeignKey("cameras.camera_id"), nullable=False)
    to_camera_id = Column(String(64), ForeignKey("cameras.camera_id"), nullable=False)
    distance_m = Column(Float, nullable=False, default=0.0)
    road_name = Column(String(128), nullable=True)
    estimated_travel_time = Column(Float, nullable=False, default=0.0) # in seconds
    connection_source = Column(String(64), default="OSM_Road_Network")

    def to_dict(self):
        return {
            "id": self.id,
            "from_camera_id": self.from_camera_id,
            "to_camera_id": self.to_camera_id,
            "distance_m": self.distance_m,
            "road_name": self.road_name or "Highway Corridor",
            "estimated_travel_time": self.estimated_travel_time,
            "connection_source": self.connection_source
        }


class ANPREvent(Base):
    """
    Represents individual vehicle license plate detection events captured by cameras.
    """
    __tablename__ = "anpr_events"

    event_id = Column(Integer, primary_key=True, autoincrement=True)
    plate_number = Column(String(32), nullable=False, index=True)
    camera_id = Column(String(64), ForeignKey("cameras.camera_id"), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    ocr_confidence = Column(Float, nullable=False, default=0.95)
    confidence_flag = Column(Boolean, nullable=False, default=True)
    vehicle_type = Column(String(32), nullable=False, default="car")
    direction = Column(String(32), nullable=True)

    def to_dict(self):
        return {
            "event_id": self.event_id,
            "plate_number": self.plate_number,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S") if isinstance(self.timestamp, datetime.datetime) else str(self.timestamp),
            "ocr_confidence": round(self.ocr_confidence, 2),
            "confidence_flag": self.confidence_flag,
            "vehicle_type": self.vehicle_type,
            "direction": self.direction or "N/A"
        }
