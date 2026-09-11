"""
================================================================================
File: tests/test_osm_sync.py
Project: TrackNet AI - City-Wide Multi-Camera ANPR & Urban Traffic Analytics Engine
Purpose: Comprehensive unit test suite covering OSM parsing, camera classification,
         database persistence, trajectory reconstruction, and API endpoints.
Why this file was made:
  To verify system correctness, ensure data integrity (no fabricated OSM tags),
  and validate API contracts and GeoJSON trajectory outputs.
================================================================================
"""

import os
import unittest
import json
from database.db_engine import init_db, get_db_session
from database.models import Camera, ANPREvent
from analytics_engine.osm_camera_sync import classify_osm_camera, OSMCameraSynchronizer
from analytics_engine.trajectory_tracker import TrajectoryTracker, haversine_distance
from config.city_config import CITIES_CONFIG, DEFAULT_CITY
from app import app


class TestOSMCameraSync(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initializes in-memory database and Flask test client."""
        cls.test_db_url = "sqlite:///:memory:"
        init_db(cls.test_db_url)
        cls.session = get_db_session(cls.test_db_url)
        cls.app = app.test_client()

    def test_01_camera_classification_strict_rules(self):
        """Tests camera tag classification logic and strict non-inference of unknown cameras as ANPR."""
        # ANPR / ALPR tag
        cat, _ = classify_osm_camera({"man_made": "surveillance", "surveillance:type": "ALPR"})
        self.assertEqual(cat, "ANPR / ALPR")

        # Traffic camera tag
        cat, _ = classify_osm_camera({"man_made": "surveillance", "surveillance:zone": "traffic"})
        self.assertEqual(cat, "traffic camera")

        # CCTV camera tag
        cat, _ = classify_osm_camera({"man_made": "surveillance", "surveillance:type": "camera"})
        self.assertEqual(cat, "CCTV / surveillance")

        # Unknown camera tag (STRICT RULE: Must return 'unknown', never ANPR)
        cat, _ = classify_osm_camera({"man_made": "surveillance"})
        self.assertEqual(cat, "unknown")

    def test_02_element_parsing_and_normalization(self):
        """Tests parsing raw OSM Overpass JSON elements into internal Camera record schemas."""
        synchronizer = OSMCameraSynchronizer(session=self.session)
        sample_el = {
            "type": "node",
            "id": 10160403770,
            "lat": 19.0304801,
            "lon": 73.0343032,
            "tags": {
                "man_made": "surveillance",
                "surveillance:type": "camera",
                "surveillance:zone": "traffic",
                "camera:mount": "wall",
                "camera:direction": "222"
            }
        }
        rec = synchronizer.parse_and_normalize_element(sample_el, "Mumbai")
        self.assertIsNotNone(rec)
        self.assertEqual(rec["camera_id"], "OSM_NODE_10160403770")
        self.assertEqual(rec["camera_type"], "traffic camera")
        self.assertEqual(rec["source"], "OpenStreetMap")
        self.assertEqual(rec["verification_status"], "osm_mapped")
        self.assertIn("10160403770", rec["source_url"])

    def test_03_haversine_distance_calculation(self):
        """Tests Haversine formula distance calculation between spatial coordinates."""
        # Connaught Place (28.6315, 77.2167) to Lajpat Nagar (28.5672, 77.2433) ~ 7.6 km
        dist = haversine_distance(28.6315, 77.2167, 28.5672, 77.2433)
        self.assertGreater(dist, 5.0)
        self.assertLess(dist, 10.0)

    def test_04_demo_network_seeding_and_provenance(self):
        """Tests demo network camera seeding and verification status tagging."""
        # Clean up any pre-existing test records for Gandhinagar to ensure idempotency
        self.session.query(Camera).filter(Camera.city == "Gandhinagar").delete()
        self.session.commit()

        synchronizer = OSMCameraSynchronizer(session=self.session)
        count = synchronizer.seed_demo_cameras("Gandhinagar")
        self.assertGreater(count, 0)

        demo_cam = self.session.query(Camera).filter(Camera.city == "Gandhinagar").first()
        self.assertIsNotNone(demo_cam)
        self.assertEqual(demo_cam.source, "Demo Network")
        self.assertEqual(demo_cam.verification_status, "demo_proposed")

    def test_05_trajectory_reconstruction_and_geojson(self):
        """Tests trajectory reconstruction joined with camera database and GeoJSON Feature creation."""
        tracker = TrajectoryTracker(session=self.session)
        res = tracker.reconstruct_trajectory("MH12AB1234")

        self.assertEqual(res["target_plate"], "MH12AB1234")
        self.assertIn("geojson", res)
        self.assertEqual(res["geojson"]["type"], "Feature")
        self.assertEqual(res["geojson"]["geometry"]["type"], "LineString")
        self.assertGreater(len(res["trajectory_nodes"]), 0)

    def test_06_flask_api_camera_endpoints(self):
        """Tests Flask REST API endpoints for cameras, trajectory, and analytics."""
        # Test GET /api/cameras
        response = self.app.get("/api/cameras?city=Mumbai")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data["success"])
        self.assertIn("cameras", data)

        # Test GET /api/vehicles/MH12AB1234/trajectory
        res_traj = self.app.get("/api/vehicles/MH12AB1234/trajectory")
        self.assertEqual(res_traj.status_code, 200)
        data_traj = json.loads(res_traj.data)
        self.assertTrue(data_traj["success"])
        self.assertIn("geojson", data_traj["trajectory"])

        # Test GET /api/traffic/density
        res_density = self.app.get("/api/traffic/density?city=Mumbai")
        self.assertEqual(res_density.status_code, 200)
        data_density = json.loads(res_density.data)
        self.assertTrue(data_density["success"])


if __name__ == "__main__":
    unittest.main()
