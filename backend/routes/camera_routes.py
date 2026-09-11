"""
================================================================================
File: routes/camera_routes.py
Project: TrackNet AI - City-Wide Multi-Camera ANPR & Urban Traffic Analytics Engine
Purpose: Flask REST API Blueprint for OpenStreetMap camera discovery, synchronization,
         vehicle trajectory reconstruction, and city-wide traffic analytics.
Why this file was made:
  To provide modular API endpoints serving camera node metadata, spatial proximity queries,
  manual Overpass synchronization, GeoJSON vehicle trajectories, and traffic analytics.
================================================================================
"""

from flask import Blueprint, jsonify, request
from database.db_engine import get_db_session
from database.models import Camera, ANPREvent
from analytics_engine.osm_camera_sync import OSMCameraSynchronizer
from analytics_engine.trajectory_tracker import TrajectoryTracker, haversine_distance
from analytics_engine.macro_analytics import MacroTrafficAnalytics
from config.city_config import CITIES_CONFIG, DEFAULT_CITY

camera_api = Blueprint("camera_api", __name__)


@camera_api.route("/api/cameras", methods=["GET"])
def get_all_cameras():
    """
    Returns list of camera nodes. Supports optional filters:
    ?city=Mumbai&source=OpenStreetMap&camera_type=ANPR&verification_status=osm_mapped
    """
    session = get_db_session()
    query = session.query(Camera)

    city = request.args.get("city")
    source = request.args.get("source")
    camera_type = request.args.get("camera_type")
    v_status = request.args.get("verification_status")

    if city:
        query = query.filter(Camera.city.ilike(f"%{city}%"))
    if source:
        query = query.filter(Camera.source == source)
    if camera_type:
        query = query.filter(Camera.camera_type == camera_type)
    if v_status:
        query = query.filter(Camera.verification_status == v_status)

    cameras = query.all()
    
    # Auto-seed startup fallback if database is empty
    if not cameras and not city:
        synchronizer = OSMCameraSynchronizer(session=session)
        synchronizer.sync_if_cache_empty(DEFAULT_CITY)
        cameras = session.query(Camera).all()

    return jsonify({
        "success": True,
        "count": len(cameras),
        "cameras": [c.to_dict() for c in cameras]
    })


@camera_api.route("/api/cameras/<camera_id>", methods=["GET"])
def get_camera_by_id(camera_id):
    """Retrieves metadata for a specific camera ID."""
    session = get_db_session()
    cam = session.query(Camera).filter(Camera.camera_id == camera_id).first()
    if not cam:
        return jsonify({"success": False, "error": f"Camera '{camera_id}' not found"}), 404
    return jsonify({"success": True, "camera": cam.to_dict()})


@camera_api.route("/api/cameras/city/<city>", methods=["GET"])
def get_cameras_by_city(city):
    """Retrieves all camera nodes located within a specific city."""
    session = get_db_session()
    cameras = session.query(Camera).filter(Camera.city.ilike(f"%{city}%")).all()
    return jsonify({
        "success": True,
        "city": city,
        "count": len(cameras),
        "cameras": [c.to_dict() for c in cameras]
    })


@camera_api.route("/api/cameras/nearby", methods=["GET"])
def get_nearby_cameras():
    """
    Finds cameras within a given radial distance (in km) of lat/lng coordinates.
    Params: ?lat=19.0760&lon=72.8777&radius=5.0
    """
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon") or request.args.get("lng"))
        radius_km = float(request.args.get("radius", 5.0))
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "Invalid lat, lon, or radius parameters"}), 400

    session = get_db_session()
    cameras = session.query(Camera).all()
    nearby = []

    for cam in cameras:
        dist = haversine_distance(lat, lon, cam.latitude, cam.longitude)
        if dist <= radius_km:
            c_dict = cam.to_dict()
            c_dict["distance_km"] = dist
            nearby.append(c_dict)

    nearby = sorted(nearby, key=lambda x: x["distance_km"])
    return jsonify({
        "success": True,
        "center": {"lat": lat, "lon": lon},
        "radius_km": radius_km,
        "count": len(nearby),
        "cameras": nearby
    })


@camera_api.route("/api/cameras/sync", methods=["POST"])
def sync_cameras():
    """
    Triggers manual OpenStreetMap Overpass API camera synchronization for a target city.
    Body JSON: {"city": "Mumbai"}
    """
    data = request.get_json(silent=True) or {}
    city_name = data.get("city", DEFAULT_CITY)

    if city_name not in CITIES_CONFIG:
        return jsonify({
            "success": False,
            "error": f"City '{city_name}' is not in supported config: {list(CITIES_CONFIG.keys())}"
        }), 400

    session = get_db_session()
    synchronizer = OSMCameraSynchronizer(session=session)
    synced_count = synchronizer.sync_city_cameras(city_name)

    # If OSM returns 0 mapped cameras and DB is empty, seed baseline demo nodes clearly marked as 'Demo Network'
    total_cameras = session.query(Camera).filter_by(city=city_name).count()
    if total_cameras == 0:
        synced_count = synchronizer.seed_demo_cameras(city_name)
        total_cameras = session.query(Camera).filter_by(city=city_name).count()

    return jsonify({
        "success": True,
        "message": f"Successfully synchronized OpenStreetMap cameras for {city_name}",
        "city": city_name,
        "cameras_synced": total_cameras,
        "new_cameras_added": synced_count
    })


@camera_api.route("/api/vehicles/<plate>/trajectory", methods=["GET"])
def get_vehicle_trajectory(plate):
    """
    Reconstructs spatial-temporal vehicle trajectory for a target plate.
    Returns GeoJSON LineString and node step array.
    """
    tracker = TrajectoryTracker()
    result = tracker.reconstruct_trajectory(plate)
    return jsonify({"success": True, "trajectory": result})


@camera_api.route("/api/traffic/density", methods=["GET"])
def get_traffic_density():
    """Returns camera traffic volume & GIS density heatmap points for a city."""
    city_name = request.args.get("city", DEFAULT_CITY)
    macro = MacroTrafficAnalytics()
    summary = macro.get_city_traffic_summary(city_name)
    return jsonify({
        "success": True,
        "city": city_name,
        "total_active_cameras": summary["total_active_cameras"],
        "heatmap_points": summary["heatmap_points"],
        "camera_node_stats": summary["camera_node_stats"]
    })


@camera_api.route("/api/traffic/od", methods=["GET"])
def get_traffic_od_matrix():
    """Returns Origin-Destination (O-D) traffic flow matrix."""
    city_name = request.args.get("city", DEFAULT_CITY)
    macro = MacroTrafficAnalytics()
    summary = macro.get_city_traffic_summary(city_name)
    return jsonify({
        "success": True,
        "city": city_name,
        "od_matrix": summary["od_matrix"]
    })


@camera_api.route("/api/traffic/congestion", methods=["GET"])
def get_traffic_congestion():
    """Returns top traffic congestion bottlenecks and hourly trends."""
    city_name = request.args.get("city", DEFAULT_CITY)
    macro = MacroTrafficAnalytics()
    summary = macro.get_city_traffic_summary(city_name)
    return jsonify({
        "success": True,
        "city": city_name,
        "bottlenecks": summary["bottlenecks"],
        "hourly_trend": summary["hourly_trend"]
    })
