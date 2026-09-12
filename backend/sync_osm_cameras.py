"""
================================================================================
File: sync_osm_cameras.py
Project: TrackNet AI - City-Wide Multi-Camera ANPR & Urban Traffic Analytics Engine
Purpose: Command-line interface (CLI) script to manually trigger OpenStreetMap camera discovery.
Why this file was made:
  To allow system administrators, cron jobs, or developers to execute camera synchronization
  for any supported city (Mumbai, Pune, Ahmedabad, etc.) directly from the command line.
================================================================================
"""

import sys
import argparse
from database.db_engine import init_db, get_db_session
from analytics_engine.osm_camera_sync import OSMCameraSynchronizer
from config.city_config import CITIES_CONFIG, DEFAULT_CITY


def main():
    parser = argparse.ArgumentParser(description="TrackNet AI - OpenStreetMap Camera Synchronizer CLI")
    parser.add_argument(
        "--city",
        type=str,
        default=DEFAULT_CITY,
        help=f"Target city name to sync (Options: {', '.join(CITIES_CONFIG.keys())})"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Synchronize OpenStreetMap cameras across all supported cities"
    )

    args = parser.parse_args()

    print("[CLI OSMSync] Initializing Database connection...")
    init_db()
    session = get_db_session()
    synchronizer = OSMCameraSynchronizer(session=session)

    if args.all:
        print(f"[CLI OSMSync] Synchronizing all cities: {list(CITIES_CONFIG.keys())}")
        total = 0
        for city in CITIES_CONFIG.keys():
            print(f"\n--- Syncing {city} ---")
            count = synchronizer.sync_city_cameras(city)
            if count == 0:
                print(f"[CLI OSMSync] Seeding fallback demo network for {city}...")
                count = synchronizer.seed_demo_cameras(city)
            total += count
        print(f"\n[CLI OSMSync] Completed full synchronization. Total camera nodes stored: {total}")
    else:
        target_city = args.city
        if target_city not in CITIES_CONFIG:
            print(f"[CLI OSMSync] ERROR: '{target_city}' is not in supported cities config: {list(CITIES_CONFIG.keys())}")
            sys.exit(1)

        print(f"[CLI OSMSync] Synchronizing OpenStreetMap camera nodes for: {target_city}")
        count = synchronizer.sync_city_cameras(target_city)
        if count == 0:
            print(f"[CLI OSMSync] Seeding baseline demo network nodes for {target_city}...")
            count = synchronizer.seed_demo_cameras(target_city)
        print(f"[CLI OSMSync] Synchronization complete. Total camera nodes in {target_city}: {count}")


if __name__ == "__main__":
    main()
