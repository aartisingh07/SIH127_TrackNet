import sys
import os
import numpy as np
import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from analytics_engine.trajectory_tracker import CAMERA_NODES

# Sample Hotlist / Blacklisted Vehicles
DEFAULT_BLACKLIST = [
    {"plate": "MH12AB1234", "reason": "Stolen Vehicle Alert", "risk": "CRITICAL", "registered": "2026-08-15"},
    {"plate": "DL6CJ8404", "reason": "Traffic Offense Warrant", "risk": "HIGH", "registered": "2026-08-20"},
    {"plate": "HR26DQ9999", "reason": "Unpaid Toll Violations", "risk": "MEDIUM", "registered": "2026-09-01"},
]

class MacroTrafficAnalytics:
    def __init__(self, trajectory_tracker=None):
        self.camera_nodes = CAMERA_NODES
        self.blacklist = list(DEFAULT_BLACKLIST)
        self.trajectory_tracker = trajectory_tracker

    def get_city_traffic_summary(self):
        """Calculates macro level city traffic statistics across all ANPR camera nodes."""
        total_nodes = len(self.camera_nodes)
        
        # Simulated live volume metrics per camera
        node_stats = []
        heatmap_points = []
        total_detected_vehicles = 0
        
        for cam_id, info in self.camera_nodes.items():
            base_count = np.random.randint(120, 650)
            total_detected_vehicles += base_count
            avg_speed = round(float(np.random.uniform(22.0, 78.0)), 1)
            
            # Congestion status heuristic
            if avg_speed < 28.0 or base_count > 500:
                congestion = "HEAVY"
            elif avg_speed < 45.0:
                congestion = "MODERATE"
            else:
                congestion = "SMOOTH"

            node_stats.append({
                'camera_id': cam_id,
                'name': info['name'],
                'sector': info['sector'],
                'lat': info['lat'],
                'lng': info['lng'],
                'hourly_volume': base_count,
                'avg_speed_kmh': avg_speed,
                'congestion_level': congestion
            })
            
            # GIS Heatmap intensity (0.0 to 1.0)
            intensity = min(1.0, max(0.2, base_count / 650.0))
            heatmap_points.append([info['lat'], info['lng'], round(intensity, 2)])

        # Hourly Trend Chart Data (24-hour city traffic profile)
        hours = [f"{h:02d}:00" for h in range(24)]
        hourly_counts = [
            int(300 + 400 * np.sin((h - 6) * np.pi / 12) ** 2 + np.random.randint(-50, 50))
            for h in range(24)
        ]
        hourly_counts = [max(100, c) for c in hourly_counts]

        # Origin-Destination (O-D) Movement Matrix
        sectors = ["Central", "South", "East", "West", "North", "South-West"]
        od_matrix = []
        for s1 in sectors:
            for s2 in sectors:
                if s1 != s2:
                    flow = np.random.randint(150, 1200)
                    od_matrix.append({
                        'origin': s1,
                        'destination': s2,
                        'vehicle_count': flow,
                        'corridor_status': "NORMAL" if flow < 800 else "CONGESTED"
                    })

        # Top Congestion Bottlenecks
        bottlenecks = [n for n in node_stats if n['congestion_level'] == "HEAVY"]
        if not bottlenecks:
            bottlenecks = sorted(node_stats, key=lambda x: x['avg_speed_kmh'])[:2]

        return {
            'total_active_cameras': total_nodes,
            'total_vehicles_detected_24h': total_detected_vehicles,
            'city_avg_speed_kmh': round(float(np.mean([n['avg_speed_kmh'] for n in node_stats])), 1),
            'camera_node_stats': node_stats,
            'heatmap_points': heatmap_points,
            'hourly_trend': {'labels': hours, 'counts': hourly_counts},
            'od_matrix': sorted(od_matrix, key=lambda x: x['vehicle_count'], reverse=True)[:8],
            'bottlenecks': bottlenecks
        }

    def check_blacklist_and_alerts(self, plate_text, camera_id="CAM-01"):
        """Checks if a plate is on the hotlist or triggers route anomaly alerts."""
        plate_clean = plate_text.upper().strip()
        matched_alert = None
        
        for item in self.blacklist:
            if item['plate'] == plate_clean:
                matched_alert = {
                    'type': 'HOTLIST_MATCH',
                    'plate': plate_clean,
                    'reason': item['reason'],
                    'risk_level': item['risk'],
                    'camera_id': camera_id,
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                break
                
        if not matched_alert:
            # Check for generic anomaly triggers
            if plate_clean.startswith("HR26"):
                matched_alert = {
                    'type': 'SPEED_ANOMALY',
                    'plate': plate_clean,
                    'reason': 'Excessive Speed Anomaly (>135 km/h)',
                    'risk_level': 'HIGH',
                    'camera_id': camera_id,
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

        return matched_alert

    def get_all_blacklist(self):
        return self.blacklist

    def add_to_blacklist(self, plate, reason, risk="HIGH"):
        record = {
            'plate': plate.upper().strip(),
            'reason': reason,
            'risk': risk,
            'registered': datetime.datetime.now().strftime("%Y-%m-%d")
        }
        self.blacklist.append(record)
        return record

if __name__ == "__main__":
    analytics = MacroTrafficAnalytics()
    summary = analytics.get_city_traffic_summary()
    print("Total Active Cameras:", summary['total_active_cameras'])
    print("City Avg Speed:", summary['city_avg_speed_kmh'])
    print("Bottlenecks count:", len(summary['bottlenecks']))
