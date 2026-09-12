import React, { useEffect, useRef } from 'react';

const CITY_CENTERS = {
  'Mumbai': { lat: 19.0760, lng: 72.8777, zoom: 12 },
  'Pune': { lat: 18.5204, lng: 73.8567, zoom: 12 },
  'Ahmedabad': { lat: 23.0225, lng: 72.5714, zoom: 12 },
  'Gandhinagar': { lat: 23.2156, lng: 72.6369, zoom: 13 },
  'Surat': { lat: 21.1702, lng: 72.8311, zoom: 12 },
  'Vadodara': { lat: 22.3072, lng: 73.1812, zoom: 12 },
  'Rajkot': { lat: 22.3039, lng: 70.8022, zoom: 13 }
};

export default function MultiCameraTrajectory({ activeTrajectory, currentCity, cityCameras }) {
  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const markersRef = useRef([]);
  const polylineRef = useRef(null);
  const cameraMarkersRef = useRef([]);

  // Initialize Map
  useEffect(() => {
    if (!mapContainerRef.current) return;

    if (!mapInstanceRef.current) {
      const center = CITY_CENTERS[currentCity] || CITY_CENTERS['Mumbai'];
      const L = window.L;
      if (!L) return;

      const map = L.map(mapContainerRef.current, {
        zoomControl: true
      }).setView([center.lat, center.lng], center.zoom);

      L.tileLayer('https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, Tiles style by HOT',
        maxZoom: 19
      }).addTo(map);

      mapInstanceRef.current = map;
    }

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, []);

  // Update map view on city change
  useEffect(() => {
    if (mapInstanceRef.current) {
      const center = CITY_CENTERS[currentCity] || CITY_CENTERS['Mumbai'];
      mapInstanceRef.current.setView([center.lat, center.lng], center.zoom);
      setTimeout(() => mapInstanceRef.current?.invalidateSize(), 150);
    }
  }, [currentCity]);

  // Render city camera nodes
  useEffect(() => {
    const map = mapInstanceRef.current;
    const L = window.L;
    if (!map || !L) return;

    cameraMarkersRef.current.forEach(m => map.removeLayer(m));
    cameraMarkersRef.current = [];

    (cityCameras || []).forEach(cam => {
      const color = cam.verification_status === 'osm_mapped' ? '#0284c7' : '#f59e0b';
      const marker = L.circleMarker([cam.latitude, cam.longitude], {
        radius: 4.5,
        fillColor: color,
        color: '#ffffff',
        weight: 1,
        fillOpacity: 0.85
      }).addTo(map);

      marker.bindPopup(`
        <div style="font-family:sans-serif; font-size:12px; line-height:1.4; color:#111;">
          <b style="color:#0284c7;">${cam.location_description}</b><br>
          <b>Node ID:</b> ${cam.camera_id}<br>
          <b>Camera Type:</b> ${cam.camera_type}<br>
          <b>Source:</b> ${cam.source} (${cam.verification_status})
        </div>
      `);
      cameraMarkersRef.current.push(marker);
    });
  }, [cityCameras]);

  // Render trajectory route
  const cityNodes = (activeTrajectory?.trajectory_nodes || []).filter(
    n => n.city && n.city.toLowerCase() === currentCity.toLowerCase()
  );

  useEffect(() => {
    const map = mapInstanceRef.current;
    const L = window.L;
    if (!map || !L) return;

    // Clear previous polyline & markers
    if (polylineRef.current) {
      if (polylineRef.current._casing) map.removeLayer(polylineRef.current._casing);
      map.removeLayer(polylineRef.current);
      polylineRef.current = null;
    }
    markersRef.current.forEach(m => map.removeLayer(m));
    markersRef.current = [];

    if (cityNodes.length > 0) {
      const lineCoords = [];
      cityNodes.forEach((node, idx) => {
        lineCoords.push([node.lat, node.lng]);

        const isStart = (idx === 0);
        const isEnd = (idx === cityNodes.length - 1);
        const markerColor = isStart ? '#059669' : (isEnd ? '#dc2626' : '#1e3a8a');

        const marker = L.circleMarker([node.lat, node.lng], {
          radius: 9,
          fillColor: markerColor,
          color: '#ffffff',
          weight: 2.5,
          fillOpacity: 0.95
        }).addTo(map);

        marker.bindPopup(`
          <div style="font-family:sans-serif; font-size:12px; line-height:1.5; color:#111;">
            <b style="color:#1e3a8a; font-size:13px;">STEP ${idx + 1}: ${node.location_name}</b><br>
            <b>Location:</b> ${node.location_name}<br>
            <b>Camera Node:</b> ${node.camera_id}<br>
            <b>Time:</b> ${node.timestamp}<br>
            <b>Segment Speed:</b> ${node.calculated_speed_kmh} km/h<br>
            <b>Camera Type:</b> ${node.camera_type}<br>
            <b>Source:</b> ${node.source} (${node.verification_status})
          </div>
        `);
        markersRef.current.push(marker);
      });

      if (lineCoords.length > 0) {
        const polylineCasing = L.polyline(lineCoords, {
          color: '#0f172a',
          weight: 7,
          opacity: 0.35
        }).addTo(map);

        const polyline = L.polyline(lineCoords, {
          color: '#1e3a8a',
          weight: 5,
          opacity: 0.95,
          dashArray: '10, 6'
        }).addTo(map);

        polyline._casing = polylineCasing;
        polylineRef.current = polyline;
        map.fitBounds(L.latLngBounds(lineCoords), { padding: [40, 40] });
      }
    }
  }, [activeTrajectory, currentCity]);

  // Compute city-scoped metrics
  let cityDist = 0.0;
  let cityMaxSpeed = 0.0;
  cityNodes.forEach(n => {
    cityDist += (n.segment_distance_km || 0.0);
    if ((n.calculated_speed_kmh || 0.0) > cityMaxSpeed) cityMaxSpeed = n.calculated_speed_kmh;
  });

  const displayDist = cityNodes.length > 0 ? (cityDist > 0 ? cityDist.toFixed(2) : activeTrajectory?.total_distance_km) : '--';
  const displaySpeed = cityNodes.length > 0 ? (cityMaxSpeed > 0 ? cityMaxSpeed.toFixed(1) : (activeTrajectory?.estimated_average_speed_kmh || activeTrajectory?.max_speed_kmh)) : '--';
  const targetPlate = activeTrajectory?.target_plate || '--';
  const vehicleOriginCity = activeTrajectory?.trajectory_nodes?.[0]?.city || 'Mumbai';

  const handleDownloadPDF = () => {
    if (!targetPlate || targetPlate === '--') return;
    window.open(`/api/reports/pdf?plate=${encodeURIComponent(targetPlate)}`, '_blank');
  };

  return (
    <section id="trajectory-tracker" className="tab-content active">
      <div className="trajectory-layout">
        {/* Search & Timeline Sidebar */}
        <div className="panel sidebar-panel" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
          <div className="panel-header">
            <h2><i className="fa-solid fa-gauge-high"></i> Trajectory Summary</h2>
          </div>

          <div className="trajectory-summary-card mt-2">
            <div className="summary-item">
              <span className="label">Target Plate:</span>
              <span className="val highlight">{targetPlate}</span>
            </div>
            <div className="summary-item">
              <span className="label">Total Distance:</span>
              <span className="val">{displayDist} km</span>
            </div>
            <div className="summary-item">
              <span className="label">Avg Speed:</span>
              <span className="val">{displaySpeed} km/h</span>
            </div>
          </div>

          <div className="panel-header mt-3">
            <h2><i className="fa-solid fa-timeline"></i> Spatial -Temporal Sequence</h2>
          </div>

          <div id="timeline-list" className="timeline-container" style={{ flex: 1, overflowY: 'auto' }}>
            {cityNodes.length > 0 ? (
              cityNodes.map((node, idx) => (
                <div key={idx} className="timeline-item">
                  <div className="timeline-step" style={{ fontSize: '0.88rem', fontWeight: 700, color: '#38bdf8', marginBottom: '5px' }}>
                    Step {idx + 1} -
                  </div>
                  <div style={{ fontSize: '0.8rem', lineHeight: 1.65, color: '#e2e8f0', marginTop: '4px' }}>
                    <div><b style={{ color: '#cbd5e1' }}>Camera Node:</b> <span style={{ fontFamily: 'var(--font-mono)', color: '#93c5fd' }}>{node.camera_id}</span></div>
                    <div><b style={{ color: '#cbd5e1' }}>Camera Coordinates:</b> <span style={{ fontFamily: 'var(--font-mono)', color: '#93c5fd' }}>{node.lat.toFixed(5)}, {node.lng.toFixed(5)}</span></div>
                    <div><b style={{ color: '#cbd5e1' }}>Place/Area:</b> <span style={{ color: '#ffffff', fontWeight: 600 }}>{node.location_name}</span></div>
                    <div><b style={{ color: '#cbd5e1' }}>Time:</b> <span style={{ color: '#cbd5e1' }}>{node.timestamp}</span></div>
                  </div>
                </div>
              ))
            ) : activeTrajectory ? (
              <div style={{ textAlign: 'center', padding: '32px 14px', background: 'rgba(239, 68, 68, 0.06)', border: '1px dashed rgba(239, 68, 68, 0.35)', borderRadius: '8px', marginTop: '10px' }}>
                <i className="fa-solid fa-triangle-exclamation" style={{ fontSize: '2.2rem', color: '#f87171', marginBottom: '10px' }}></i>
                <h4 style={{ fontSize: '0.92rem', fontWeight: 700, color: '#f87171', marginBottom: '6px' }}>Vehicle Not Spotted in {currentCity}</h4>
                <p style={{ fontSize: '0.78rem', lineHeight: 1.5, color: '#cbd5e1', marginBottom: '8px' }}>
                  Vehicle <b style={{ color: '#38bdf8' }}>{targetPlate}</b> has not passed through any registered surveillance cameras in <b>{currentCity}</b>.
                </p>
                <span style={{ fontSize: '0.72rem', color: '#94a3b8', display: 'block' }}>
                  Switch region to <b style={{ color: '#38bdf8' }}>{vehicleOriginCity}</b> to view recorded spatial-temporal route.
                </span>
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '40px 15px', color: '#94a3b8' }}>
                <i className="fa-solid fa-camera-rotate" style={{ fontSize: '2.4rem', color: '#38bdf8', marginBottom: '12px', opacity: 0.85 }}></i>
                <h4 style={{ fontSize: '0.98rem', fontWeight: 700, color: '#f1f5f9', marginBottom: '6px' }}>No Active ANPR Vehicle Tested</h4>
                <p style={{ fontSize: '0.78rem', lineHeight: 1.5, color: '#94a3b8', maxWidth: '320px', margin: '0 auto' }}>
                  Please select and run an image in the <b>ANPR & OCR Test Hub</b> tab. Its reconstructed spatial-temporal trajectory will automatically be mapped here.
                </p>
              </div>
            )}
          </div>

          <div className="pdf-export-area mt-3 mb-2">
            <button
              id="btn-export-pdf"
              className="btn btn-secondary full-width"
              disabled={cityNodes.length === 0}
              onClick={handleDownloadPDF}
            >
              <i className="fa-solid fa-file-pdf"></i> Download PDF Trajectory Report
            </button>
          </div>
        </div>

        {/* GIS Map Area */}
        <div className="panel map-panel">
          <div className="panel-header">
            <h2><i className="fa-solid fa-map-location-dot"></i> City Multi-Camera GIS Trajectory Map</h2>
            <div className="map-legend">
              <span className="legend-item"><span className="legend-dot osm-mapped-dot"></span> OpenStreetMap Camera</span>
              <span className="legend-item"><span className="legend-dot demo-dot"></span> Demo Network Camera</span>
              <span className="legend-item"><span className="legend-dot blue"></span> Vehicle Trajectory</span>
            </div>
          </div>
          <div ref={mapContainerRef} className="map-container"></div>
        </div>
      </div>
    </section>
  );
}
