/*
================================================================================
File: static/js/app.js
Project: TrackNet AI - City-Wide Multi-Camera ANPR & Urban Traffic Analytics Engine
Purpose: Frontend JavaScript logic for Leaflet GIS maps, OpenStreetMap camera layer,
         city switching, manual Overpass synchronization, trajectory rendering, and UI.
Why this file was made:
  To provide an interactive GIS dashboard that renders real OpenStreetMap camera markers,
  draws vehicle trajectory polylines from GeoJSON, updates macro analytics, and controls
  the 2-Stage ANPR image zoom inspection viewer.
================================================================================
*/

// Global GIS Map Instances & State
let trajectoryMap = null;
let heatmapMap = null;
let trajectoryPolyline = null;
let trajectoryMarkers = [];
let allCameraMarkers = [];
let currentCity = 'Mumbai';

const CITY_CENTERS = {
    'Mumbai': { lat: 19.0760, lng: 72.8777, zoom: 12 },
    'Pune': { lat: 18.5204, lng: 73.8567, zoom: 12 },
    'Ahmedabad': { lat: 23.0225, lng: 72.5714, zoom: 12 },
    'Gandhinagar': { lat: 23.2156, lng: 72.6369, zoom: 13 },
    'Surat': { lat: 21.1702, lng: 72.8311, zoom: 12 },
    'Vadodara': { lat: 22.3072, lng: 73.1812, zoom: 12 },
    'Rajkot': { lat: 22.3039, lng: 70.8022, zoom: 13 }
};

document.addEventListener('DOMContentLoaded', () => {
    initNavigationTabs();
    loadTestDatasetOptions();
    initTrajectoryMap();
    initHeatmapMap();
    loadBlacklistTable();
    initImageZoomControls();

    // Event Listeners
    document.getElementById('global-city-select')?.addEventListener('change', (e) => {
        handleCityChange(e.target.value);
    });

    document.getElementById('btn-sync-osm')?.addEventListener('click', handleOSMCameraSync);
    document.getElementById('btn-run-test')?.addEventListener('click', runSelectedTestImage);
    document.getElementById('file-upload-input')?.addEventListener('change', handleCustomFileUpload);
    document.getElementById('btn-search-trajectory')?.addEventListener('click', () => {
        const p = document.getElementById('plate-search-input').value;
        if (p) searchTrajectory(p);
    });
    document.getElementById('btn-export-pdf')?.addEventListener('click', exportPDFReport);
    document.getElementById('btn-add-blacklist')?.addEventListener('click', handleAddBlacklist);

    // Initial load for default city (Mumbai)
    handleCityChange('Mumbai');
    searchTrajectory('MH12AB1234');
});

/* Tab Switching Logic */
function initNavigationTabs() {
    const tabs = document.querySelectorAll('.nav-tab');
    const contents = document.querySelectorAll('.tab-content');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            contents.forEach(c => c.classList.remove('active'));

            tab.classList.add('active');
            const target = tab.getAttribute('data-tab');
            document.getElementById(target).classList.add('active');

            if (target === 'trajectory-tracker' && trajectoryMap) {
                setTimeout(() => trajectoryMap.invalidateSize(), 200);
            } else if (target === 'macro-analytics' && heatmapMap) {
                setTimeout(() => heatmapMap.invalidateSize(), 200);
                loadMacroAnalytics(currentCity);
            }
        });
    });
}

/* Handles Switching City via Region Dropdown */
async function handleCityChange(cityName) {
    currentCity = cityName;
    console.log(`[TrackNet UI] Switching city focus to: ${cityName}`);

    const centerInfo = CITY_CENTERS[cityName] || CITY_CENTERS['Mumbai'];

    if (trajectoryMap) {
        trajectoryMap.setView([centerInfo.lat, centerInfo.lng], centerInfo.zoom);
    }
    if (heatmapMap) {
        heatmapMap.setView([centerInfo.lat, centerInfo.lng], centerInfo.zoom);
    }

    await loadCityCameras(cityName);
    await loadMacroAnalytics(cityName);
}

/* Loads Camera Nodes for Selected City from DB API */
async function loadCityCameras(cityName) {
    try {
        const resp = await fetch(`/api/cameras?city=${encodeURIComponent(cityName)}`);
        const data = await resp.json();
        if (!data.success) return;

        const cameras = data.cameras || [];
        document.getElementById('header-cam-count').textContent = cameras.length;

        // Populate Simulated Camera Dropdown in ANPR Hub
        const camSelect = document.getElementById('camera-node-select');
        if (camSelect) {
            camSelect.innerHTML = '';
            cameras.forEach(c => {
                const opt = document.createElement('option');
                opt.value = c.camera_id;
                opt.textContent = `${c.camera_id}: ${c.location_description} [${c.camera_type}]`;
                camSelect.appendChild(opt);
            });
        }

        // Render Camera Node Markers on Trajectory Map
        renderCameraMarkersOnMap(cameras);

    } catch (e) {
        console.error('Failed to load city cameras:', e);
    }
}

/* Manual OpenStreetMap Overpass Synchronization Handler */
async function handleOSMCameraSync() {
    const syncBtn = document.getElementById('btn-sync-osm');
    const origHTML = syncBtn.innerHTML;

    syncBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Syncing...';
    syncBtn.disabled = true;

    try {
        const resp = await fetch('/api/cameras/sync', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ city: currentCity })
        });
        const data = await resp.json();

        if (data.success) {
            alert(`[OSM Sync Success] ${data.message}\nTotal Nodes: ${data.cameras_synced}`);
            await loadCityCameras(currentCity);
            await loadMacroAnalytics(currentCity);
        } else {
            alert('OSM Sync Failed: ' + data.error);
        }
    } catch (e) {
        console.error('OSM Sync Error:', e);
        alert('Failed to connect to Overpass synchronization service.');
    } finally {
        syncBtn.innerHTML = origHTML;
        syncBtn.disabled = false;
    }
}

/* Renders Camera Node Markers with Provenance Badges on Leaflet Map */
function renderCameraMarkersOnMap(cameras) {
    if (!trajectoryMap) return;

    // Clear existing static camera markers
    allCameraMarkers.forEach(m => trajectoryMap.removeLayer(m));
    allCameraMarkers = [];

    cameras.forEach(c => {
        const isOsm = (c.verification_status === 'osm_mapped');
        const markerColor = isOsm ? '#0284c7' : '#f59e0b'; // Cyan/Blue for OSM, Orange for Demo

        const marker = L.circleMarker([c.lat, c.lng], {
            radius: isOsm ? 7 : 6,
            fillColor: markerColor,
            color: '#ffffff',
            weight: 1.5,
            fillOpacity: 0.85
        }).addTo(trajectoryMap);

        const badgeClass = isOsm ? 'sys-ok' : 'live-dot';
        const badgeLabel = isOsm ? 'Mapped camera infrastructure from OpenStreetMap' : 'DEMO / PROPOSED CAMERA NETWORK';

        const popupContent = `
            <div style="font-family: sans-serif; font-size: 13px; line-height: 1.5; color: #111;">
                <b style="font-size: 14px; color: #0284c7;">${c.camera_id}</b><br>
                <div style="margin: 4px 0; font-size: 11px; padding: 2px 6px; border-radius: 4px; background: ${isOsm ? '#e0f2fe' : '#fef3c7'}; color: ${isOsm ? '#0369a1' : '#92400e'}; font-weight: bold; display: inline-block;">
                    ${badgeLabel}
                </div><br>
                <b>Area / Location:</b> <b style="color: #0f172a;">${c.location_description}</b><br>
                <b>Type:</b> ${c.camera_type}<br>
                <b>City:</b> ${c.city} (${c.state})<br>
                <b>Coords:</b> ${c.lat.toFixed(5)}, ${c.lng.toFixed(5)}<br>
                <b>Mount:</b> ${c.mount} | <b>Direction:</b> ${c.direction}<br>
                ${c.source_url ? `<a href="${c.source_url}" target="_blank" style="color: #0284c7; text-decoration: underline;">View on OpenStreetMap ↗</a>` : ''}
            </div>
        `;

        marker.bindPopup(popupContent);
        allCameraMarkers.push(marker);
    });
}

/* Load Test Dataset Images into Dropdown */
async function loadTestDatasetOptions() {
    try {
        const resp = await fetch('/api/test_dataset');
        const data = await resp.json();
        if (data.success) {
            const sel = document.getElementById('test-image-select');
            sel.innerHTML = '<option value="">-- Choose from test_dataset/ --</option>';
            data.test_images.forEach(img => {
                const opt = document.createElement('option');
                opt.value = img;
                opt.textContent = img;
                sel.appendChild(opt);
            });
        }
    } catch (e) {
        console.error('Failed to load test dataset:', e);
    }
}

/* Run ANPR on Selected Test Image */
async function runSelectedTestImage() {
    const sel = document.getElementById('test-image-select');
    const filename = sel.value;
    if (!filename) {
        alert('Please select an image from the dropdown first!');
        return;
    }

    const camera_id = document.getElementById('camera-node-select')?.value || 'CAM-01';
    const startTime = performance.now();

    try {
        const resp = await fetch(`/api/test_dataset/run/${filename}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ camera_id: camera_id })
        });
        const data = await resp.json();
        const duration = Math.round(performance.now() - startTime);
        document.getElementById('processing-time').textContent = `Latency: ${duration} ms`;

        if (data.success) {
            renderANPRResults(data);
        } else {
            alert('Error running ANPR: ' + data.error);
        }
    } catch (e) {
        console.error('ANPR Error:', e);
    }
}

/* Handle Custom Image Upload */
async function handleCustomFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    const camera_id = document.getElementById('camera-node-select')?.value || 'CAM-01';
    const formData = new FormData();
    formData.append('image', file);
    formData.append('camera_id', camera_id);

    const startTime = performance.now();
    try {
        const resp = await fetch('/api/anpr/detect', {
            method: 'POST',
            body: formData
        });
        const data = await resp.json();
        const duration = Math.round(performance.now() - startTime);
        document.getElementById('processing-time').textContent = `Latency: ${duration} ms`;

        if (data.success) {
            renderANPRResults(data);
        } else {
            alert('Error running ANPR: ' + data.error);
        }
    } catch (e) {
        console.error('Upload Error:', e);
    }
}

/* Render ANPR Detection Cards & Image Outputs */
function renderANPRResults(data) {
    const imgEl = document.getElementById('annotated-result-img');
    const placeholder = document.querySelector('.placeholder-text');

    if (data.annotated_image_b64) {
        imgEl.src = 'data:image/jpeg;base64,' + data.annotated_image_b64;
        imgEl.classList.remove('hidden');
        if (placeholder) placeholder.classList.add('hidden');
    }

    const container = document.getElementById('output-cards-container');
    container.innerHTML = '';

    if (!data.detections || data.detections.length === 0) {
        container.innerHTML = '<div class="no-data-msg">No license plates detected in this frame</div>';
        return;
    }

    data.detections.forEach(det => {
        const card = document.createElement('div');
        card.className = 'output-card';

        let alertHTML = '';
        if (det.alert) {
            alertHTML = `<div class="alert-banner"><i class="fa-solid fa-triangle-exclamation"></i> ALERT: ${det.alert.reason} (${det.alert.risk_level})</div>`;
        }

        card.innerHTML = `
            <div class="plate-badge-container">
                ${det.crop_b64 ? `<img src="data:image/jpeg;base64,${det.crop_b64}" class="plate-crop-img" title="Click to Zoom Crop" alt="Plate Crop">` : ''}
                <div class="plate-string">${det.plate_text}</div>
            </div>
            <div class="card-metrics">
                <span>Confidence: <b>${Math.round(det.confidence * 100)}%</b></span>
                <span>Box: [${det.bbox.join(', ')}]</span>
            </div>
            ${alertHTML}
        `;

        card.querySelector('.plate-crop-img')?.addEventListener('click', (e) => {
            openImageModal(e.target.src);
        });

        container.appendChild(card);
    });

    if (data.preprocessing_previews) {
        if (data.preprocessing_previews.clahe) {
            document.getElementById('prep-clahe').style.backgroundImage = `url('data:image/jpeg;base64,${data.preprocessing_previews.clahe}')`;
        }
        if (data.preprocessing_previews.blackhat) {
            document.getElementById('prep-blackhat').style.backgroundImage = `url('data:image/jpeg;base64,${data.preprocessing_previews.blackhat}')`;
        }
        if (data.preprocessing_previews.thresh) {
            document.getElementById('prep-thresh').style.backgroundImage = `url('data:image/jpeg;base64,${data.preprocessing_previews.thresh}')`;
        }
    }
}

/* Initialize Trajectory GIS Map (Leaflet) */
function initTrajectoryMap() {
    const center = CITY_CENTERS['Mumbai'];
    trajectoryMap = L.map('gis-map').setView([center.lat, center.lng], center.zoom);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 19
    }).addTo(trajectoryMap);
}

/* Quick Search Preset Button */
function quickTrack(plate) {
    document.getElementById('plate-search-input').value = plate;
    searchTrajectory(plate);
}

/* Search Trajectory & Render Multi-Camera Route on GIS Map */
async function searchTrajectory(plateText) {
    try {
        const resp = await fetch(`/api/vehicles/${encodeURIComponent(plateText)}/trajectory`);
        const data = await resp.json();
        if (!data.success) return;

        const traj = data.trajectory;

        // Update Summary Card
        document.getElementById('traj-plate-display').textContent = traj.target_plate;
        document.getElementById('traj-seq-display').textContent = traj.camera_sequence;
        document.getElementById('traj-dist-display').textContent = `${traj.total_distance_km} km`;
        document.getElementById('traj-speed-display').textContent = `${traj.estimated_average_speed_kmh || traj.max_speed_kmh} km/h`;

        // Render Timeline Sequence
        const timelineList = document.getElementById('timeline-list');
        timelineList.innerHTML = '';

        const latLngs = [];

        traj.trajectory_nodes.forEach(node => {
            latLngs.push([node.lat, node.lng]);

            const item = document.createElement('div');
            item.className = 'timeline-item';
            item.innerHTML = `
                <div class="timeline-step">STEP ${node.step} - CAMERA ${node.camera_id}</div>
                <div class="timeline-title">${node.location_name} (${node.camera_type})</div>
                <div class="timeline-time"><i class="fa-regular fa-clock"></i> ${node.timestamp} | Speed: <b>${node.calculated_speed_kmh} km/h</b></div>
            `;
            timelineList.appendChild(item);
        });

        // Clear previous trajectory route objects
        if (trajectoryPolyline) trajectoryMap.removeLayer(trajectoryPolyline);
        trajectoryMarkers.forEach(m => trajectoryMap.removeLayer(m));
        trajectoryMarkers = [];

        // Draw Polyline using GeoJSON LineString coordinates
        if (traj.geojson && traj.geojson.geometry && traj.geojson.geometry.coordinates.length > 0) {
            const lineCoords = traj.geojson.geometry.coordinates.map(c => [c[1], c[0]]); // Swap [lon, lat] to [lat, lon] for Leaflet
            trajectoryPolyline = L.polyline(lineCoords, {
                color: '#0284c7',
                weight: 5,
                opacity: 0.9,
                dashArray: '8, 8'
            }).addTo(trajectoryMap);

            if (lineCoords.length > 0) {
                trajectoryMap.fitBounds(L.latLngBounds(lineCoords), { padding: [40, 40] });
            }
        }

        // Add Step Markers
        traj.trajectory_nodes.forEach((node, idx) => {
            const marker = L.circleMarker([node.lat, node.lng], {
                radius: 9,
                fillColor: idx === 0 ? '#10b981' : (idx === traj.trajectory_nodes.length - 1 ? '#ef4444' : '#0284c7'),
                color: '#ffffff',
                weight: 2,
                fillOpacity: 0.95
            }).addTo(trajectoryMap);

            marker.bindPopup(`
                <div style="font-size:12px;">
                    <b style="color:#0284c7;">STEP ${node.step}: ${node.camera_id}</b><br>
                    <b>Location:</b> ${node.location_name}<br>
                    <b>Time:</b> ${node.timestamp}<br>
                    <b>Segment Speed:</b> ${node.calculated_speed_kmh} km/h<br>
                    <b>Source:</b> ${node.source} (${node.verification_status})
                </div>
            `);
            trajectoryMarkers.push(marker);
        });

    } catch (e) {
        console.error('Trajectory Search Error:', e);
    }
}

/* Download PDF Trajectory Report */
function exportPDFReport() {
    const plate = document.getElementById('traj-plate-display').textContent || 'MH12AB1234';
    window.open(`/api/reports/pdf?plate=${encodeURIComponent(plate)}`, '_blank');
}

/* Initialize Heatmap GIS Map */
function initHeatmapMap() {
    const center = CITY_CENTERS['Mumbai'];
    heatmapMap = L.map('heatmap-gis-map').setView([center.lat, center.lng], center.zoom);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 19
    }).addTo(heatmapMap);
}

/* Load Macro Traffic Analytics Data for Target City */
async function loadMacroAnalytics(cityName = 'Mumbai') {
    try {
        const resp = await fetch(`/api/analytics/macro?city=${encodeURIComponent(cityName)}`);
        const data = await resp.json();
        if (!data.success) return;

        const analytics = data.analytics;

        document.getElementById('stat-cameras').textContent = analytics.total_active_cameras;
        document.getElementById('stat-volume').textContent = analytics.total_vehicles_detected_24h.toLocaleString();
        document.getElementById('stat-avg-speed').textContent = `${analytics.city_avg_speed_kmh} km/h`;
        document.getElementById('stat-bottlenecks').textContent = analytics.bottlenecks.length;

        // Render O-D Matrix Table
        const odBody = document.getElementById('od-matrix-body');
        if (odBody) {
            odBody.innerHTML = '';
            analytics.od_matrix.forEach(row => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><b>${row.origin}</b></td>
                    <td><b>${row.destination}</b></td>
                    <td>${row.vehicle_count.toLocaleString()}</td>
                    <td><span class="badge-risk ${row.corridor_status === 'CONGESTED' ? 'HIGH' : 'MEDIUM'}">${row.corridor_status}</span></td>
                `;
                odBody.appendChild(tr);
            });
        }

        // Render Bottlenecks Table
        const bnBody = document.getElementById('bottlenecks-body');
        if (bnBody) {
            bnBody.innerHTML = '';
            analytics.bottlenecks.forEach(row => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><b class="highlight">${row.camera_id}</b></td>
                    <td>${row.name}</td>
                    <td>${row.sector}</td>
                    <td>${row.hourly_volume} veh/h</td>
                    <td><font color="#ef4444"><b>${row.avg_speed_kmh} km/h</b></font></td>
                `;
                bnBody.appendChild(tr);
            });
        }

        // Render Heatmap Layer if available
        if (heatmapMap && typeof L.heatLayer === 'function' && analytics.heatmap_points) {
            L.heatLayer(analytics.heatmap_points, { radius: 25, blur: 15, maxZoom: 17 }).addTo(heatmapMap);
        }

    } catch (e) {
        console.error('Failed to load macro analytics:', e);
    }
}

/* Load Blacklist Registry Table */
async function loadBlacklistTable() {
    try {
        const resp = await fetch('/api/alerts/blacklist');
        const data = await resp.json();
        if (!data.success) return;

        const body = document.getElementById('blacklist-table-body');
        if (!body) return;
        body.innerHTML = '';

        data.blacklist.forEach(item => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><b class="plate-string" style="font-size:0.9rem; padding: 2px 6px;">${item.plate}</b></td>
                <td>${item.reason}</td>
                <td><span class="badge-risk ${item.risk}">${item.risk}</span></td>
                <td>${item.registered}</td>
                <td><font color="#10b981"><b>ACTIVE</b></font></td>
            `;
            body.appendChild(tr);
        });
    } catch (e) {
        console.error('Failed to load blacklist:', e);
    }
}

/* Handle Adding New Plate to Blacklist */
async function handleAddBlacklist() {
    const plate = document.getElementById('blacklist-plate-input').value;
    const reason = document.getElementById('blacklist-reason-input').value;
    const risk = document.getElementById('blacklist-risk-select').value;

    if (!plate) {
        alert('Please enter a license plate number!');
        return;
    }

    try {
        const resp = await fetch('/api/alerts/blacklist', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ plate, reason, risk })
        });
        const data = await resp.json();
        if (data.success) {
            alert(`Plate ${plate} registered on Hotlist Watchdog!`);
            document.getElementById('blacklist-plate-input').value = '';
            document.getElementById('blacklist-reason-input').value = '';
            loadBlacklistTable();
        }
    } catch (e) {
        console.error('Failed to add blacklist:', e);
    }
}

/* Interactive Zoom & Pan Engine with Lightbox Modal */
let inlineZoomState = { scale: 1.0, tx: 0, ty: 0, isDragging: false, startX: 0, startY: 0 };
let modalZoomState = { scale: 1.0, tx: 0, ty: 0, isDragging: false, startX: 0, startY: 0 };

function initImageZoomControls() {
    const inlineImg = document.getElementById('annotated-result-img');
    const inlineContainer = document.getElementById('image-container');
    const indicator = document.getElementById('zoom-level-indicator');

    const updateInlineTransform = () => {
        if (inlineImg) {
            inlineImg.style.transform = `scale(${inlineZoomState.scale}) translate(${inlineZoomState.tx}px, ${inlineZoomState.ty}px)`;
        }
        if (indicator) {
            indicator.textContent = `${Math.round(inlineZoomState.scale * 100)}%`;
        }
    };

    document.getElementById('btn-zoom-in')?.addEventListener('click', () => {
        inlineZoomState.scale = Math.min(5.0, inlineZoomState.scale + 0.3);
        updateInlineTransform();
    });

    document.getElementById('btn-zoom-out')?.addEventListener('click', () => {
        inlineZoomState.scale = Math.max(1.0, inlineZoomState.scale - 0.3);
        if (inlineZoomState.scale === 1.0) { inlineZoomState.tx = 0; inlineZoomState.ty = 0; }
        updateInlineTransform();
    });

    document.getElementById('btn-zoom-reset')?.addEventListener('click', () => {
        inlineZoomState = { scale: 1.0, tx: 0, ty: 0, isDragging: false, startX: 0, startY: 0 };
        updateInlineTransform();
    });

    inlineContainer?.addEventListener('wheel', (e) => {
        if (!inlineImg || inlineImg.classList.contains('hidden')) return;
        e.preventDefault();
        const delta = e.deltaY < 0 ? 0.25 : -0.25;
        inlineZoomState.scale = Math.min(5.0, Math.max(1.0, inlineZoomState.scale + delta));
        if (inlineZoomState.scale === 1.0) { inlineZoomState.tx = 0; inlineZoomState.ty = 0; }
        updateInlineTransform();
    }, { passive: false });

    inlineContainer?.addEventListener('mousedown', (e) => {
        if (inlineZoomState.scale <= 1.0 || !inlineImg || inlineImg.classList.contains('hidden')) return;
        inlineZoomState.isDragging = true;
        inlineZoomState.startX = e.clientX - inlineZoomState.tx;
        inlineZoomState.startY = e.clientY - inlineZoomState.ty;
        inlineContainer.classList.add('dragging');
    });

    window.addEventListener('mousemove', (e) => {
        if (!inlineZoomState.isDragging) return;
        inlineZoomState.tx = e.clientX - inlineZoomState.startX;
        inlineZoomState.ty = e.clientY - inlineZoomState.startY;
        updateInlineTransform();
    });

    window.addEventListener('mouseup', () => {
        if (inlineZoomState.isDragging) {
            inlineZoomState.isDragging = false;
            inlineContainer?.classList.remove('dragging');
        }
    });

    inlineImg?.addEventListener('dblclick', () => {
        if (inlineImg.src && !inlineImg.classList.contains('hidden')) {
            openImageModal(inlineImg.src);
        }
    });

    document.getElementById('btn-zoom-modal')?.addEventListener('click', () => {
        if (inlineImg && inlineImg.src && !inlineImg.classList.contains('hidden')) {
            openImageModal(inlineImg.src);
        } else {
            alert('Please select or upload an image to run ANPR first!');
        }
    });

    initModalZoomControls();
}

function openImageModal(imgSrc) {
    const modal = document.getElementById('image-modal');
    const modalImg = document.getElementById('modal-img');
    if (!modal || !modalImg || !imgSrc) return;

    modalImg.src = imgSrc;
    modalZoomState = { scale: 1.0, tx: 0, ty: 0, isDragging: false, startX: 0, startY: 0 };
    if (window.updateModalTransform) window.updateModalTransform();
    modal.classList.remove('hidden');
}

function initModalZoomControls() {
    const modal = document.getElementById('image-modal');
    const modalImg = document.getElementById('modal-img');
    const modalWrapper = document.getElementById('modal-img-wrapper');
    const modalIndicator = document.getElementById('modal-zoom-indicator');
    const overlay = modal?.querySelector('.modal-overlay');
    const closeBtn = document.getElementById('modal-close');

    const updateModalTransform = () => {
        if (modalImg) {
            modalImg.style.transform = `scale(${modalZoomState.scale}) translate(${modalZoomState.tx}px, ${modalZoomState.ty}px)`;
        }
        if (modalIndicator) {
            modalIndicator.textContent = `${Math.round(modalZoomState.scale * 100)}%`;
        }
    };

    window.updateModalTransform = updateModalTransform;

    document.getElementById('modal-zoom-in')?.addEventListener('click', () => {
        modalZoomState.scale = Math.min(8.0, modalZoomState.scale + 0.35);
        updateModalTransform();
    });

    document.getElementById('modal-zoom-out')?.addEventListener('click', () => {
        modalZoomState.scale = Math.max(1.0, modalZoomState.scale - 0.35);
        if (modalZoomState.scale === 1.0) { modalZoomState.tx = 0; modalZoomState.ty = 0; }
        updateModalTransform();
    });

    document.getElementById('modal-zoom-reset')?.addEventListener('click', () => {
        modalZoomState = { scale: 1.0, tx: 0, ty: 0, isDragging: false, startX: 0, startY: 0 };
        updateModalTransform();
    });

    const closeModal = () => modal?.classList.add('hidden');
    closeBtn?.addEventListener('click', closeModal);
    overlay?.addEventListener('click', closeModal);

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !modal?.classList.contains('hidden')) {
            closeModal();
        }
    });

    modalWrapper?.addEventListener('wheel', (e) => {
        e.preventDefault();
        const delta = e.deltaY < 0 ? 0.3 : -0.3;
        modalZoomState.scale = Math.min(8.0, Math.max(1.0, modalZoomState.scale + delta));
        if (modalZoomState.scale === 1.0) { modalZoomState.tx = 0; modalZoomState.ty = 0; }
        updateModalTransform();
    }, { passive: false });

    modalWrapper?.addEventListener('mousedown', (e) => {
        if (modalZoomState.scale <= 1.0) return;
        modalZoomState.isDragging = true;
        modalZoomState.startX = e.clientX - modalZoomState.tx;
        modalZoomState.startY = e.clientY - modalZoomState.ty;
        modalWrapper.classList.add('dragging');
    });

    window.addEventListener('mousemove', (e) => {
        if (!modalZoomState.isDragging) return;
        modalZoomState.tx = e.clientX - modalZoomState.startX;
        modalZoomState.ty = e.clientY - modalZoomState.startY;
        updateModalTransform();
    });

    window.addEventListener('mouseup', () => {
        if (modalZoomState.isDragging) {
            modalZoomState.isDragging = false;
            modalWrapper?.classList.remove('dragging');
        }
    });
}
