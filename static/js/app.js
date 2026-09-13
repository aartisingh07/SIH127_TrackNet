// Global GIS Map Instances
let trajectoryMap = null;
let heatmapMap = null;
let trajectoryPolyline = null;
let trajectoryMarkers = [];

document.addEventListener('DOMContentLoaded', () => {
    initNavigationTabs();
    loadTestDatasetOptions();
    initTrajectoryMap();
    initHeatmapMap();
    loadBlacklistTable();
    loadMacroAnalytics();
    initImageZoomControls();

    // Default search on load
    searchTrajectory('MH12AB1234');

    // Event Listeners
    document.getElementById('btn-run-test').addEventListener('click', runSelectedTestImage);
    document.getElementById('file-upload-input').addEventListener('change', handleCustomFileUpload);
    document.getElementById('btn-search-trajectory').addEventListener('click', () => {
        const p = document.getElementById('plate-search-input').value;
        if (p) searchTrajectory(p);
    });
    document.getElementById('btn-export-pdf').addEventListener('click', exportPDFReport);
    document.getElementById('btn-add-blacklist').addEventListener('click', handleAddBlacklist);
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

            // Invalidate Map sizes when switching tabs
            if (target === 'trajectory-tracker' && trajectoryMap) {
                setTimeout(() => trajectoryMap.invalidateSize(), 200);
            } else if (target === 'macro-analytics' && heatmapMap) {
                setTimeout(() => heatmapMap.invalidateSize(), 200);
                loadMacroAnalytics();
            }
        });
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

/* Show ANPR Loading Spinner Overlay & Disable Buttons */
function showANPRLoading(btnElement, statusMsg = "Running 2-Stage YOLO & EasyOCR Engine...") {
    if (btnElement) {
        btnElement.disabled = true;
        btnElement.dataset.origHtml = btnElement.innerHTML;
        btnElement.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Processing ANPR...`;
    }
    const timeEl = document.getElementById('processing-time');
    if (timeEl) {
        timeEl.innerHTML = `<span class="pulse-text"><i class="fa-solid fa-spinner fa-spin"></i> Processing...</span>`;
    }
    const container = document.getElementById('image-container');
    if (container) {
        let overlay = container.querySelector('.anpr-loading-overlay');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.className = 'anpr-loading-overlay';
            container.appendChild(overlay);
        }
        overlay.innerHTML = `
            <div class="spinner-ring"></div>
            <div class="loading-status-text"><i class="fa-solid fa-gear fa-spin"></i> ${statusMsg}</div>
            <div class="loading-sub-text">YOLO Bounding Box Detection • 5x Super-Resolution Upscaling • CLAHE & Positional Indian Grammar Verification</div>
            <div class="loading-progress-bar"><div class="progress-fill"></div></div>
        `;
        overlay.classList.remove('hidden');
    }
    const cardsContainer = document.getElementById('output-cards-container');
    if (cardsContainer) {
        cardsContainer.innerHTML = '<div class="no-data-msg"><i class="fa-solid fa-spinner fa-spin"></i> Running ANPR inference pipeline...</div>';
    }
}

/* Hide ANPR Loading Overlay & Restore Buttons */
function hideANPRLoading(btnElement) {
    if (btnElement && btnElement.dataset.origHtml) {
        btnElement.disabled = false;
        btnElement.innerHTML = btnElement.dataset.origHtml;
    }
    const container = document.getElementById('image-container');
    if (container) {
        const overlay = container.querySelector('.anpr-loading-overlay');
        if (overlay) overlay.classList.add('hidden');
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

    const btn = document.getElementById('btn-run-test');
    const camera_id = document.getElementById('camera-node-select').value;
    const startTime = performance.now();

    showANPRLoading(btn, `Running ANPR on ${filename}...`);

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
    } finally {
        hideANPRLoading(btn);
    }
}

/* Handle Custom Image Upload */
async function handleCustomFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    const fileLabel = document.querySelector('.file-label');
    const camera_id = document.getElementById('camera-node-select').value;
    const formData = new FormData();
    formData.append('image', file);
    formData.append('camera_id', camera_id);

    const startTime = performance.now();
    showANPRLoading(fileLabel, `Processing Uploaded Image (${file.name})...`);

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
    } finally {
        hideANPRLoading(fileLabel);
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

    // Render Preprocessing Filter Thumbnails
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

let cameraNodeMarkers = {};

/* Load City Camera Nodes onto Map */
async function loadCameraNodes() {
    try {
        const resp = await fetch('/api/cameras');
        const data = await resp.json();
        if (data.success && data.cameras) {
            data.cameras.forEach(cam => {
                const marker = L.circleMarker([cam.lat, cam.lng], {
                    radius: 8,
                    fillColor: '#10b981',
                    color: '#ffffff',
                    weight: 2,
                    opacity: 1,
                    fillOpacity: 0.85
                }).addTo(trajectoryMap);

                marker.bindPopup(`
                    <div style="font-family: sans-serif; color: #111;">
                        <strong>${cam.id}: ${cam.name}</strong><br/>
                        Sector: ${cam.sector}<br/>
                        Speed Limit: ${cam.speed_limit_kmh} km/h
                    </div>
                `);

                cameraNodeMarkers[cam.id] = marker;
            });
        }
    } catch (e) {
        console.error('Failed to load camera nodes:', e);
    }
}

/* Initialize Trajectory GIS Map (Leaflet) */
function initTrajectoryMap() {
    const center = [28.58, 77.20]; // City Camera Network Center
    trajectoryMap = L.map('gis-map').setView(center, 11);

    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; OpenStreetMap &copy; CARTO',
        maxZoom: 18
    }).addTo(trajectoryMap);

    loadCameraNodes();
}

/* Quick Search Preset Button */
function quickTrack(plate) {
    document.getElementById('plate-search-input').value = plate;
    searchTrajectory(plate);
}

/* Search Trajectory & Render Multi-Camera Route on GIS Map */
async function searchTrajectory(plateText) {
    try {
        const resp = await fetch(`/api/trajectory/search?plate=${encodeURIComponent(plateText)}`);
        const data = await resp.json();
        if (!data.success) return;

        const traj = data.trajectory;

        // Update Summary Card
        document.getElementById('traj-plate-display').textContent = traj.target_plate;
        document.getElementById('traj-seq-display').textContent = traj.camera_sequence;
        document.getElementById('traj-dist-display').textContent = `${traj.total_distance_km} km`;
        document.getElementById('traj-speed-display').textContent = `${traj.max_speed_kmh} km/h`;

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
                <div class="timeline-title">${node.location_name}</div>
                <div class="timeline-time"><i class="fa-regular fa-clock"></i> ${node.timestamp} | Speed: <b>${node.calculated_speed_kmh} km/h</b></div>
            `;
            timelineList.appendChild(item);
        });

        // Clear previous map objects
        if (trajectoryPolyline) trajectoryMap.removeLayer(trajectoryPolyline);
        trajectoryMarkers.forEach(m => trajectoryMap.removeLayer(m));
        trajectoryMarkers = [];

        // Draw Polyline for Trajectory Path
        trajectoryPolyline = L.polyline(latLngs, {
            color: '#0284c7',
            weight: 4,
            opacity: 0.8,
            dashArray: '8, 8'
        }).addTo(trajectoryMap);

        // Add Camera Node Markers
        traj.trajectory_nodes.forEach((node, idx) => {
            const marker = L.circleMarker([node.lat, node.lng], {
                radius: 8,
                fillColor: idx === 0 ? '#10b981' : (idx === traj.trajectory_nodes.length - 1 ? '#ef4444' : '#0284c7'),
                color: '#ffffff',
                weight: 2,
                fillOpacity: 0.9
            }).addTo(trajectoryMap);

            marker.bindPopup(`
                <b>${node.location_name} (${node.camera_id})</b><br>
                Timestamp: ${node.timestamp}<br>
                Step: ${node.step} of ${traj.trajectory_nodes.length}<br>
                Speed: ${node.calculated_speed_kmh} km/h
            `);
            trajectoryMarkers.push(marker);
        });

        if (latLngs.length > 0) {
            trajectoryMap.fitBounds(L.latLngBounds(latLngs), { padding: [40, 40] });
        }

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
    const center = [28.58, 77.20];
    heatmapMap = L.map('heatmap-gis-map').setView(center, 11);

    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; OpenStreetMap &copy; CARTO',
        maxZoom: 18
    }).addTo(heatmapMap);
}

/* Load Macro Traffic Analytics Data */
async function loadMacroAnalytics() {
    try {
        const resp = await fetch('/api/analytics/macro');
        const data = await resp.json();
        if (!data.success) return;

        const analytics = data.analytics;

        document.getElementById('stat-cameras').textContent = analytics.total_active_cameras;
        document.getElementById('stat-volume').textContent = analytics.total_vehicles_detected_24h.toLocaleString();
        document.getElementById('stat-avg-speed').textContent = `${analytics.city_avg_speed_kmh} km/h`;
        document.getElementById('stat-bottlenecks').textContent = analytics.bottlenecks.length;

        // Render O-D Matrix Table
        const odBody = document.getElementById('od-matrix-body');
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

        // Render Bottlenecks Table
        const bnBody = document.getElementById('bottlenecks-body');
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

        // Render Heatmap Layer if available
        if (typeof L.heatLayer === 'function' && analytics.heatmap_points) {
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

    // Toolbar Buttons
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

    // Scroll Wheel Zoom on Image Container
    inlineContainer?.addEventListener('wheel', (e) => {
        if (!inlineImg || inlineImg.classList.contains('hidden')) return;
        e.preventDefault();
        const delta = e.deltaY < 0 ? 0.25 : -0.25;
        inlineZoomState.scale = Math.min(5.0, Math.max(1.0, inlineZoomState.scale + delta));
        if (inlineZoomState.scale === 1.0) { inlineZoomState.tx = 0; inlineZoomState.ty = 0; }
        updateInlineTransform();
    }, { passive: false });

    // Drag / Pan Mouse Events for Inline Image
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

    // Double click to open fullscreen modal
    inlineImg?.addEventListener('dblclick', () => {
        if (inlineImg.src && !inlineImg.classList.contains('hidden')) {
            openImageModal(inlineImg.src);
        }
    });

    // Fullscreen Expand Button
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
