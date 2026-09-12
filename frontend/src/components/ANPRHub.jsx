import React, { useState, useEffect } from 'react';

export default function ANPRHub({ cityCameras, onANPRSuccess, onViewTrajectory }) {
  const [testImages, setTestImages] = useState([]);
  const [selectedImage, setSelectedImage] = useState('');
  const [selectedCamera, setSelectedCamera] = useState('');
  const [loading, setLoading] = useState(false);
  const [resultData, setResultData] = useState(null);
  const [latency, setLatency] = useState(null);

  // Zoom & Pan state for inline image viewer
  const [zoomScale, setZoomScale] = useState(1);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // Lightbox Modal state & zoom controls
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [modalZoomScale, setModalZoomScale] = useState(1);
  const [modalPanOffset, setModalPanOffset] = useState({ x: 0, y: 0 });
  const [modalIsDragging, setModalIsDragging] = useState(false);
  const [modalDragStart, setModalDragStart] = useState({ x: 0, y: 0 });

  useEffect(() => {
    fetch('/api/test_dataset')
      .then(res => res.json())
      .then(data => {
        const imgs = data.test_images || data.images || [];
        if (data.success && imgs.length > 0) {
          setTestImages(imgs);
          setSelectedImage(imgs[0]);
        }
      })
      .catch(err => console.error('Failed to load test dataset:', err));
  }, []);

  useEffect(() => {
    if (cityCameras && cityCameras.length > 0) {
      setSelectedCamera(cityCameras[0].camera_id);
    }
  }, [cityCameras]);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') setIsModalOpen(false);
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const handleRunANPR = async () => {
    if (!selectedImage) return;
    setLoading(true);
    // Reset zoom when running new image
    setZoomScale(1);
    setPanOffset({ x: 0, y: 0 });
    const start = performance.now();
    try {
      const resp = await fetch(`/api/test_dataset/run/${selectedImage}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera_id: selectedCamera })
      });
      const data = await resp.json();
      setLatency(Math.round(performance.now() - start));
      if (data.success) {
        setResultData(data);
        if (data.detections && data.detections.length > 0) {
          onANPRSuccess(data.detections[0].plate_text);
        }
      } else {
        alert('ANPR Error: ' + data.error);
      }
    } catch (e) {
      console.error('ANPR Execution Error:', e);
    } finally {
      setLoading(false);
    }
  };

  // Inline Viewer Zoom Handlers
  const handleZoomIn = () => {
    setZoomScale(prev => Math.min(prev + 0.25, 4));
  };

  const handleZoomOut = () => {
    setZoomScale(prev => {
      const next = Math.max(prev - 0.25, 1);
      if (next === 1) setPanOffset({ x: 0, y: 0 });
      return next;
    });
  };

  const handleResetZoom = () => {
    setZoomScale(1);
    setPanOffset({ x: 0, y: 0 });
  };

  const handleWheel = (e) => {
    if (!resultData?.annotated_image_b64) return;
    e.preventDefault();
    if (e.deltaY < 0) {
      setZoomScale(prev => Math.min(prev + 0.2, 4));
    } else {
      setZoomScale(prev => {
        const next = Math.max(prev - 0.2, 1);
        if (next === 1) setPanOffset({ x: 0, y: 0 });
        return next;
      });
    }
  };

  const handleMouseDown = (e) => {
    if (zoomScale <= 1) return;
    setIsDragging(true);
    setDragStart({ x: e.clientX - panOffset.x, y: e.clientY - panOffset.y });
  };

  const handleMouseMove = (e) => {
    if (!isDragging) return;
    setPanOffset({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleDoubleClick = () => {
    if (!resultData?.annotated_image_b64) return;
    if (zoomScale === 1) {
      setZoomScale(2);
    } else {
      setZoomScale(1);
      setPanOffset({ x: 0, y: 0 });
    }
  };

  // Modal Zoom Handlers
  const handleModalZoomIn = () => {
    setModalZoomScale(prev => Math.min(prev + 0.25, 5));
  };

  const handleModalZoomOut = () => {
    setModalZoomScale(prev => {
      const next = Math.max(prev - 0.25, 1);
      if (next === 1) setModalPanOffset({ x: 0, y: 0 });
      return next;
    });
  };

  const handleModalResetZoom = () => {
    setModalZoomScale(1);
    setModalPanOffset({ x: 0, y: 0 });
  };

  const handleModalWheel = (e) => {
    e.preventDefault();
    if (e.deltaY < 0) {
      setModalZoomScale(prev => Math.min(prev + 0.25, 5));
    } else {
      setModalZoomScale(prev => {
        const next = Math.max(prev - 0.25, 1);
        if (next === 1) setModalPanOffset({ x: 0, y: 0 });
        return next;
      });
    }
  };

  const handleModalMouseDown = (e) => {
    if (modalZoomScale <= 1) return;
    setModalIsDragging(true);
    setModalDragStart({ x: e.clientX - modalPanOffset.x, y: e.clientY - modalPanOffset.y });
  };

  const handleModalMouseMove = (e) => {
    if (!modalIsDragging) return;
    setModalPanOffset({
      x: e.clientX - modalDragStart.x,
      y: e.clientY - modalDragStart.y
    });
  };

  const handleModalMouseUp = () => {
    setModalIsDragging(false);
  };

  const handleModalDoubleClick = () => {
    if (modalZoomScale === 1) {
      setModalZoomScale(2.5);
    } else {
      setModalZoomScale(1);
      setModalPanOffset({ x: 0, y: 0 });
    }
  };

  return (
    <section id="anpr-hub" className="tab-content active">
      <div className="grid-layout">
        {/* Left Controls Panel */}
        <div className="panel control-panel">
          <div className="panel-header">
            <h2><i className="fa-solid fa-sliders"></i> Input & Controls</h2>
          </div>

          <div className="form-group mb-4">
            <label htmlFor="test-image-select">Select Test Dataset Image:</label>
            <select
              id="test-image-select"
              className="select-input"
              value={selectedImage}
              onChange={(e) => setSelectedImage(e.target.value)}
            >
              {testImages.map(img => (
                <option key={img} value={img}>{img}</option>
              ))}
            </select>
          </div>

          <div className="form-group my-4">
            <button
              id="btn-run-test"
              className="btn btn-primary full-width"
              disabled={loading}
              onClick={handleRunANPR}
            >
              {loading ? <i className="fa-solid fa-spinner fa-spin"></i> : <i className="fa-solid fa-play"></i>} Run ANPR on Selected Image
            </button>
          </div>

          {/* Recognized Output & 4-Step Engine Status Moved to Left Sidebar */}
          <div className="results-cards-area mt-4" style={{ borderTop: '1px solid #334155', paddingTop: '16px' }}>
            <h3 style={{ fontSize: '0.9rem', color: '#38bdf8', marginBottom: '12px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
              <i className="fa-solid fa-square-check"></i> Recognized Output & Engine Status
            </h3>
            <div id="output-cards-container" className="cards-list">
              {resultData && resultData.detections && resultData.detections.length > 0 ? (
                resultData.detections.map((det, idx) => (
                  <div key={idx} className="result-card" style={{ display: 'flex', flexDirection: 'column', gap: '10px', background: 'rgba(15, 23, 42, 0.9)', border: '1px solid #1e293b', borderRadius: '8px', padding: '12px' }}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                      <div className="plate-badge-container" style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        {det.crop_b64 && (
                          <img
                            src={`data:image/jpeg;base64,${det.crop_b64}`}
                            className="plate-crop-img"
                            alt="Plate Crop"
                            style={{ height: '36px', borderRadius: '4px', border: '1px solid #334155' }}
                          />
                        )}
                        <div className="plate-string" style={{ background: '#f59e0b', color: '#000', padding: '4px 10px', borderRadius: '4px', fontWeight: 800, fontSize: '1rem', fontFamily: 'monospace' }}>
                          {det.plate_text}
                        </div>
                      </div>

                      {/* Step 4 Trajectory Navigation Button */}
                      <button
                        className="btn btn-secondary full-width"
                        onClick={() => onViewTrajectory && onViewTrajectory(det.plate_text, resultData?.target_city || 'Mumbai')}
                        title="Click to view route history on interactive multi-camera map"
                        style={{ padding: '8px 12px', background: '#0284c7', color: '#fff', border: 'none', borderRadius: '6px', cursor: 'pointer', fontWeight: 600, fontSize: '0.82rem' }}
                      >
                        <i className="fa-solid fa-route"></i> Step 4: View Route Trajectory ({resultData?.target_city || 'Mumbai'})
                      </button>
                    </div>

                    <div className="card-metrics" style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '0.82rem', color: '#cbd5e1' }}>
                      <div><b>Step 1 (Vehicle):</b> {det.vehicle_type?.toUpperCase() || 'VEHICLE'} ({(det.vehicle_confidence * 100).toFixed(1)}%)</div>
                      <div><b>Step 2 (Plate Box):</b> [{det.bbox?.join(', ')}]</div>
                      <div><b>Step 3 (Exact Conf):</b> <b style={{ color: '#38bdf8' }}>{(det.confidence * 100).toFixed(1)}%</b></div>
                      <div><b>DB Status:</b> <span style={{ color: '#4ade80' }}><i className="fa-solid fa-database"></i> Saved to SQLite</span></div>
                    </div>

                    {det.state_inferred && (
                      <div className="alert-banner" style={{ background: 'rgba(2, 132, 199, 0.2)', borderColor: '#0284c7', color: '#e0f2fe', fontSize: '0.8rem', padding: '8px', borderRadius: '6px' }}>
                        <i className="fa-solid fa-location-dot"></i> <b>Geospatial State Prediction:</b> Inferred initials <b>"{det.inferred_state_code}"</b> from camera node
                      </div>
                    )}

                    {det.alert && (
                      <div className="alert-banner" style={{ fontSize: '0.8rem', padding: '8px', borderRadius: '6px' }}>
                        <i className="fa-solid fa-triangle-exclamation"></i> ALERT: {det.alert.reason} ({det.alert.risk_level})
                      </div>
                    )}
                  </div>
                ))
              ) : (
                <div className="no-data-msg" style={{ fontSize: '0.85rem', color: '#64748b', fontStyle: 'italic', textAlign: 'center', padding: '12px 0' }}>
                  Run ANPR on an image to view recognized output & trajectory
                </div>
              )}
            </div>
          </div>

          <div className="info-box mt-4">
            <h4><i className="fa-solid fa-circle-info"></i> 4-Step Pipeline Architecture</h4>
            <p><b>Step 1:</b> Vehicle Detection (30-Epoch Fine-Tuned Model)</p>
            <p><b>Step 2:</b> License Plate Localization (Vehicle ROI Scoped)</p>
            <p><b>Step 3:</b> OCR + Indian Regex Syntax Grammar Engine</p>
            <p><b>Step 4:</b> Database Storage & Multi-Camera Trajectory</p>
          </div>
        </div>

        {/* Center Display Panel */}
        <div className="panel display-panel">
          <div className="panel-header">
            <h2><i className="fa-solid fa-object-group"></i> Detection & Recognition Result</h2>
            <div className="header-meta">
              <span id="processing-time" className="time-badge">
                Latency: {latency !== null ? `${latency} ms` : '-- ms'}
              </span>
            </div>
          </div>

          <div className="image-viewer-container">
            <div className="image-toolbar" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span>Detection & Bounding Box Overlay (Blue: Vehicle | Green: License Plate)</span>
              <div className="zoom-toolbar">
                <button
                  className="zoom-btn"
                  onClick={handleZoomOut}
                  title="Zoom Out (-)"
                  disabled={!resultData?.annotated_image_b64}
                >
                  <i className="fa-solid fa-minus"></i>
                </button>
                <span className="zoom-level-tag">{Math.round(zoomScale * 100)}%</span>
                <button
                  className="zoom-btn"
                  onClick={handleZoomIn}
                  title="Zoom In (+)"
                  disabled={!resultData?.annotated_image_b64}
                >
                  <i className="fa-solid fa-plus"></i>
                </button>
                <button
                  className="zoom-btn"
                  onClick={handleResetZoom}
                  title="Reset Zoom & Pan"
                  disabled={!resultData?.annotated_image_b64}
                >
                  <i className="fa-solid fa-rotate-left"></i>
                </button>
                <button
                  className="zoom-btn modal-btn"
                  onClick={() => setIsModalOpen(true)}
                  title="Expand Fullscreen Lightbox"
                  disabled={!resultData?.annotated_image_b64}
                >
                  <i className="fa-solid fa-expand"></i>
                </button>
              </div>
            </div>
            <div
              id="image-container"
              className={`image-wrapper ${isDragging ? 'dragging' : ''}`}
              onWheel={handleWheel}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
              onDoubleClick={handleDoubleClick}
              style={{
                overflow: 'hidden',
                cursor: zoomScale > 1 ? (isDragging ? 'grabbing' : 'grab') : 'default'
              }}
            >
              {resultData && resultData.annotated_image_b64 ? (
                <img
                  src={`data:image/jpeg;base64,${resultData.annotated_image_b64}`}
                  className="result-img"
                  alt="Detection Output"
                  style={{
                    transform: `translate(${panOffset.x}px, ${panOffset.y}px) scale(${zoomScale})`,
                    transition: isDragging ? 'none' : 'transform 0.1s ease-out'
                  }}
                />
              ) : (
                <div className="placeholder-text">
                  <i className="fa-solid fa-image"></i> Select a test image to run ANPR engine
                </div>
              )}
            </div>
          </div>

          <div className="results-cards-area mt-4">
            <h3>OpenCV Preprocessing Pipeline Steps</h3>
            <div id="preprocessing-steps-grid" className="prep-grid" style={{ marginTop: '12px' }}>
              <div className="prep-card">
                <span>CLAHE Contrast</span>
                <div
                  className="prep-img-placeholder"
                  style={{
                    backgroundImage: resultData?.preprocessing_previews?.clahe
                      ? `url('data:image/jpeg;base64,${resultData.preprocessing_previews.clahe}')`
                      : 'none'
                  }}
                ></div>
              </div>
              <div className="prep-card">
                <span>BlackHat Filter</span>
                <div
                  className="prep-img-placeholder"
                  style={{
                    backgroundImage: resultData?.preprocessing_previews?.blackhat
                      ? `url('data:image/jpeg;base64,${resultData.preprocessing_previews.blackhat}')`
                      : 'none'
                  }}
                ></div>
              </div>
              <div className="prep-card">
                <span>Adaptive Thresh</span>
                <div
                  className="prep-img-placeholder"
                  style={{
                    backgroundImage: resultData?.preprocessing_previews?.thresh
                      ? `url('data:image/jpeg;base64,${resultData.preprocessing_previews.thresh}')`
                      : 'none'
                  }}
                ></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Fullscreen Lightbox Zoom Modal */}
      {isModalOpen && resultData?.annotated_image_b64 && (
        <div className="image-modal">
          <div className="modal-overlay" onClick={() => setIsModalOpen(false)}></div>
          <div className="modal-content">
            <div className="modal-header">
              <div className="modal-title">
                <i className="fa-solid fa-magnifying-glass-plus"></i> High-Resolution Inspection View — {resultData.filename || selectedImage}
              </div>
              <div className="modal-controls">
                <div className="zoom-toolbar">
                  <button className="zoom-btn modal-btn" onClick={handleModalZoomOut} title="Zoom Out (-)">
                    <i className="fa-solid fa-minus"></i>
                  </button>
                  <span className="zoom-level-tag">{Math.round(modalZoomScale * 100)}%</span>
                  <button className="zoom-btn modal-btn" onClick={handleModalZoomIn} title="Zoom In (+)">
                    <i className="fa-solid fa-plus"></i>
                  </button>
                  <button className="zoom-btn modal-btn" onClick={handleModalResetZoom} title="Reset Zoom">
                    <i className="fa-solid fa-rotate-left"></i>
                  </button>
                </div>
                <button className="modal-close-btn" onClick={() => setIsModalOpen(false)} title="Close (Esc)">
                  <i className="fa-solid fa-xmark"></i>
                </button>
              </div>
            </div>
            <div
              className={`modal-img-wrapper ${modalIsDragging ? 'dragging' : ''}`}
              onWheel={handleModalWheel}
              onMouseDown={handleModalMouseDown}
              onMouseMove={handleModalMouseMove}
              onMouseUp={handleModalMouseUp}
              onMouseLeave={handleModalMouseUp}
              onDoubleClick={handleModalDoubleClick}
              style={{ cursor: modalZoomScale > 1 ? (modalIsDragging ? 'grabbing' : 'grab') : 'default' }}
            >
              <img
                src={`data:image/jpeg;base64,${resultData.annotated_image_b64}`}
                alt="Modal Inspection"
                style={{
                  transform: `translate(${modalPanOffset.x}px, ${modalPanOffset.y}px) scale(${modalZoomScale})`,
                  transition: modalIsDragging ? 'none' : 'transform 0.1s ease-out'
                }}
              />
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

