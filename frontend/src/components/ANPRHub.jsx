import React, { useState, useEffect } from 'react';

const getVehicleIcon = (type) => {
  const t = (type || '').toLowerCase();
  if (t.includes('motorcycle') || t.includes('bike') || t.includes('scooter') || t.includes('two_wheeler') || t.includes('2-wheeler')) {
    return 'fa-solid fa-motorcycle';
  }
  if (t.includes('truck')) {
    return 'fa-solid fa-truck';
  }
  if (t.includes('bus')) {
    return 'fa-solid fa-bus';
  }
  return 'fa-solid fa-car';
};

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

  // Input Mode state: 'preset' | 'url' | 'upload'
  const [inputMode, setInputMode] = useState('preset');
  const [imageUrl, setImageUrl] = useState('');
  const [uploadedFile, setUploadedFile] = useState(null);

  // Lightbox Modal state & zoom controls
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [modalZoomScale, setModalZoomScale] = useState(1);
  const [modalPanOffset, setModalPanOffset] = useState({ x: 0, y: 0 });
  const [modalIsDragging, setModalIsDragging] = useState(false);
  const [modalDragStart, setModalDragStart] = useState({ x: 0, y: 0 });

  // Pop-Up Wide Results Modal State
  const [isResultsModalOpen, setIsResultsModalOpen] = useState(false);

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
      if (e.key === 'Escape') {
        setIsModalOpen(false);
        setIsResultsModalOpen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const handleRunANPR = async () => {
    setLoading(true);
    // Reset zoom when running new image
    setZoomScale(1);
    setPanOffset({ x: 0, y: 0 });
    const start = performance.now();

    try {
      let resp, data;

      if (inputMode === 'preset') {
        if (!selectedImage) {
          alert('Please select a dataset image');
          setLoading(false);
          return;
        }
        resp = await fetch(`/api/test_dataset/run/${selectedImage}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ camera_id: selectedCamera })
        });
        data = await resp.json();
      } else if (inputMode === 'url') {
        if (!imageUrl || !imageUrl.trim()) {
          alert('Please enter a valid Image URL');
          setLoading(false);
          return;
        }
        resp = await fetch('/api/anpr/detect', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image_url: imageUrl.trim(), camera_id: selectedCamera })
        });
        data = await resp.json();
      } else if (inputMode === 'upload') {
        if (!uploadedFile) {
          alert('Please select an image file to upload');
          setLoading(false);
          return;
        }
        const formData = new FormData();
        formData.append('image', uploadedFile);
        formData.append('camera_id', selectedCamera);

        resp = await fetch('/api/anpr/detect', {
          method: 'POST',
          body: formData
        });
        data = await resp.json();
      }

      setLatency(Math.round(performance.now() - start));
      if (data && data.success) {
        setResultData(data);
        if (data.detections && data.detections.length > 0) {
          onANPRSuccess(data.detections[0].plate_text);
        }
        setIsResultsModalOpen(true);
      } else {
        alert('ANPR Error: ' + (data?.error || 'Failed to process image'));
      }
    } catch (e) {
      console.error('ANPR Execution Error:', e);
      alert('Error running ANPR: ' + e.message);
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

          {/* Input Mode Selector Tabs */}
          <div className="input-mode-tabs mb-3" style={{ display: 'flex', gap: '6px', background: 'rgba(15, 23, 42, 0.8)', padding: '4px', borderRadius: '8px', border: '1px solid #334155' }}>
            <button
              onClick={() => setInputMode('preset')}
              style={{
                flex: 1,
                padding: '7px 6px',
                borderRadius: '6px',
                fontSize: '0.78rem',
                fontWeight: 700,
                border: 'none',
                cursor: 'pointer',
                background: inputMode === 'preset' ? 'linear-gradient(135deg, #0284c7, #2563eb)' : 'transparent',
                color: inputMode === 'preset' ? '#ffffff' : '#94a3b8',
                transition: 'all 0.2s ease',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '5px'
              }}
            >
              <i className="fa-solid fa-images"></i> Preset
            </button>

            <button
              onClick={() => setInputMode('url')}
              style={{
                flex: 1,
                padding: '7px 6px',
                borderRadius: '6px',
                fontSize: '0.78rem',
                fontWeight: 700,
                border: 'none',
                cursor: 'pointer',
                background: inputMode === 'url' ? 'linear-gradient(135deg, #0284c7, #2563eb)' : 'transparent',
                color: inputMode === 'url' ? '#ffffff' : '#94a3b8',
                transition: 'all 0.2s ease',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '5px'
              }}
            >
              <i className="fa-solid fa-link"></i> Web URL
            </button>

            <button
              onClick={() => setInputMode('upload')}
              style={{
                flex: 1,
                padding: '7px 6px',
                borderRadius: '6px',
                fontSize: '0.78rem',
                fontWeight: 700,
                border: 'none',
                cursor: 'pointer',
                background: inputMode === 'upload' ? 'linear-gradient(135deg, #0284c7, #2563eb)' : 'transparent',
                color: inputMode === 'upload' ? '#ffffff' : '#94a3b8',
                transition: 'all 0.2s ease',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '5px'
              }}
            >
              <i className="fa-solid fa-upload"></i> Upload
            </button>
          </div>

          {/* Mode 1: Preset Dataset Image Select */}
          {inputMode === 'preset' && (
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
          )}

          {/* Mode 2: Web Image URL Input */}
          {inputMode === 'url' && (
            <div className="form-group mb-4">
              <label htmlFor="image-url-input">Paste Google / Web Image URL:</label>
              <input
                id="image-url-input"
                type="text"
                className="text-input"
                placeholder="https://example.com/vehicle-image.jpg"
                value={imageUrl}
                onChange={(e) => setImageUrl(e.target.value)}
                style={{
                  width: '100%',
                  padding: '9px 12px',
                  background: 'rgba(15, 23, 42, 0.9)',
                  border: '1px solid #0284c7',
                  borderRadius: '6px',
                  color: '#f8fafc',
                  fontSize: '0.85rem'
                }}
              />
              <span style={{ fontSize: '0.72rem', color: '#94a3b8', marginTop: '6px', display: 'block', lineHeight: 1.3 }}>
                <i className="fa-solid fa-circle-info" style={{ color: '#38bdf8' }}></i> Paste any direct web link to a vehicle or license plate image.
              </span>
            </div>
          )}

          {/* Mode 3: Local File Upload */}
          {inputMode === 'upload' && (
            <div className="form-group mb-4">
              <label htmlFor="file-upload-input">Upload Local Vehicle Photo:</label>
              <input
                id="file-upload-input"
                type="file"
                accept="image/*"
                onChange={(e) => setUploadedFile(e.target.files[0] || null)}
                style={{
                  width: '100%',
                  padding: '8px 10px',
                  background: 'rgba(15, 23, 42, 0.9)',
                  border: '1px dashed #0284c7',
                  borderRadius: '6px',
                  color: '#cbd5e1',
                  fontSize: '0.82rem',
                  cursor: 'pointer'
                }}
              />
              {uploadedFile && (
                <div style={{ fontSize: '0.76rem', color: '#4ade80', marginTop: '6px', fontWeight: 600 }}>
                  <i className="fa-solid fa-circle-check"></i> {uploadedFile.name} ({(uploadedFile.size / 1024).toFixed(1)} KB)
                </div>
              )}
            </div>
          )}

          <div className="form-group my-4">
            <button
              id="btn-run-test"
              className="btn btn-primary full-width"
              disabled={loading}
              onClick={handleRunANPR}
            >
              {loading ? (
                <span><i className="fa-solid fa-spinner fa-spin"></i> Processing ANPR...</span>
              ) : (
                <span><i className="fa-solid fa-play"></i> Run ANPR {inputMode === 'url' ? 'on URL' : inputMode === 'upload' ? 'on Uploaded File' : 'on Dataset'}</span>
              )}
            </button>
          </div>

          {/* Recognized Output & 4-Step Engine Status */}
          <div className="results-cards-area mt-4" style={{ borderTop: '1px solid #334155', paddingTop: '16px' }}>
            <h3 style={{ fontSize: '0.9rem', color: '#38bdf8', marginBottom: '12px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
              <i className="fa-solid fa-square-check"></i> Recognized Output & Engine Status
            </h3>
            <div id="output-cards-container" className="cards-list">
              {loading ? (
                <div className="no-data-msg" style={{ fontSize: '0.85rem', color: '#38bdf8', fontStyle: 'italic', textAlign: 'center', padding: '16px 0' }}>
                  <i className="fa-solid fa-spinner fa-spin"></i> Running ANPR inference pipeline...
                </div>
              ) : resultData && resultData.detections && resultData.detections.length > 0 ? (
                <div>
                  <button
                    onClick={() => setIsResultsModalOpen(true)}
                    style={{
                      width: '100%',
                      padding: '10px 14px',
                      marginBottom: '12px',
                      background: 'linear-gradient(135deg, #0284c7, #2563eb)',
                      color: '#ffffff',
                      border: 'none',
                      borderRadius: '8px',
                      cursor: 'pointer',
                      fontWeight: 700,
                      fontSize: '0.85rem',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: '8px',
                      boxShadow: '0 4px 14px rgba(2, 132, 199, 0.4)'
                    }}
                  >
                    <i className="fa-solid fa-window-restore"></i> Open Full Results Modal ({resultData.detections.length})
                  </button>
                  <div style={{ fontSize: '0.82rem', color: '#94a3b8', textAlign: 'center', background: 'rgba(15, 23, 42, 0.6)', padding: '10px', borderRadius: '6px', border: '1px solid #334155' }}>
                    <i className="fa-solid fa-circle-check" style={{ color: '#4ade80', marginRight: '6px' }}></i>
                    {resultData.detections.length} vehicle(s) detected. View full details in the pop-up modal.
                  </div>
                </div>
              ) : (
                <div className="no-data-msg" style={{ fontSize: '0.85rem', color: '#64748b', fontStyle: 'italic', textAlign: 'center', padding: '12px 0' }}>
                  Run ANPR on an image to view recognized output & trajectory
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Center Display Panel */}
        <div className="panel display-panel">
          <div className="panel-header">
            <h2><i className="fa-solid fa-object-group"></i> Detection & Recognition Result</h2>
            <div className="header-meta">
              <span id="processing-time" className="time-badge">
                {loading ? (
                  <span className="pulse-text"><i className="fa-solid fa-spinner fa-spin"></i> Processing...</span>
                ) : (
                  `Latency: ${latency !== null ? `${latency} ms` : '-- ms'}`
                )}
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
                  disabled={!resultData?.annotated_image_b64 || loading}
                >
                  <i className="fa-solid fa-minus"></i>
                </button>
                <span className="zoom-level-tag">{Math.round(zoomScale * 100)}%</span>
                <button
                  className="zoom-btn"
                  onClick={handleZoomIn}
                  title="Zoom In (+)"
                  disabled={!resultData?.annotated_image_b64 || loading}
                >
                  <i className="fa-solid fa-plus"></i>
                </button>
                <button
                  className="zoom-btn"
                  onClick={handleResetZoom}
                  title="Reset Zoom & Pan"
                  disabled={!resultData?.annotated_image_b64 || loading}
                >
                  <i className="fa-solid fa-rotate-left"></i>
                </button>
                <button
                  className="zoom-btn modal-btn"
                  onClick={() => setIsModalOpen(true)}
                  title="Expand Fullscreen Lightbox"
                  disabled={!resultData?.annotated_image_b64 || loading}
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
                position: 'relative',
                cursor: zoomScale > 1 ? (isDragging ? 'grabbing' : 'grab') : 'default'
              }}
            >
              {loading && (
                <div className="anpr-loading-overlay">
                  <div className="spinner-ring"></div>
                  <div className="loading-status-text">
                    <i className="fa-solid fa-gear fa-spin"></i> Running ANPR on {selectedImage}...
                  </div>
                  <div className="loading-sub-text">
                    YOLO Bounding Box Detection • 5x Super-Resolution Upscaling • CLAHE & Positional Indian Grammar Verification
                  </div>
                  <div className="loading-progress-bar">
                    <div className="progress-fill"></div>
                  </div>
                </div>
              )}

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

      {/* Pop-Up Wide Results Modal with Background Blur */}
      {isResultsModalOpen && resultData && resultData.detections && resultData.detections.length > 0 && (
        <div
          className="anpr-results-modal-backdrop"
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(11, 15, 25, 0.85)',
            backdropFilter: 'blur(12px)',
            WebkitBackdropFilter: 'blur(12px)',
            zIndex: 9999,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '20px',
            overflowY: 'auto'
          }}
          onClick={(e) => {
            if (e.target.classList.contains('anpr-results-modal-backdrop')) {
              setIsResultsModalOpen(false);
            }
          }}
        >
          <div
            className="anpr-results-modal-card"
            style={{
              width: '100%',
              maxWidth: '1050px',
              maxHeight: '90vh',
              overflowY: 'auto',
              background: '#0f172a',
              border: '1px solid #0284c7',
              boxShadow: '0 25px 60px rgba(0, 0, 0, 0.8), 0 0 35px rgba(2, 132, 199, 0.3)',
              borderRadius: '16px',
              padding: '24px',
              margin: 'auto',
              display: 'flex',
              flexDirection: 'column',
              gap: '20px',
              color: '#f8fafc'
            }}
          >
            {/* Modal Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid rgba(255, 255, 255, 0.1)', paddingBottom: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
                <div style={{ width: '44px', height: '44px', borderRadius: '12px', background: 'rgba(2, 132, 199, 0.2)', border: '1px solid #0284c7', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#38bdf8', fontSize: '1.3rem' }}>
                  <i className="fa-solid fa-list-check"></i>
                </div>
                <div>
                  <h2 style={{ fontSize: '1.3rem', fontWeight: 700, margin: 0, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '10px' }}>
                    ANPR Recognition & Inspection Output
                    <span style={{ background: '#0284c7', color: '#fff', fontSize: '0.78rem', padding: '3px 10px', borderRadius: '20px', fontWeight: 600 }}>
                      {resultData.detections.length} {resultData.detections.length === 1 ? 'Vehicle Detected' : 'Vehicles Detected'}
                    </span>
                  </h2>
                  <span style={{ fontSize: '0.82rem', color: '#94a3b8' }}>
                    Image: <b>{resultData.filename || selectedImage}</b> • Location: <b>{resultData.target_city || 'Mumbai'} Camera Grid</b>
                  </span>
                </div>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
                <span className="time-badge" style={{ padding: '6px 12px', background: 'rgba(2, 132, 199, 0.15)', border: '1px solid #0284c7', borderRadius: '6px', color: '#38bdf8', fontSize: '0.85rem', fontWeight: 600 }}>
                  <i className="fa-solid fa-stopwatch"></i> Latency: {latency !== null ? `${latency} ms` : '-- ms'}
                </span>
                <button
                  onClick={() => setIsResultsModalOpen(false)}
                  style={{
                    background: 'rgba(255, 255, 255, 0.1)',
                    border: '1px solid rgba(255, 255, 255, 0.2)',
                    color: '#f8fafc',
                    width: '36px',
                    height: '36px',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    fontSize: '1.1rem',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    transition: 'all 0.2s ease'
                  }}
                  title="Close Modal (Esc)"
                >
                  <i className="fa-solid fa-xmark"></i>
                </button>
              </div>
            </div>

            {/* Modal Body: Detections Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: resultData.detections.length > 1 ? 'repeat(auto-fit, minmax(460px, 1fr))' : '1fr', gap: '20px' }}>
              {resultData.detections.map((det, index) => (
                <div
                  key={index}
                  style={{
                    background: 'rgba(30, 41, 59, 0.7)',
                    border: '1px solid rgba(56, 189, 248, 0.3)',
                    borderRadius: '12px',
                    padding: '20px',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '16px',
                    boxShadow: '0 8px 24px rgba(0, 0, 0, 0.3)'
                  }}
                >
                  {/* Plate Badge & Type Header */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <span style={{ fontSize: '1.6rem', fontWeight: 800, fontFamily: 'var(--font-mono)', background: 'linear-gradient(135deg, #0284c7, #38bdf8)', color: '#ffffff', padding: '6px 16px', borderRadius: '8px', letterSpacing: '1px', border: '1px solid #38bdf8', boxShadow: '0 4px 14px rgba(2, 132, 199, 0.4)' }}>
                        {det.plate_text}
                      </span>
                      <span style={{ background: 'rgba(15, 23, 42, 0.8)', border: '1px solid #334155', color: '#cbd5e1', padding: '4px 10px', borderRadius: '6px', fontSize: '0.78rem', fontWeight: 700, display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
                        <i className={getVehicleIcon(det.vehicle_type)}></i> {det.vehicle_type?.toUpperCase() || 'VEHICLE'}
                      </span>
                    </div>

                    <div style={{ textAlign: 'right' }}>
                      <div style={{ fontSize: '1.1rem', fontWeight: 800, color: '#38bdf8' }}>
                        {(det.confidence * 100).toFixed(1)}%
                      </div>
                      <div style={{ fontSize: '0.72rem', color: '#94a3b8' }}>Overall Confidence</div>
                    </div>
                  </div>

                  {/* 3 Image Crop Previews Grid */}
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px', background: 'rgba(15, 23, 42, 0.6)', padding: '12px', borderRadius: '10px', border: '1px solid #334155' }}>
                    {/* 1. Vehicle Zoom Crop */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', alignItems: 'center' }}>
                      <span style={{ fontSize: '0.72rem', color: '#94a3b8', fontWeight: 600 }}>Vehicle ROI</span>
                      <div style={{ width: '100%', height: '80px', background: '#020617', borderRadius: '6px', border: '1px solid #334155', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
                        {det.zoomed_vehicle_b64 ? (
                          <img src={`data:image/jpeg;base64,${det.zoomed_vehicle_b64}`} alt="Vehicle Crop" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                        ) : (
                          <span style={{ fontSize: '0.7rem', color: '#64748b' }}>N/A</span>
                        )}
                      </div>
                    </div>

                    {/* 2. License Plate Crop */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', alignItems: 'center' }}>
                      <span style={{ fontSize: '0.72rem', color: '#94a3b8', fontWeight: 600 }}>Original Plate Crop</span>
                      <div style={{ width: '100%', height: '80px', background: '#020617', borderRadius: '6px', border: '1px solid #334155', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
                        {det.crop_b64 ? (
                          <img src={`data:image/jpeg;base64,${det.crop_b64}`} alt="Plate Crop" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                        ) : (
                          <span style={{ fontSize: '0.7rem', color: '#64748b' }}>N/A</span>
                        )}
                      </div>
                    </div>

                    {/* 3. 5x Super-Res Lens */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', alignItems: 'center' }}>
                      <span style={{ fontSize: '0.72rem', color: '#38bdf8', fontWeight: 700 }}>5x Super-Res Lens</span>
                      <div style={{ width: '100%', height: '80px', background: '#020617', borderRadius: '6px', border: '1px solid #0284c7', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
                        {det.zoomed_plate_b64 ? (
                          <img src={`data:image/jpeg;base64,${det.zoomed_plate_b64}`} alt="Super Res Plate" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                        ) : (
                          <span style={{ fontSize: '0.7rem', color: '#64748b' }}>N/A</span>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* 4-Step Pipeline Metrics */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '0.82rem', color: '#cbd5e1', background: 'rgba(15, 23, 42, 0.4)', padding: '12px', borderRadius: '8px' }}>
                    <div><b>Step 1 (Vehicle):</b> <i className={getVehicleIcon(det.vehicle_type)} style={{ marginRight: '4px', color: '#38bdf8' }}></i> {det.vehicle_type?.toUpperCase() || 'VEHICLE'} ({(det.vehicle_confidence * 100).toFixed(1)}%)</div>
                    <div><b>Step 2 (Plate Box):</b> [{det.bbox?.join(', ')}] • DetConf: <b>{(det.det_confidence * 100).toFixed(1)}%</b></div>
                    <div>
                      <b>Step 3 (OCR & Grammar):</b> OCR Conf: <b style={{ color: '#38bdf8' }}>{(det.ocr_confidence * 100).toFixed(1)}%</b>
                      {det.confidence_flag ? (
                        <span style={{ background: 'rgba(74, 222, 128, 0.2)', color: '#4ade80', padding: '2px 8px', borderRadius: '4px', fontSize: '0.75rem', marginLeft: '8px', fontWeight: 600 }}>
                          <i className="fa-solid fa-circle-check"></i> Pattern Validated
                        </span>
                      ) : (
                        <span style={{ background: 'rgba(249, 115, 22, 0.2)', color: '#f97316', padding: '2px 8px', borderRadius: '4px', fontSize: '0.75rem', marginLeft: '8px', fontWeight: 600 }}>
                          <i className="fa-solid fa-triangle-exclamation"></i> Format Warning (Capped)
                        </span>
                      )}
                    </div>
                    <div><b>Step 4 (Database Storage):</b> <span style={{ color: '#4ade80', fontWeight: 600 }}><i className="fa-solid fa-database"></i> Saved to SQLite Database</span></div>
                  </div>

                  {det.alert && (
                    <div className="alert-banner" style={{ background: 'rgba(239, 68, 68, 0.2)', borderColor: '#ef4444', color: '#fca5a5', fontSize: '0.82rem', padding: '10px', borderRadius: '8px' }}>
                      <i className="fa-solid fa-triangle-exclamation"></i> <b>SECURITY ALERT:</b> {det.alert.reason} ({det.alert.risk_level})
                    </div>
                  )}

                  {/* Action Buttons */}
                  <div style={{ display: 'flex', gap: '10px', marginTop: '4px' }}>
                    <button
                      onClick={() => {
                        setIsResultsModalOpen(false);
                        if (onViewTrajectory) onViewTrajectory(det.plate_text, resultData?.target_city || 'Mumbai');
                      }}
                      style={{ flex: 1, padding: '10px 14px', background: '#0284c7', color: '#fff', border: 'none', borderRadius: '8px', cursor: 'pointer', fontWeight: 600, fontSize: '0.85rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
                    >
                      <i className="fa-solid fa-route"></i> Step 4: View Route Trajectory
                    </button>
                    <button
                      onClick={() => window.open(`/api/reports/pdf?plate=${encodeURIComponent(det.plate_text)}`)}
                      style={{ padding: '10px 14px', background: 'rgba(255, 255, 255, 0.1)', color: '#f8fafc', border: '1px solid rgba(255, 255, 255, 0.2)', borderRadius: '8px', cursor: 'pointer', fontWeight: 600, fontSize: '0.85rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
                    >
                      <i className="fa-solid fa-file-pdf"></i> PDF
                    </button>
                  </div>
                </div>
              ))}
            </div>

            {/* Modal Footer */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderTop: '1px solid rgba(255, 255, 255, 0.1)', paddingTop: '16px' }}>
              <span style={{ fontSize: '0.8rem', color: '#64748b' }}>
                TrackNet AI ANPR Engine • Multi-Vehicle Spatial Recognition
              </span>
              <button
                onClick={() => setIsResultsModalOpen(false)}
                style={{ padding: '10px 24px', background: '#334155', color: '#fff', border: 'none', borderRadius: '8px', cursor: 'pointer', fontWeight: 600, fontSize: '0.88rem' }}
              >
                Close Results Window
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

