import React, { useState, useEffect } from 'react';

export default function ANPRHub({ cityCameras, onANPRSuccess }) {
  const [testImages, setTestImages] = useState([]);
  const [selectedImage, setSelectedImage] = useState('');
  const [selectedCamera, setSelectedCamera] = useState('');
  const [loading, setLoading] = useState(false);
  const [resultData, setResultData] = useState(null);
  const [latency, setLatency] = useState(null);

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

  const handleRunANPR = async () => {
    if (!selectedImage) return;
    setLoading(true);
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

  return (
    <section id="anpr-hub" className="tab-content active">
      <div className="grid-layout">
        {/* Left Controls Panel */}
        <div className="panel control-panel">
          <div className="panel-header">
            <h2><i className="fa-solid fa-[#0284c7] fa-sliders"></i> Input & Controls</h2>
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
              {loading ? <i className="fa-solid fa-spinner fa-spin"></i> : <i className="fa-solid fa-play"></i>} Run ANPR on Test Image
            </button>
          </div>

          <div className="form-group mt-4 mb-4">
            <label htmlFor="camera-node-select">Simulated Camera Location Node:</label>
            <select
              id="camera-node-select"
              className="select-input"
              value={selectedCamera}
              onChange={(e) => setSelectedCamera(e.target.value)}
            >
              {cityCameras.map(c => (
                <option key={c.camera_id} value={c.camera_id}>
                  {c.camera_id}: {c.location_description} [{c.camera_type}]
                </option>
              ))}
            </select>
          </div>

          <div className="info-box mt-4">
            <h4><i className="fa-solid fa-circle-info"></i> Engine Pipeline Info</h4>
            <p><b>Primary OCR:</b> EasyOCR High-Precision Engine</p>
            <p><b>Fallback OCR:</b> Multi-Variant Super-Resolution Engine</p>
            <p><b>Preprocessors:</b> CLAHE, BlackHat, Sobel, Adaptive Thresh</p>
            <p><b>Format Rule:</b> Indian Registration Syntax Check</p>
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
            <div className="image-toolbar">
              <span>Detection & Bounding Box Overlay</span>
            </div>
            <div id="image-container" className="image-wrapper">
              {resultData && resultData.annotated_image_b64 ? (
                <img
                  src={`data:image/jpeg;base64,${resultData.annotated_image_b64}`}
                  className="result-img"
                  alt="Detection Output"
                />
              ) : (
                <div className="placeholder-text">
                  <i className="fa-solid fa-image"></i> Select a test image to run ANPR engine
                </div>
              )}
            </div>
          </div>

          <div className="results-cards-area">
            <h3>Recognized Output</h3>
            <div id="output-cards-container" className="cards-list">
              {resultData && resultData.detections && resultData.detections.length > 0 ? (
                resultData.detections.map((det, idx) => (
                  <div key={idx} className="result-card">
                    <div className="plate-badge-container">
                      {det.crop_b64 && (
                        <img
                          src={`data:image/jpeg;base64,${det.crop_b64}`}
                          className="plate-crop-img"
                          alt="Plate Crop"
                        />
                      )}
                      <div className="plate-string">{det.plate_text}</div>
                    </div>
                    <div className="card-metrics">
                      <span>Confidence: <b>{Math.round(det.confidence * 100)}%</b></span>
                      <span>Box: [{det.bbox.join(', ')}]</span>
                    </div>
                    {det.alert && (
                      <div className="alert-banner">
                        <i className="fa-solid fa-triangle-exclamation"></i> ALERT: {det.alert.reason} ({det.alert.risk_level})
                      </div>
                    )}
                  </div>
                ))
              ) : (
                <div className="no-data-msg">No active detections</div>
              )}
            </div>

            <h3 className="mt-3">OpenCV Preprocessing Pipeline Steps</h3>
            <div id="preprocessing-steps-grid" className="prep-grid">
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
    </section>
  );
}
