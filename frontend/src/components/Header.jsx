import React, { useState } from 'react';

export default function Header({ currentCity, onCityChange, cameraCount, onSyncOSM }) {
  const [isSyncing, setIsSyncing] = useState(false);

  const handleSync = async () => {
    setIsSyncing(true);
    await onSyncOSM();
    setIsSyncing(false);
  };

  return (
    <header className="app-header">
      <div className="logo-area">
        <div className="logo-icon"><i className="fa-solid fa-crosshairs"></i></div>
        <div className="logo-text">
          <h1>TrackNet <span>AI</span></h1>
          <p>Multi-Camera Trajectory Tracking & Urban Traffic Analytics</p>
        </div>
      </div>

      <div className="city-selector-bar">
        <label htmlFor="global-city-select"><i className="fa-solid fa-city"></i> Region:</label>
        <select
          id="global-city-select"
          className="city-dropdown"
          value={currentCity}
          onChange={(e) => onCityChange(e.target.value)}
        >
          <option value="Mumbai">Mumbai (Maharashtra)</option>
          <option value="Pune">Pune (Maharashtra)</option>
          <option value="Ahmedabad">Ahmedabad (Gujarat)</option>
          <option value="Gandhinagar">Gandhinagar (Gujarat)</option>
          <option value="Surat">Surat (Gujarat)</option>
          <option value="Vadodara">Vadodara (Gujarat)</option>
          <option value="Rajkot">Rajkot (Gujarat)</option>
        </select>
        <button
          id="btn-sync-osm"
          className="sync-btn"
          disabled={isSyncing}
          onClick={handleSync}
          title="Sync OpenStreetMap Cameras for Region"
        >
          {isSyncing ? (
            <><i className="fa-solid fa-spinner fa-spin"></i> Syncing...</>
          ) : (
            <><i className="fa-solid fa-rotate"></i> Sync OSM Cameras</>
          )}
        </button>
      </div>

      <div className="header-status">
        <div className="status-badge live-dot" id="header-camera-count-badge">
          <span className="dot"></span> <span id="header-cam-count">{cameraCount}</span> OSM NODES SYNCED
        </div>
        <div className="status-badge sys-ok">
          <i className="fa-solid fa-microchip"></i> OCR ENGINE: ONLINE
        </div>
      </div>
    </header>
  );
}
