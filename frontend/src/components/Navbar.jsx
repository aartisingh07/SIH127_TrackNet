import React from 'react';

export default function Navbar({ activeTab, onTabChange }) {
  return (
    <nav className="main-nav">
      <button
        className={`nav-tab ${activeTab === 'anpr-hub' ? 'active' : ''}`}
        onClick={() => onTabChange('anpr-hub')}
      >
        <i className="fa-solid fa-eye"></i> ANPR & OCR Test Hub
      </button>
      <button
        className={`nav-tab ${activeTab === 'trajectory-tracker' ? 'active' : ''}`}
        onClick={() => onTabChange('trajectory-tracker')}
      >
        <i className="fa-solid fa-route"></i> Multi-Camera Trajectory
      </button>
      <button
        className={`nav-tab ${activeTab === 'macro-analytics' ? 'active' : ''}`}
        onClick={() => onTabChange('macro-analytics')}
      >
        <i className="fa-solid fa-chart-line"></i> Macro Traffic Analytics
      </button>
      <button
        className={`nav-tab ${activeTab === 'alert-watchdog' ? 'active' : ''}`}
        onClick={() => onTabChange('alert-watchdog')}
      >
        <i className="fa-solid fa-shield-halved"></i> Blacklist & Alerts
      </button>
    </nav>
  );
}
