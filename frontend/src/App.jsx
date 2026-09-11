import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import Navbar from './components/Navbar';
import ANPRHub from './components/ANPRHub';
import MultiCameraTrajectory from './components/MultiCameraTrajectory';
import MacroAnalytics from './components/MacroAnalytics';
import BlacklistAlerts from './components/BlacklistAlerts';

export default function App() {
  const [activeTab, setActiveTab] = useState('anpr-hub');
  const [currentCity, setCurrentCity] = useState('Mumbai');
  const [cityCameras, setCityCameras] = useState([]);
  const [activeTrajectory, setActiveTrajectory] = useState(null);

  // Load cameras for selected city
  const loadCityCameras = async (cityName) => {
    try {
      const resp = await fetch(`/api/cameras?city=${encodeURIComponent(cityName)}`);
      const data = await resp.json();
      if (data.success) {
        setCityCameras(data.cameras || []);
      }
    } catch (e) {
      console.error('Failed to load city cameras:', e);
    }
  };

  useEffect(() => {
    loadCityCameras(currentCity);
  }, [currentCity]);

  const handleCityChange = (newCity) => {
    setCurrentCity(newCity);
  };

  const handleSyncOSM = async () => {
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
      } else {
        alert('OSM Sync Failed: ' + data.error);
      }
    } catch (e) {
      console.error('OSM Sync Error:', e);
      alert('Failed to connect to Overpass synchronization service.');
    }
  };

  const fetchTrajectory = async (plateText) => {
    try {
      const resp = await fetch(`/api/vehicles/${encodeURIComponent(plateText)}/trajectory`);
      const data = await resp.json();
      if (data.success) {
        setActiveTrajectory(data.trajectory);
      }
    } catch (e) {
      console.error('Failed to fetch vehicle trajectory:', e);
    }
  };

  const handleANPRSuccess = (plateText) => {
    if (plateText) {
      fetchTrajectory(plateText);
    }
  };

  return (
    <div className="dark-theme" style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Header
        currentCity={currentCity}
        onCityChange={handleCityChange}
        cameraCount={cityCameras.length}
        onSyncOSM={handleSyncOSM}
      />
      <Navbar activeTab={activeTab} onTabChange={setActiveTab} />
      <main className="app-container" style={{ flex: 1 }}>
        {activeTab === 'anpr-hub' && (
          <ANPRHub cityCameras={cityCameras} onANPRSuccess={handleANPRSuccess} />
        )}
        {activeTab === 'trajectory-tracker' && (
          <MultiCameraTrajectory
            activeTrajectory={activeTrajectory}
            currentCity={currentCity}
            cityCameras={cityCameras}
          />
        )}
        {activeTab === 'macro-analytics' && (
          <MacroAnalytics currentCity={currentCity} />
        )}
        {activeTab === 'alert-watchdog' && (
          <BlacklistAlerts />
        )}
      </main>
    </div>
  );
}
