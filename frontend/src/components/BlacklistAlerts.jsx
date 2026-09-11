import React, { useState, useEffect } from 'react';

export default function BlacklistAlerts() {
  const [blacklist, setBlacklist] = useState([]);
  const [plateInput, setPlateInput] = useState('');
  const [reasonInput, setReasonInput] = useState('');
  const [riskInput, setRiskInput] = useState('HIGH');
  const [showModal, setShowModal] = useState(false);

  const loadBlacklist = () => {
    fetch('/api/alerts/blacklist')
      .then(res => res.json())
      .then(data => {
        if (data.success) setBlacklist(data.blacklist || []);
      })
      .catch(err => console.error('Failed to load blacklist:', err));
  };

  useEffect(() => {
    loadBlacklist();
  }, []);

  const handleAddBlacklist = async (e) => {
    e.preventDefault();
    if (!plateInput) return;

    try {
      const resp = await fetch('/api/alerts/blacklist', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          plate_number: plateInput,
          reason: reasonInput || 'Stolen Vehicle Alert',
          risk_level: riskInput
        })
      });
      const data = await resp.json();
      if (data.success) {
        setPlateInput('');
        setReasonInput('');
        setShowModal(false);
        loadBlacklist();
      } else {
        alert('Error adding blacklist: ' + data.error);
      }
    } catch (err) {
      console.error('Failed to add to blacklist:', err);
    }
  };

  return (
    <section id="alert-watchdog" className="tab-content active">
      <div className="alert-grid">
        {/* Blacklist Control Panel */}
        <div className="panel blacklist-panel">
          <div className="panel-header">
            <h2><i className="fa-solid fa-[#ef4444] fa-triangle-exclamation"></i> Blacklisted Target Vehicles</h2>
            <button className="btn btn-primary btn-sm" onClick={() => setShowModal(true)}>
              <i className="fa-solid fa-plus"></i> Add Blacklist Plate
            </button>
          </div>

          <div className="table-responsive mt-3">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Plate Number</th>
                  <th>Reason / Offense</th>
                  <th>Risk Level</th>
                  <th>Date Added</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {blacklist.map((item, idx) => (
                  <tr key={idx}>
                    <td><span className="plate-badge">{item.plate_number}</span></td>
                    <td>{item.reason}</td>
                    <td>
                      <span className={`risk-tag ${item.risk_level?.toLowerCase() || 'high'}`}>
                        {item.risk_level}
                      </span>
                    </td>
                    <td>{item.created_at || 'Recently'}</td>
                    <td><span className="status-dot-active">Active Watch</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Real-time Alert Feed */}
        <div className="panel feed-panel">
          <div className="panel-header">
            <h2><i className="fa-solid fa-[#0284c7] fa-bell"></i> Real-time ANPR Alert Watchdog Feed</h2>
          </div>

          <div className="feed-list mt-3">
            {blacklist.slice(0, 4).map((item, idx) => (
              <div key={idx} className="feed-item high-risk">
                <div className="feed-icon"><i className="fa-solid fa-[#ef4444] fa-shield-cat"></i></div>
                <div className="feed-content">
                  <div className="feed-title">
                    Target Vehicle Detected: <span className="highlight">{item.plate_number}</span>
                  </div>
                  <div className="feed-desc">
                    Reason: <b>{item.reason}</b> &bull; Priority: <b style={{ color: '#ef4444' }}>{item.risk_level}</b>
                  </div>
                  <div className="feed-time">OpenStreetMap Camera Network Surveillance Watchdog Active</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Add Modal */}
      {showModal && (
        <div className="modal-overlay active">
          <div className="modal-content" style={{ maxWidth: '420px', padding: '24px' }}>
            <div className="modal-header" style={{ marginBottom: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h3 style={{ fontSize: '1.1rem', color: '#f1f5f9' }}><i className="fa-solid fa-plus" style={{ color: '#0284c7' }}></i> Add Blacklisted Vehicle</h3>
              <button onClick={() => setShowModal(false)} style={{ background: 'none', border: 'none', color: '#94a3b8', fontSize: '1.2rem', cursor: 'pointer' }}>&times;</button>
            </div>
            <form onSubmit={handleAddBlacklist}>
              <div className="form-group mb-3">
                <label style={{ display: 'block', marginBottom: '6px', fontSize: '0.82rem', color: '#cbd5e1' }}>Vehicle License Plate Number:</label>
                <input
                  type="text"
                  className="select-input"
                  placeholder="e.g. MH02GO7249"
                  value={plateInput}
                  onChange={(e) => setPlateInput(e.target.value.toUpperCase())}
                  required
                />
              </div>
              <div className="form-group mb-3">
                <label style={{ display: 'block', marginBottom: '6px', fontSize: '0.82rem', color: '#cbd5e1' }}>Reason / Alert Description:</label>
                <input
                  type="text"
                  className="select-input"
                  placeholder="e.g. Stolen Vehicle Alert"
                  value={reasonInput}
                  onChange={(e) => setReasonInput(e.target.value)}
                  required
                />
              </div>
              <div className="form-group mb-4">
                <label style={{ display: 'block', marginBottom: '6px', fontSize: '0.82rem', color: '#cbd5e1' }}>Threat Risk Level:</label>
                <select
                  className="select-input"
                  value={riskInput}
                  onChange={(e) => setRiskInput(e.target.value)}
                >
                  <option value="CRITICAL">CRITICAL</option>
                  <option value="HIGH">HIGH</option>
                  <option value="MEDIUM">MEDIUM</option>
                  <option value="LOW">LOW</option>
                </select>
              </div>
              <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end' }}>
                <button type="button" className="btn btn-secondary" onClick={() => setShowModal(false)}>Cancel</button>
                <button type="submit" className="btn btn-primary">Save to Blacklist</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </section>
  );
}
