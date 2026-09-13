"""
================================================================================
File: app.py (Root entrypoint wrapper for Cloud Hosting)
Project: TrackNet AI
Purpose: Exposes backend.app as top-level WSGI entrypoint for Render/Gunicorn.
================================================================================
"""

import sys
import os

# Ensure backend directory is in Python path for imports to resolve
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend'))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from backend.app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
