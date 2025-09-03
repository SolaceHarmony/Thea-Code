#!/usr/bin/env python3
"""
UI Debug utility for streaming screenshots as base64 to console
Perfect for tailing during development
"""

import subprocess
import tempfile
import os
import base64
import time
import sys

def screenshot_to_base64():
    """Take screenshot and return as base64 string"""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as tmp:
        tmp_path = tmp.name
        
        # Take screenshot
        result = subprocess.run(['screencapture', '-x', '-t', 'png', tmp_path], 
                              capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(tmp_path):
            with open(tmp_path, 'rb') as f:
                img_data = f.read()
                return base64.b64encode(img_data).decode('utf-8')
        return None

def debug_screenshot(label="UI_DEBUG"):
    """Print screenshot as base64 - perfect for console streaming"""
    b64_data = screenshot_to_base64()
    if b64_data:
        print(f"=== {label} SCREENSHOT START ===")
        print(f"data:image/png;base64,{b64_data}")
        print(f"=== {label} SCREENSHOT END ===")
        sys.stdout.flush()  # Ensure immediate output
    else:
        print(f"=== {label} SCREENSHOT FAILED ===")

def watch_mode(interval=2):
    """Continuously take screenshots for development"""
    print(f"Starting UI debug watch mode (every {interval}s)")
    print("Press Ctrl+C to stop")
    
    try:
        counter = 0
        while True:
            debug_screenshot(f"WATCH_{counter}")
            time.sleep(interval)
            counter += 1
    except KeyboardInterrupt:
        print("\nStopping UI debug watch mode")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "watch":
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        watch_mode(interval)
    else:
        debug_screenshot("MANUAL")