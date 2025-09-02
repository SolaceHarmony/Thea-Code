#!/usr/bin/env python3
"""
Simple screenshot tool that captures the entire screen automatically
"""

import subprocess
import tempfile
import os
import time

def take_screenshot():
    """Take a screenshot of the entire screen on macOS"""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
    
    # Small delay to let you arrange windows if needed
    print("Taking screenshot in 3 seconds...")
    time.sleep(3)
    
    # Use macOS screencapture command
    # -x: no sound, -t png: PNG format, no -i for automatic full screen
    result = subprocess.run(['screencapture', '-x', '-t', 'png', tmp_path], 
                          capture_output=True, text=True)
    
    if result.returncode == 0 and os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
        print(f"Screenshot saved to: {tmp_path}")
        return tmp_path
    else:
        print(f"Screenshot failed: {result.stderr}")
        return None

def main():
    screenshot_path = take_screenshot()
    
    if screenshot_path:
        print(f"\nScreenshot ready!")
        print(f"Path: {screenshot_path}")
        print(f"\nTo view it in Claude, use the Read tool with the path above")
    else:
        print("Failed to take screenshot")

if __name__ == "__main__":
    main()