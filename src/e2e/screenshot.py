#!/usr/bin/env python3
"""
Simple screenshot tool that captures the screen and saves it as a temporary file
Then outputs the path so it can be read by Claude
"""

import sys
import base64
import tempfile
import os

def take_screenshot():
    try:
        # Try using pyautogui first (cross-platform)
        import pyautogui
        import PIL.Image
        
        # Take screenshot
        screenshot = pyautogui.screenshot()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            screenshot.save(tmp.name, 'PNG')
            print(f"Screenshot saved to: {tmp.name}")
            return tmp.name
            
    except ImportError:
        # Fallback to macOS screencapture command
        import subprocess
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        
        # Use macOS screencapture command
        # -x: no sound, -i: interactive mode (select window/area)
        result = subprocess.run(['screencapture', '-x', '-i', tmp_path], 
                              capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
            print(f"Screenshot saved to: {tmp_path}")
            return tmp_path
        else:
            print("Screenshot cancelled or failed")
            return None

def main():
    print("Taking screenshot...")
    print("(On macOS: Click a window or drag to select an area)")
    
    screenshot_path = take_screenshot()
    
    if screenshot_path:
        print(f"\nYou can now read the screenshot with:")
        print(f"Read tool: {screenshot_path}")
    else:
        print("No screenshot taken")
        sys.exit(1)

if __name__ == "__main__":
    main()