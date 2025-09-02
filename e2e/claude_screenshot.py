#!/usr/bin/env python3
"""
Screenshot tool for AI development workflow
Outputs compressed base64 JPEG that can be pasted directly into Claude conversations
With tail mode for continuous monitoring during development
"""

import subprocess
import tempfile
import os
import time
import base64
import sys

def take_screenshot_base64(label="SCREENSHOT"):
    """Take screenshot and output as base64 JPEG for Claude"""
    
    # Create temp PNG file first
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_png:
        png_path = tmp_png.name
    
    try:
        # Use macOS screencapture command to get PNG
        result = subprocess.run(['screencapture', '-x', '-t', 'png', png_path], 
                              capture_output=True, text=True)
        
        if result.returncode != 0 or not os.path.exists(png_path) or os.path.getsize(png_path) == 0:
            print(f"Screenshot failed: {result.stderr}")
            return None
        
        # Create temp JPEG path
        jpg_path = png_path.replace('.png', '.jpg')
        
        # Convert PNG to compressed JPEG using sips
        # 50% quality for good balance of size/quality
        convert_result = subprocess.run([
            'sips', '-s', 'format', 'jpeg', 
            '-s', 'formatOptions', '50',
            png_path, '--out', jpg_path
        ], capture_output=True, text=True)
        
        if convert_result.returncode != 0:
            print(f"JPEG conversion failed: {convert_result.stderr}")
            return None
        
        # Read JPEG and encode as base64
        with open(jpg_path, 'rb') as f:
            jpg_data = f.read()
        
        b64_data = base64.b64encode(jpg_data).decode('utf-8')
        
        # Simple output with timestamp
        timestamp = time.strftime("%H:%M:%S")
        print(f"\n[{timestamp}] {label}:")
        print(f"data:image/jpeg;base64,{b64_data}")
        sys.stdout.flush()
        
        return b64_data
        
    finally:
        # Clean up temp files
        if os.path.exists(png_path):
            os.unlink(png_path)
        if 'jpg_path' in locals() and os.path.exists(jpg_path):
            os.unlink(jpg_path)

def tail_mode(interval=5, max_screenshots=None):
    """Continuously take screenshots for development monitoring"""
    print(f"Starting screenshot tail mode (every {interval}s)")
    if max_screenshots:
        print(f"Will take {max_screenshots} screenshots total")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    counter = 0
    try:
        while True:
            if max_screenshots and counter >= max_screenshots:
                print(f"\n[{time.strftime('%H:%M:%S')}] Reached maximum {max_screenshots} screenshots, stopping")
                break
                
            b64_data = take_screenshot_base64(f"TAIL_{counter:03d}")
            if b64_data:
                counter += 1
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print(f"\n[{time.strftime('%H:%M:%S')}] Stopping screenshot tail mode")
        print(f"Captured {counter} screenshots")

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "tail":
            interval = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            max_shots = int(sys.argv[3]) if len(sys.argv) > 3 else None
            tail_mode(interval, max_shots)
        elif sys.argv[1] == "help":
            print("Usage:")
            print("  python3 claude_screenshot.py              # Take single screenshot")
            print("  python3 claude_screenshot.py tail         # Tail mode every 5 seconds")
            print("  python3 claude_screenshot.py tail 10      # Tail mode every 10 seconds")
            print("  python3 claude_screenshot.py tail 10 3    # Take 3 screenshots, 10 seconds apart")
            print("")
            print("Output is base64 JPEG that you can copy/paste directly into Claude conversations")
        else:
            print("Unknown option. Use 'help' for usage.")
    else:
        # Give 3 second delay for single shots
        print("Taking screenshot in 3 seconds...")
        time.sleep(3)
        take_screenshot_base64("MANUAL")

if __name__ == "__main__":
    main()