#!/usr/bin/env python3
"""
Auto-deployment script for Yoga Persian Assistant Backend
Monitors file changes and automatically commits and pushes to GitHub
"""

import os
import time
import subprocess
import sys
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class BackendChangeHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_commit_time = 0
        self.commit_delay = 5  # Wait 5 seconds before committing to batch changes
        
    def on_modified(self, event):
        if event.is_directory:
            return
            
        # Only monitor Python files and important config files
        if event.src_path.endswith(('.py', '.txt', '.yml', '.yaml', '.json', '.md')):
            self.schedule_commit()
    
    def on_created(self, event):
        if event.is_directory:
            return
            
        if event.src_path.endswith(('.py', '.txt', '.yml', '.yaml', '.json', '.md')):
            self.schedule_commit()
    
    def schedule_commit(self):
        current_time = time.time()
        if current_time - self.last_commit_time > self.commit_delay:
            self.last_commit_time = current_time
            # Schedule commit after delay
            time.sleep(self.commit_delay)
            self.commit_and_push()
    
    def commit_and_push(self):
        try:
            print(f"\n🔄 Detected changes - Committing and pushing...")
            
            # Add all changes
            subprocess.run(['git', 'add', '.'], check=True, capture_output=True)
            
            # Check if there are changes to commit
            result = subprocess.run(['git', 'diff', '--cached', '--quiet'], capture_output=True)
            if result.returncode == 0:
                print("No changes to commit")
                return
            
            # Commit with timestamp
            commit_message = f"Auto-commit: {time.strftime('%Y-%m-%d %H:%M:%S')}"
            subprocess.run(['git', 'commit', '-m', commit_message], check=True, capture_output=True)
            
            # Push to origin
            subprocess.run(['git', 'push', 'origin', 'master'], check=True, capture_output=True)
            
            print(f"✅ Successfully pushed changes to GitHub!")
            print(f"📝 Commit: {commit_message}")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error during commit/push: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

def main():
    print("🚀 Starting auto-deployment monitor for Yoga Persian Assistant Backend")
    print("📁 Monitoring directory:", os.getcwd())
    print("⏰ Auto-commit delay: 5 seconds")
    print("🔄 Press Ctrl+C to stop monitoring")
    print("-" * 60)
    
    # Check if we're in a git repository
    if not os.path.exists('.git'):
        print("❌ Not in a git repository. Please run this from the backend directory.")
        sys.exit(1)
    
    # Check if we have a remote origin
    try:
        result = subprocess.run(['git', 'remote', 'get-url', 'origin'], 
                              capture_output=True, text=True, check=True)
        print(f"🔗 Remote repository: {result.stdout.strip()}")
    except subprocess.CalledProcessError:
        print("❌ No remote origin found. Please set up the remote repository first.")
        sys.exit(1)
    
    # Set up file monitoring
    event_handler = BackendChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, '.', recursive=True)
    
    try:
        observer.start()
        print("👀 Monitoring for file changes...")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping auto-deployment monitor...")
        observer.stop()
        print("✅ Monitor stopped successfully")
    
    observer.join()

if __name__ == "__main__":
    main()
