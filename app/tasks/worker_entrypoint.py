import threading
import http.server
import socketserver
import os
import subprocess
import sys

def run_health_check_server():
    PORT = int(os.environ.get("PORT", 8080))
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Health check server serving at port {PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    # Start health check server in a background thread
    threading.Thread(target=run_health_check_server, daemon=True).start()
    
    # Start Celery worker
    # Using subprocess to run celery so it becomes the main process logic
    print("Starting Celery worker...")
    subprocess.run([
        "celery", 
        "-A", "app.tasks.celery_app", 
        "worker", 
        "--loglevel=info", 
        "--concurrency=1"
    ])
