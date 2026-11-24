import threading
import webview
from werkzeug.serving import make_server
from app import app

server = make_server(
    "127.0.0.1",
    5000,
    app,
    threaded=True
)

thread = threading.Thread(
    target=server.serve_forever,
    daemon=True
)
thread.start()

webview.create_window(
    "Fake News Detection",
    "http://127.0.0.1:5000",
    width=1200,
    height=850,
    min_size=(900, 650)
)

try:
    webview.start()
finally:
    server.shutdown()
