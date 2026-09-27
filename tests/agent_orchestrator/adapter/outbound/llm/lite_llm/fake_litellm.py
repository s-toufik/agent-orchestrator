import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

parser = argparse.ArgumentParser()
parser.add_argument("--config")
parser.add_argument("--host")
parser.add_argument("--port", type=int)
parser.add_argument("--fail", action="store_true")
arguments = parser.parse_args()

if arguments.fail:
    print("fatal: bad config", flush=True)
    sys.exit(3)

json.load(open(arguments.config))
print("fake litellm up", flush=True)


class Health(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200 if self.path == "/health/liveliness" else 404)
        self.end_headers()

    def log_message(self, *args) -> None:
        pass


HTTPServer((arguments.host, arguments.port), Health).serve_forever()
