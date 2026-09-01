"""Lightweight Model Context Protocol (MCP) Server for Project NOIR."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional
from noir.core.logging import get_logger
from noir.mcp.resources import MCPResourceManager
from noir.mcp.tools import MCPToolRegistry

logger = get_logger("mcp.server")


class MCPRequestHandler(BaseHTTPRequestHandler):
    """Handles JSON-RPC requests conforming to MCP specifications."""

    tool_registry: MCPToolRegistry
    resource_manager: MCPResourceManager

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")

        try:
            req = json.loads(body)
            response = self._handle_rpc(req)
            self._send_json_response(200, response)
        except Exception as e:
            logger.error("MCP Server handling error: %s", e)
            self._send_json_response(500, {"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e)}, "id": None})

    def _handle_rpc(self, req: Dict[str, Any]) -> Dict[str, Any]:
        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        if method == "tools/list":
            tools = self.tool_registry.list_tools()
            return {"jsonrpc": "2.0", "result": {"tools": tools}, "id": req_id}

        elif method == "tools/call":
            name = params.get("name")
            args = params.get("arguments", {})
            res = self.tool_registry.execute(name, args)
            return {"jsonrpc": "2.0", "result": res, "id": req_id}

        elif method == "resources/list":
            resources = self.resource_manager.list_resources()
            return {"jsonrpc": "2.0", "result": {"resources": resources}, "id": req_id}

        elif method == "resources/read":
            uri = params.get("uri")
            content = self.resource_manager.read_resource(uri)
            return {"jsonrpc": "2.0", "result": {"contents": [{"uri": uri, "text": content}]}, "id": req_id}

        else:
            return {"jsonrpc": "2.0", "error": {"code": -32601, "message": f"Method '{method}' not found"}, "id": req_id}

    def _send_json_response(self, status: int, data: Dict[str, Any]) -> None:
        response_bytes = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress standard HTTP server console noise
        pass


class MCPServer:
    """Embeddable MCP Server running as a background daemon."""

    def __init__(self, engine: Any, host: str = "127.0.0.1", port: int = 8765):
        self.engine = engine
        self.host = host
        self.port = port
        self.tool_registry = MCPToolRegistry(engine)
        self.resource_manager = MCPResourceManager(engine)

        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the MCP HTTP server daemon."""
        if self._thread and self._thread.is_alive():
            return

        handler_cls = type(
            "BoundMCPRequestHandler",
            (MCPRequestHandler,),
            {
                "tool_registry": self.tool_registry,
                "resource_manager": self.resource_manager,
            },
        )

        try:
            self._server = HTTPServer((self.host, self.port), handler_cls)
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True, name="NoirMCPServer")
            self._thread.start()
            logger.info("MCP Server started at http://%s:%d", self.host, self.port)
        except Exception as e:
            logger.warning("Could not start MCP Server on port %d: %s", self.port, e)

    def stop(self) -> None:
        """Stop the MCP server."""
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            logger.info("MCP Server stopped.")
