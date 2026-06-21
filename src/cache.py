import hashlib
import json
import os
import socket
import threading
import time

from dotenv import load_dotenv

load_dotenv()

MAX_RETRIES = 5
RETRY_DELAY = 3  # seconds


class _InlineRedis:
    """Minimal Redis client for our cache-only use case (GET/SET with string values).

    Redis Cloud quirks for this instance (Redis 8.4, RESP3-only):
    1. RESP binary-format AUTH hangs — inline AUTH works
    2. Must send PING before AUTH or AUTH hangs
    3. After sending a RESP-format command, inline commands stop working
       on the same connection (mode sticks)

    Strategy: use inline for PING/AUTH during connection setup, then
    RESP format for all data commands (GET/SET). Reconnect before each
    data command to avoid hitting the per-connection command limit.
    """

    def __init__(self, host: str, port: int, password: str | None):
        self._host = host
        self._port = port
        self._password = password
        self._sock: socket.socket | None = None
        self._buf: bytes = b""

    def _connect(self) -> None:
        self.close()
        self._buf = b""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(10)
        self._sock.connect((self._host, self._port))

        # Redis 8.4 quirk: must send PING before AUTH or AUTH hangs
        self._send_inline("PING")
        self._read_line()  # -NOAUTH or +PONG, either is fine

        if self._password:
            self._send_inline(f"AUTH {self._password}")
            resp = self._read_line()
            if not resp.startswith("+OK"):
                raise ConnectionError(f"AUTH failed: {resp}")

    def _fresh_connect(self) -> None:
        """Reconnect for each data command (server limits commands per connection)."""
        self._connect()

    def _send_inline(self, cmd: str) -> None:
        self._sock.sendall(f"{cmd}\r\n".encode())

    def _read_line(self) -> str:
        """Read one RESP line (ending in \\r\\n), keeping leftovers in buffer."""
        while b"\r\n" not in self._buf:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise ConnectionError("Connection closed")
            self._buf += chunk
        idx = self._buf.index(b"\r\n")
        line = self._buf[:idx]
        self._buf = self._buf[idx + 2:]
        return line.decode(errors="replace")

    def _read_bulk(self) -> str | None:
        """Read a RESP bulk string response (e.g. $5\\r\\nhello\\r\\n)."""
        line = self._read_line()
        if line.startswith("$-1") or line.startswith("_"):
            return None  # nil
        if line.startswith("$"):
            length = int(line[1:])
            # Read exactly length + 2 bytes (data + \r\n) from buffer
            needed = length + 2
            while len(self._buf) < needed:
                chunk = self._sock.recv(4096)
                if not chunk:
                    raise ConnectionError("Connection closed during bulk read")
                self._buf += chunk
            data = self._buf[:length]
            self._buf = self._buf[needed:]
            return data.decode(errors="replace")
        # Simple string response (+OK, etc.)
        if line.startswith("+"):
            return line[1:]
        if line.startswith("-"):
            raise ConnectionError(f"Redis error: {line}")
        return line

    def close(self) -> None:
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def ping(self) -> bool:
        self._fresh_connect()
        self._send_inline("PING")
        resp = self._read_line()
        return "PONG" in resp

    def _send_resp(self, *args: str) -> None:
        """Send a command in RESP format."""
        parts = [f"*{len(args)}\r\n"]
        for arg in args:
            encoded = arg.encode()
            parts.append(f"${len(encoded)}\r\n")
            parts.append(arg + "\r\n")
        self._sock.sendall("".join(parts).encode())

    def get(self, key: str) -> str | None:
        self._fresh_connect()
        self._send_resp("GET", key)
        return self._read_bulk()

    def set(self, key: str, value: str) -> None:
        self._fresh_connect()
        self._send_resp("SET", key, value)
        resp = self._read_line()
        if not resp.startswith("+OK"):
            raise ConnectionError(f"SET failed: {resp}")


_client: _InlineRedis | None = None
_lock = threading.Lock()


def get_redis() -> _InlineRedis:
    """Get or create the singleton Redis client."""
    global _client
    if _client is None:
        redis_link = os.environ["REDIS_LINK"]
        host, port_str = redis_link.rsplit(":", 1)
        port = int(port_str)
        password = os.environ.get("REDIS_PASSWORD")
        _client = _InlineRedis(host, port, password)
    return _client


def _reset_client() -> None:
    """Force reconnect on next call."""
    global _client
    if _client is not None:
        _client.close()
    _client = None


def make_key(*parts: str) -> str:
    """Create a SHA256 cache key from concatenated parts."""
    raw = "||".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()


def cache_get(key: str) -> dict | None:
    """Retrieve a cached value by key. Returns None on miss. Retries on connection errors."""
    for attempt in range(MAX_RETRIES):
        try:
            with _lock:
                r = get_redis()
                val = r.get(key)
            if val is None:
                return None
            return json.loads(val)
        except (ConnectionError, TimeoutError, OSError) as e:
            with _lock:
                _reset_client()
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise


def cache_set(key: str, value: dict) -> None:
    """Store a value in the cache as JSON. Retries on connection errors."""
    for attempt in range(MAX_RETRIES):
        try:
            with _lock:
                r = get_redis()
                r.set(key, json.dumps(value))
            return
        except (ConnectionError, TimeoutError, OSError) as e:
            with _lock:
                _reset_client()
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise
