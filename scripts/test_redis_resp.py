"""Quick Redis connectivity test."""
import socket
import os
import time
from dotenv import load_dotenv
load_dotenv()

host = os.environ['REDIS_LINK'].rsplit(':', 1)[0]
port = int(os.environ['REDIS_LINK'].rsplit(':', 1)[1])
pw = os.environ.get('REDIS_PASSWORD', '')

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(5)
sock.connect((host, port))
print("TCP connected")

# Basic inline PING
sock.send(b"PING\r\n")
sock.settimeout(3)
try:
    resp = sock.recv(1024)
    print(f"PING: {resp!r}")
except Exception as e:
    print(f"PING timeout: {e}")
    sock.close()
    exit(1)

# Inline 1-arg AUTH (what worked before)
sock.send(f"AUTH {pw}\r\n".encode())
time.sleep(0.5)
try:
    resp = sock.recv(1024)
    print(f"AUTH: {resp!r}")
except Exception as e:
    print(f"AUTH timeout: {e}")
    sock.close()
    exit(1)

# Now try RESP PING after successful inline auth
sock.send(b"*1\r\n$4\r\nPING\r\n")
time.sleep(0.5)
try:
    resp = sock.recv(1024)
    print(f"RESP PING after auth: {resp!r}")
except Exception as e:
    print(f"RESP PING timeout: {e}")

# Try RESP SET/GET
sock.send(b"*3\r\n$3\r\nSET\r\n$9\r\ntest_key1\r\n$11\r\ntest_value1\r\n")
time.sleep(0.5)
try:
    resp = sock.recv(1024)
    print(f"RESP SET: {resp!r}")
except Exception as e:
    print(f"RESP SET timeout: {e}")

sock.close()
