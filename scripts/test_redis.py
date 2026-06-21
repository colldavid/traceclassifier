"""Quick Redis connectivity test."""
import socket
import os
from dotenv import load_dotenv
load_dotenv()

host = os.environ['REDIS_LINK'].rsplit(':', 1)[0]
port = int(os.environ['REDIS_LINK'].rsplit(':', 1)[1])
pw = os.environ.get('REDIS_PASSWORD', '')

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(5)
sock.connect((host, port))

# Build RESP format manually — no shell escaping issues
parts = []
parts.append(b'*2\r\n')
parts.append(b'$4\r\n')
parts.append(b'AUTH\r\n')
parts.append(f'${len(pw)}\r\n'.encode())
parts.append(f'{pw}\r\n'.encode())
cmd = b''.join(parts)
print(f'Sending RESP AUTH: {cmd[:40]}... ({len(cmd)} bytes)')
print(f'Hex: {cmd.hex()[:80]}...')
sock.send(cmd)

import time
time.sleep(1)
try:
    resp = sock.recv(1024)
    print(f'RESP AUTH response: {resp!r}')
except Exception as e:
    print(f'RESP AUTH timeout: {e}')
    # Try inline on same connection
    print('Trying inline AUTH on same connection...')
    sock.send(f'AUTH {pw}\r\n'.encode())
    time.sleep(0.5)
    try:
        resp = sock.recv(1024)
        print(f'Inline AUTH response: {resp!r}')
    except Exception as e2:
        print(f'Inline AUTH also failed: {e2}')

sock.close()
