import socket
import ssl
import threading
import traceback
from datetime import datetime

UPSTREAM_HOST = "api.kite.trade"
UPSTREAM_PORT = 443
LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 8080
BUF = 8192


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def read_http_request(sock):
    data = b""
    sock.settimeout(5)

    log("read_http_request(): reading headers")

    while b"\r\n\r\n" not in data:
        chunk = sock.recv(BUF)
        if not chunk:
            break
        data += chunk
        log(f"read_http_request(): total={len(data)}")

    return data


def rewrite_request(req):
    lines = req.split(b"\r\n")
    out = []

    for i, line in enumerate(lines):
        if i == 0:
            parts = line.split(b" ")
            if len(parts) == 3 and parts[1].startswith(b"http://"):
                url = parts[1].decode(errors="ignore")
                path = "/" + url.split("/", 3)[-1] if "/" in url[7:] else "/"
                newline = parts[0] + b" " + path.encode() + b" " + parts[2]
                out.append(newline)
                continue

        if line.lower().startswith(b"host:"):
            out.append(b"Host: api.kite.trade")
            continue

        if line.lower().startswith(b"proxy-connection:"):
            continue

        out.append(line)

    return b"\r\n".join(out)


def handle(client, addr):
    raw = None
    tls = None

    try:
        log(f"handle(): client connected {addr}")

        req = read_http_request(client)
        if not req:
            return

        log(req.decode(errors="ignore"))

        req = rewrite_request(req)

        raw = socket.create_connection((UPSTREAM_HOST, UPSTREAM_PORT), timeout=30)

        ctx = ssl.create_default_context()
        tls = ctx.wrap_socket(raw, server_hostname=UPSTREAM_HOST)

        tls.settimeout(30)

        log("sending request upstream")
        tls.sendall(req)

        while True:
            data = tls.recv(BUF)
            if not data:
                break
            client.sendall(data)

    except Exception as e:
        log(f"ERROR: {e}")
        traceback.print_exc()

    finally:
        log("cleanup")
        for s in [client, tls, raw]:
            if s:
                try:
                    s.close()
                except:
                    pass


def main():
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((LISTEN_HOST, LISTEN_PORT))
    srv.listen(50)

    log(f"Listening on {LISTEN_HOST}:{LISTEN_PORT}")

    while True:
        client, addr = srv.accept()
        threading.Thread(target=handle, args=(client, addr), daemon=True).start()


main()
