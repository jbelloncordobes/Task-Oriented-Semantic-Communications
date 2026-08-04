import argparse
import socket
import struct
import time

# --- SOCKET FRAMING HELPERS ---
def send_framed(sock, name: str, payload: bytes):
    name_bytes = name.encode()
    header = struct.pack("!I", len(name_bytes)) + name_bytes + struct.pack("!Q", len(payload))
    sock.sendall(header + payload)

def recv_exact(sock, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(65536, n - len(buf)))
        if not chunk:
            raise ConnectionError(f"Socket closed with {n - len(buf)} bytes still expected")
        buf.extend(chunk)
    return bytes(buf)

def recv_framed(sock):
    name_len = struct.unpack("!I", recv_exact(sock, 4))[0]
    name = recv_exact(sock, name_len).decode()
    payload_len = struct.unpack("!Q", recv_exact(sock, 8))[0]
    payload = recv_exact(sock, payload_len)
    return name, payload

def handshake(sock, role: str, timeout: float = 30.0):
    sock.settimeout(timeout)
    try:
        name, payload = recv_framed(sock)
        if name != "HELLO" or payload != b"READY":
            raise ConnectionError(f"Unexpected handshake message: {name}, {payload}")
        send_framed(sock, "HELLO", b"READY")
    finally:
        sock.settimeout(None)

def client_side(port: int, expected_files_count: int, log_file: str = None):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", port))
    server.listen(1)

    print(f"[Fast RX] Listening on port {port}...")
    server.settimeout(35.0)
    try:
        client, addr = server.accept()
        client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        handshake(client, "receiver")

        print("[Fast RX] Receiving Data...")
        rx_start = time.time()

        for _ in range(expected_files_count):
            filename, payload_bytes = recv_framed(client)
            with open(filename, "wb") as f:
                f.write(payload_bytes)
            print(f"[Fast RX] Successfully written '{filename}' ({len(payload_bytes)} bytes)")

        client.sendall(b"{FINISHED_TRANSMISSION}")
        rx_end = time.time()
        print(f"[Fast RX] Finished reception in {rx_end - rx_start:.6f} seconds")

        client.close()
    except Exception as e:
        print(f"[Fast RX] Reception stopped or timed out: {e}")
    finally:
        server.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast RX Receiver Script")
    parser.add_argument("--port", "-p", type=int, required=True, help="Port to listen on")
    parser.add_argument("--expected-files", "-n", type=int, default=1, help="Number of files expected per session")
    parser.add_argument("--log-file", "--log", type=str, default=None, help="Optional log file path")
    args = parser.parse_args()

    client_side(args.port, args.expected_files, log_file=args.log_file)
