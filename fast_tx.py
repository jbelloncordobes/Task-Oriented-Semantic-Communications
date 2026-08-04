import os
import time
import socket
import struct
import argparse
from datetime import datetime

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
        if role == "sender":
            send_framed(sock, "HELLO", b"READY")
            name, payload = recv_framed(sock)
            if name != "HELLO" or payload != b"READY":
                raise ConnectionError(f"Unexpected handshake reply: {name}, {payload}")
    finally:
        sock.settimeout(None)

def server_side(receiver_ip: str, port: int, files_to_send: list, log_file: str = None):
    file_data_list = []
    total_bytes = 0

    for filename in files_to_send:
        if not os.path.exists(filename):
            print(f"[!] Warning: '{filename}' not found. Creating placeholder file for network test...")
            dummy_data = b"0" * 1024 * 500  # 500 KB dummy payload
            with open(filename, "wb") as f:
                f.write(dummy_data)

        with open(filename, "rb") as f:
            data = f.read()
            file_data_list.append((filename, data))
            total_bytes += len(data)

    print(f"[Fast TX] Pre-loaded {len(files_to_send)} file(s) ({total_bytes / 1024:.2f} KB total)")

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    
    transmission_successful = False
    transmission_duration = 0.0

    try:
        client.connect((receiver_ip, port))
        handshake(client, "sender")
        
        tx_start = time.time()

        for filename, data in file_data_list:
            send_framed(client, filename, data)

        print("[Fast TX] Payloads sent, awaiting ACK...")
        recv = client.recv(1024)
        tx_end = time.time()
        
        transmission_successful = b"{FINISHED_TRANSMISSION}" in recv
        transmission_duration = tx_end - tx_start

        if transmission_successful:
            print(f"[✓] Fast TX Complete. ACK verified in {transmission_duration:.6f}s")
        else:
            print("[!] Received invalid ACK or connection dropped.")

    except Exception as e:
        print(f"[!] Socket Transmission failed: {e}")
    finally:
        client.close()

    if not log_file:
        current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = f"experiment_{current_time_str}.txt"

    try:
        with open(log_file, "w") as f:
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Files Sent: {', '.join(files_to_send)}\n")
            f.write(f"Total Transmitted: {total_bytes / 1024:.2f} KB\n")
            f.write(f"Transmission Time: {transmission_duration:.6f} seconds\n")
            f.write(f"Status: {'SUCCESS' if transmission_successful else 'FAILED'}\n")
        print(f"[Logger] Log saved to {log_file}")
    except Exception as e:
        print(f"[Logger] Error writing experiment log: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast TX Transmitter Script")
    parser.add_argument("--ip", type=str, required=True, help="Target Receiver IP")
    parser.add_argument("--port", "-p", type=int, required=True, help="Target Port")
    parser.add_argument("--files", "-f", nargs="+", required=True, help="List of files to transmit")
    parser.add_argument("--log-file", "--log", type=str, default=None, help="Name of log file to save")
    args = parser.parse_args()

    server_side(args.ip, args.port, args.files, log_file=args.log_file)
