import os
import numpy as np
from PIL import Image
import lzma
from reedsolo import RSCodec, ReedSolomonError
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
import socket
import time
import struct
from datetime import datetime
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# --- CONFIGURATION ---
ECC_OVERHEAD_BYTES = 50
SECRET_KEY = b"PUT_KEY_HERE"
IMAGE_NAME = ""
MODULO = 32

# --- SOCKET FRAMING HELPERS ---
def handshake(sock, role: str, timeout: float = 30.0):
    sock.settimeout(timeout)
    try:
        if role == "sender":
            send_framed(sock, "HELLO", b"READY")
            name, payload = recv_framed(sock)
            if name != "HELLO" or payload != b"READY":
                raise ConnectionError(f"Unexpected handshake reply: {name}, {payload}")
        elif role == "receiver":
            name, payload = recv_framed(sock)
            if name != "HELLO" or payload != b"READY":
                raise ConnectionError(f"Unexpected handshake message: {name}, {payload}")
            send_framed(sock, "HELLO", b"READY")
        else:
            raise ValueError("role must be 'sender' or 'receiver'")
    finally:
        sock.settimeout(None)

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

# --- SECURITY & ECC HELPERS ---
def encrypt_payload(data_bytes, key):
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ciphertext = aesgcm.encrypt(nonce, data_bytes, None)
    return nonce + ciphertext

def apply_ecc(data_bytes, ecc_symbols):
    rs = RSCodec(ecc_symbols)
    return bytearray(rs.encode(data_bytes))

# --- MODEL INFERENCE HELPERS ---
def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    og_size = image.size
    new_size = (MODULO * (og_size[0] // MODULO), MODULO * (og_size[1] // MODULO))
    return TF.to_tensor(TF.resize(image, new_size)), og_size

def create_segmentation_model(weights, device):
    model = smp.DeepLabV3Plus(encoder_name="efficientnet-b3", encoder_weights="imagenet", classes=182, activation=None)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device)
    model.eval()
    return model

def server_side():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("-------- [Mask-Only TX] Creating Segmentation Model --------")
    start = time.time()
    seg_model = create_segmentation_model("seg_model_efficientnet_deeplab.pth", device)
    print(f"[Mask-Only TX] Model loaded in {time.time() - start:.4f} seconds")

    print("[Mask-Only TX] Processing image...")
    start = time.time()
    image_tensor, og_size = process_image(IMAGE_NAME)
    input_tensor = image_tensor.unsqueeze(0).to(device) 

    # 1. SEMANTIC SEGMENTATION MASK EXTRACTION
    print("[Mask-Only TX] Extracting Semantic Mask...")
    with torch.no_grad():
        output = seg_model(input_tensor)
        mask_pred = torch.argmax(output, dim=1).squeeze(0).cpu() 

    # Compression (LZMA)
    mask_tensor = mask_pred.unsqueeze(0).float()
    mask_tensor = TF.resize(mask_tensor, og_size[::-1], interpolation=TF.InterpolationMode.NEAREST)
    mask_bytes = mask_tensor.squeeze(0).byte()
    
    mask_numpy = mask_bytes.cpu().numpy()
    compressed_mask_bytes = lzma.compress(mask_numpy, preset=9)

    # Encryption (AES-GCM)
    encrypted_mask = encrypt_payload(compressed_mask_bytes, SECRET_KEY)
    
    # Error Correction (Reed-Solomon)
    protected_mask = apply_ecc(encrypted_mask, ECC_OVERHEAD_BYTES)
    
    mask_filename = "mask_payload.rsxaz"
    with open(mask_filename, "wb") as f: 
        f.write(protected_mask)
        
    mask_size_kb = os.path.getsize(mask_filename) / 1024
    
    print(f"Mask Payload Size:     {mask_size_kb:.2f} KB")
    print(f"Total Transmission:    {mask_size_kb:.2f} KB")

    # 2. TRANSMISSION
    print("[Mask-Only TX] Connecting to Receiver...")
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    
    # Change IP to server used to replicate experiments
    client.connect(("", 9999))

    handshake(client, "sender")  
    tx_start = time.time()

    print("[Mask-Only TX] Sending Semantic Mask Frame...")
    send_framed(client, mask_filename, protected_mask)

    print("[Mask-Only TX] Data sent, waiting for ACK...")
    recv = client.recv(1024)
    
    transmission_successful = False
    if b"{FINISHED_TRANSMISSION}" in recv:
        tx_end = time.time()
        print("[Mask-Only TX] ACK verified! Transmission successful.")
        transmission_successful = True
    else:
        tx_end = time.time()
        print("[Mask-Only TX] Received invalid response or disconnected.")

    client.close()

    transmission_duration = tx_end - tx_start
    print(f"[Mask-Only TX] Transmission finished in {transmission_duration:.6f} seconds")

    # Log generation matching automation regex: "Transmission Time:\s+([\d.]+)"
    current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"experiment_{current_time_str}.txt"
    
    try:
        with open(log_filename, "w") as f:
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Image Used: {IMAGE_NAME}\n")
            f.write(f"Mask Size: {mask_size_kb:.2f} KB\n")
            f.write(f"ROI Latent Size: 0.00 KB\n")
            f.write(f"Total Transmitted: {mask_size_kb:.2f} KB\n")
            f.write(f"Transmission Time: {transmission_duration:.6f} seconds\n")
            f.write(f"Status: {'SUCCESS' if transmission_successful else 'FAILED'}\n")
        print(f"[Logger] Successfully saved transmission log to {log_filename}")
    except Exception as e:
        print(f"[Logger] Error writing experiment log: {e}")

if __name__ == "__main__":
    server_side()
