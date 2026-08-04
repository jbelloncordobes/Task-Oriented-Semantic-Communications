import os
import numpy as np
from PIL import Image
import lzma
from reedsolo import RSCodec, ReedSolomonError
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
import time
import socket
import struct
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
def decrypt_payload(encrypted_blob, key):
    aesgcm = AESGCM(key)
    blob_bytes = bytes(encrypted_blob)
    nonce = blob_bytes[:12]
    ciphertext = blob_bytes[12:]
    try:
        return aesgcm.decrypt(nonce, ciphertext, None)
    except Exception as e:
        print(f"CRITICAL: Decryption failed! {e}")
        return None

def remove_ecc(encoded_bytes, ecc_symbols):
    rs = RSCodec(ecc_symbols)
    try:
        decoded_data, _, _ = rs.decode(encoded_bytes)
        return bytes(decoded_data)
    except ReedSolomonError:
        print("CRITICAL NETWORK FAILURE: Reed-Solomon repair failed!")
        return None

# --- MODEL DEFINITIONS ---
class SegToImageModel(nn.Module):
    def __init__(self, num_classes=182):
        super().__init__()
        self.model = smp.Unet(encoder_name="efficientnet-b3", encoder_weights=None, 
                              in_channels=num_classes, classes=3, activation="sigmoid")
    def forward(self, x): return self.model(x)

def create_reconstruction_model(weights, device):
    model = SegToImageModel(182).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device)
    model.eval()
    return model

def client_side():
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("-------- [Mask-Only RX] Loading Reconstruction Model --------")
    img_model = create_reconstruction_model("seg_to_img_model_efficientnet.pth", device)
    print(f"[Mask-Only RX] Model loaded in {time.time()-start:.4f} seconds")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", 9999))
    server.listen()

    print("[Mask-Only RX] Waiting for connection...")
    client, addr = server.accept()
    client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    handshake(client, "receiver")  

    print("\n[Mask-Only RX] Receiving Data...")
    rx_start = time.time()

    mask_filename, mask_bytes = recv_framed(client)

    print("[Mask-Only RX] Semantic mask frame received! Sending ACK...")
    client.sendall(b"{FINISHED_TRANSMISSION}")
    rx_end = time.time()
    print(f"[Mask-Only RX] Data received and ACK sent in {rx_end - rx_start:.6f} seconds")

    with open(mask_filename, "wb") as f:
        f.write(mask_bytes)

    # 1. DECODE & REPAIR MASK
    og_size = Image.open(IMAGE_NAME).size

    repaired_mask_bytes = remove_ecc(mask_bytes, ECC_OVERHEAD_BYTES)
    if repaired_mask_bytes is None: 
        return
    
    decrypted_mask_lzma = decrypt_payload(repaired_mask_bytes, SECRET_KEY)
    if decrypted_mask_lzma is None: 
        return 

    decompressed_mask_bytes = lzma.decompress(decrypted_mask_lzma)
    print("[Mask-Only RX] Decryption & LZMA Decompression: SUCCESS")

    # 2. SYNTHESIZE MASK & BASE IMAGE
    mask_array = np.frombuffer(decompressed_mask_bytes, dtype=np.uint8).reshape((og_size[1], og_size[0]))
    mask_image = Image.fromarray(mask_array)
    
    maximum_size = (int(og_size[0] / MODULO), int(og_size[1] / MODULO))
    new_size = (MODULO * maximum_size[0], MODULO * maximum_size[1])
    mask_resized = TF.resize(mask_image, new_size, interpolation=Image.NEAREST)
    
    mask_tensor = torch.tensor(np.array(mask_resized), dtype=torch.long, device=device)
    mask_tensor[mask_tensor == 255] = 0
    mask_onehot = F.one_hot(mask_tensor, 182).permute(2, 0, 1).float().unsqueeze(0).to(device)

    print("[Mask-Only RX] Running generative U-Net reconstruction...")
    with torch.no_grad():
        final_image_tensor = img_model(mask_onehot) 

    # Save final mask-only output
    final_image_tensor = torch.clamp(final_image_tensor, 0.0, 1.0)
    final_image_tensor = TF.resize(final_image_tensor, og_size[::-1])

    final_img_cpu = final_image_tensor.squeeze(0).cpu()
    final_pil = TF.to_pil_image(final_img_cpu)
    
    save_path = "final_mask_only_reconstruction.png"
    final_pil.save(save_path)
    print(f"Success! Final Mask-Only reconstruction saved as '{save_path}'")

if __name__ == "__main__":
    client_side()
