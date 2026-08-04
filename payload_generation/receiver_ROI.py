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
import random
import socket
import re
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import struct

# Config
ECC_OVERHEAD_BYTES = 50 
# TARGET_CLASSES = [0, 4] # Person, plane
TARGET_CLASSES = [0, 164, 43, 26] # Person, table, bottle, backpack
SECRET_KEY = b"PUT_KEY_HERE"
IMAGE_NAME = ""
MODULO = 32
LATENT_DOWNSAMPLE = 16


def handshake(sock, role: str, timeout: float = 30.0):
    """
    Simple mutual readiness check.
    role: 'sender' sends first then waits for reply.
          'receiver' waits for a message first then replies.
    Raises ConnectionError / TimeoutError if the handshake fails.
    """
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
        sock.settimeout(None)  # back to blocking mode for the rest of the transfer
    print(f"[{role}] Handshake complete — both sides ready.")

def send_framed(sock, name: str, payload: bytes):
    """Send one named chunk: [4-byte name_len][name][8-byte payload_len][payload]"""
    name_bytes = name.encode()
    header = struct.pack("!I", len(name_bytes)) + name_bytes + struct.pack("!Q", len(payload))
    sock.sendall(header + payload)

def recv_exact(sock, n: int) -> bytes:
    """Read exactly n bytes or raise if the connection drops early."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(65536, n - len(buf)))
        if not chunk:
            raise ConnectionError(f"Socket closed with {n - len(buf)} bytes still expected")
        buf.extend(chunk)
    return bytes(buf)

def recv_framed(sock):
    """Read one named chunk sent with send_framed. Returns (name, payload)."""
    name_len = struct.unpack("!I", recv_exact(sock, 4))[0]
    name = recv_exact(sock, name_len).decode()
    payload_len = struct.unpack("!Q", recv_exact(sock, 8))[0]
    payload = recv_exact(sock, payload_len)
    return name, payload

# --- ENCRYPTION HELPERS ---    
def decrypt_payload(encrypted_blob, key):
    """Splits nonce and ciphertext, then decrypts."""
    aesgcm = AESGCM(key)
    
    # Cast the incoming blob to bytes to ensure slicing yields bytes objects
    blob_bytes = bytes(encrypted_blob)
    nonce = blob_bytes[:12]
    ciphertext = blob_bytes[12:]
    
    try:
        return aesgcm.decrypt(nonce, ciphertext, None)
    except Exception as e:
        print(f"CRITICAL: Decryption failed (Integrity check failed or wrong key)! {e}")
        return None

# --- ECC HELPERS ---
def remove_ecc(encoded_bytes, ecc_symbols):
    rs = RSCodec(ecc_symbols)
    try:
        decoded_data, _, _ = rs.decode(encoded_bytes)
        return bytes(decoded_data)  # Changed from bytearray(decoded_data)
    except ReedSolomonError:
        print("CRITICAL NETWORK FAILURE: Too many dropped bytes to repair!")
        return None

# --- MODEL DEFINITIONS ---

class ResidualAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 16, 3, stride=2, padding=1), # 16-Channel Bottleneck
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Tanh()  
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

class SegToImageModel(nn.Module):
    def __init__(self, num_classes=182):
        super().__init__()
        self.model = smp.Unet(encoder_name="efficientnet-b3", encoder_weights=None, 
                              in_channels=num_classes, classes=3, activation="sigmoid")
    def forward(self, x): return self.model(x)

def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    og_size = image.size
    new_size = (MODULO * (og_size[0]//MODULO), MODULO * (og_size[1]//MODULO))
    return TF.to_tensor(TF.resize(image, new_size)), og_size

def create_segmentation_model(weights, device):
    model = smp.DeepLabV3Plus(encoder_name="efficientnet-b3", encoder_weights="imagenet", classes=182, activation=None)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device)
    model.eval()
    return model

def create_reconstruction_model(weights, device):
    model = SegToImageModel(182).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device)
    model.eval()
    return model

def create_autoencoder_model(weights, device):
    model = ResidualAE().to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    return model

# --- PIPELINE CORE ---

def client_side():
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("-------- Loading Client Models --------")
    img_model = create_reconstruction_model("seg_to_img_model_efficientnet.pth", device)
    autoencoder = create_autoencoder_model("residual_ae.pth", device)
    print("-------- Client Models Ready --------")
    end = time.time()
    print(f"Models loaded in {end-start} seconds")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Enable address reuse to avoid 'Address already in use' errors during rapid restarts
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", 9999))
    server.listen()

    client, addr = server.accept()
    client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    handshake(client, "receiver")  

    start = time.time()

    print("\n[Client] Receiving Data...")

    mask_filename, mask_bytes = recv_framed(client)
    residual_filename, residual_bytes = recv_framed(client)

    print("[Receiver] Full payload received! Sending ACK...")
    client.sendall(b"{FINISHED_TRANSMISSION}")

    end = time.time()
    print(f"Data received and ACK sent in {end-start} seconds")

    with open(mask_filename, "wb") as f:
        f.write(mask_bytes)

    with open(residual_filename, "wb") as f:
        f.write(residual_bytes)



    # 1. LOAD & DECODE THE RECEIVED MASK
    og_size = Image.open(IMAGE_NAME).size

    # Repair
    start = time.time()
    repaired_mask_bytes = remove_ecc(mask_bytes, ECC_OVERHEAD_BYTES)

    if repaired_mask_bytes is None: return
    
    # Step B: Decrypt
    decrypted_mask_lzma = decrypt_payload(repaired_mask_bytes, SECRET_KEY)
    if decrypted_mask_lzma is None: return # Stop if decryption fails

    # Step C: Decompress
    mask_bytes = lzma.decompress(decrypted_mask_lzma)
    print("Decryption and Integrity Check: SUCCESS")

    # Reconstruct mask tensor
    mask_array = np.frombuffer(mask_bytes, dtype=np.uint8).reshape((og_size[1], og_size[0]))
    mask_image = Image.fromarray(mask_array)
    
    maximum_size = (int(og_size[0]/MODULO), int(og_size[1]/MODULO))
    new_size = (MODULO*maximum_size[0], MODULO*maximum_size[1])
    mask_resized = TF.resize(mask_image, new_size, interpolation=Image.NEAREST)
    
    mask_tensor = torch.tensor(np.array(mask_resized), dtype=torch.long, device=device)
    mask_tensor[mask_tensor == 255] = 0
    mask_onehot = F.one_hot(mask_tensor, 182).permute(2,0,1).float().unsqueeze(0).to(device)

    with torch.no_grad():
        base_img_tensor = img_model(mask_onehot) 
    
    # 2. LOAD & DECODE THE RECEIVED RESIDUAL
    
    repaired_residual_bytes = remove_ecc(residual_bytes, ECC_OVERHEAD_BYTES)

    if repaired_residual_bytes is None: return

    decompressed_bytes = lzma.decompress(repaired_residual_bytes)
    
    latent_h = base_img_tensor.shape[2] // LATENT_DOWNSAMPLE
    latent_w = base_img_tensor.shape[3] // LATENT_DOWNSAMPLE
    
    received_data = np.frombuffer(decompressed_bytes, dtype=np.float16).reshape((1, 16, latent_h, latent_w))
    received_z = torch.tensor(received_data).float().to(device)

    with torch.no_grad():
        decoded_residual = autoencoder.decoder(received_z)

    roi_boolean = torch.isin(mask_tensor, torch.tensor(TARGET_CLASSES, device=device))
    roi_mask = roi_boolean.unsqueeze(0).unsqueeze(0).float()
    decoded_residual = decoded_residual * roi_mask

    # 3. FINAL RECONSTRUCTION & CLAMPING
    print("\nCompositing final image...")
    final_image_tensor = base_img_tensor + decoded_residual
    final_image_tensor = torch.clamp(final_image_tensor, 0.0, 1.0)
    final_image_tensor = TF.resize(final_image_tensor, og_size[::-1])

    final_img_cpu = final_image_tensor.squeeze(0).cpu()
    final_pil = TF.to_pil_image(final_img_cpu)
    
    save_path = "final_xr_reconstruction.png"
    final_pil.save(save_path)
    end = time.time()
    print(f"Success! Final image saved as {save_path} in {end-start} seconds")

if __name__ == "__main__":
    client_side()
