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
import socket
import re
import time
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import struct

# Config
ECC_OVERHEAD_BYTES = 50 
# TARGET_CLASSES = [0, 4] # Person, plane
TARGET_CLASSES = [0, 164, 43, 26] # Person, table, bottle, backpack
SECRET_KEY = b"PUT_KEY_HERE"
IMAGE_NAME = ""

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

def encrypt_payload(data_bytes, key):
    """Encrypts bytes using AES-GCM. Returns (nonce + ciphertext)."""
    aesgcm = AESGCM(key)
    nonce = os.urandom(12) # GCM standard nonce size
    ciphertext = aesgcm.encrypt(nonce, data_bytes, None)
    return nonce + ciphertext # Concatenate so we only send one blob

# --- ECC HELPERS ---

def apply_ecc(data_bytes, ecc_symbols):
    rs = RSCodec(ecc_symbols)
    return bytearray(rs.encode(data_bytes))

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
    new_size = (32 * (og_size[0]//32), 32 * (og_size[1]//32))
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

def server_side():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("-------- Creating models --------")
    start = time.time()
    seg_model = create_segmentation_model("seg_model_efficientnet_deeplab.pth", device)
    img_model = create_reconstruction_model("seg_to_img_model_efficientnet.pth", device)
    autoencoder = create_autoencoder_model("residual_ae.pth", device)
    print("-------- All models created --------")
    end = time.time()
    print(f"Models loaded in {end-start} seconds")
    print("Processing image...")
    start = time.time()

    image_tensor, og_size = process_image(IMAGE_NAME)
    
    # 1. SEMANTIC SEGMENTATION
    print("Extracting semantic label...")
    input_tensor = image_tensor.unsqueeze(0).to(device) 

    # 1. SEMANTIC MASK + ENCRYPTION
    print("[Server] Processing Semantic Mask...")
    with torch.no_grad():
        output = seg_model(input_tensor)
        mask_pred = torch.argmax(output, dim=1).squeeze(0).cpu() 
        # .byte()

    # Step A: LZMA Compression
    mask_tensor = mask_pred.unsqueeze(0).float()
    mask_tensor = TF.resize(mask_tensor, og_size[::-1], interpolation=TF.InterpolationMode.NEAREST)
    mask_bytes = mask_tensor.squeeze(0).byte()
    mask_pil = TF.to_pil_image(mask_bytes)
    mask_pil.save("base_image_mask.png")

    mask_numpy = mask_bytes.cpu().numpy()
    compressed_mask_bytes = lzma.compress(mask_numpy, preset=9)
    with open("test_lzma.xz", "wb") as f:
        f.write(compressed_mask_bytes)

    # Step B: AES-GCM Encryption (New!)
    encrypted_mask = encrypt_payload(compressed_mask_bytes, SECRET_KEY)
    with open("test_aesgcm.xaz", "wb") as f:
        f.write(encrypted_mask)
    
    # Step C: RS Error Correction
    protected_mask = apply_ecc(encrypted_mask, ECC_OVERHEAD_BYTES)
    
    mask_filename = "mask_payload.rsxaz"
    with open(mask_filename, "wb") as f: f.write(protected_mask)
    mask_size_kb = os.path.getsize(mask_filename) / 1024

    # 2. RESIDUAL + ECC (Not encrypted)
    print("[Server] Processing Latent Residual...")
    # Get base image for residual calc
    mask_for_recon = mask_pred.clone().long()
    mask_for_recon[mask_for_recon == 255] = 0 
    mask_input = F.one_hot(mask_for_recon, 182).permute(2,0,1).float().unsqueeze(0).to(device)
        
    with torch.no_grad():
        base_img = img_model(mask_input)
    
    # Encode residual
    base_img_tensor = base_img.squeeze(0).cpu() 
    residual_tensor = image_tensor - base_img_tensor
    residual_input = residual_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        encoded_z = autoencoder.encoder(residual_input)

    roi_boolean = torch.isin(mask_pred.to(device), torch.tensor(TARGET_CLASSES, device=device))
    roi_mask = roi_boolean.unsqueeze(0).unsqueeze(0).float()
    
    latent_h, latent_w = encoded_z.shape[2], encoded_z.shape[3]
    strict_latent_mask = TF.resize(roi_mask, [latent_h, latent_w], interpolation=TF.InterpolationMode.NEAREST)

    # This provides the spatial context (padding) the decoder needs to prevent black patches
    halo_latent_mask = F.max_pool2d(strict_latent_mask, kernel_size=3, stride=1, padding=1)

    # Apply the Halo Mask
    encoded_z = encoded_z * halo_latent_mask
    encoded_z_fp16 = encoded_z.half()

    z_numpy = encoded_z_fp16.cpu().numpy()
    compressed_residual_bytes = lzma.compress(z_numpy.tobytes(), preset=9)
    
    # Apply ECC to Residual
    protected_residual_bytes = apply_ecc(compressed_residual_bytes, ECC_OVERHEAD_BYTES)
    
    payload_filename = "latent_payload.rsxz"
    with open(payload_filename, "wb") as f: f.write(protected_residual_bytes)
    file_size_kb = os.path.getsize("latent_payload.rsxz") / 1024
    
    print(f"Mask Payload Size:     {mask_size_kb:.2f} KB")
    print(f"ROI Latent Size:       {file_size_kb:.2f} KB")
    print(f"Total Transmission:    {mask_size_kb + file_size_kb:.2f} KB")

    end = time.time()
    print(f"Image processed and payload prepared in {end-start} seconds")

    print("Starting data transmission")
    
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    # client.connect(("localhost", 9999))
    client.connect(("", 9999)) # Put here the IP of the server used to replicate experiments

    handshake(client, "sender")  
    start = time.time()

    ack = False
    
    print("Sending data")
    send_framed(client, mask_filename, protected_mask)
    send_framed(client, payload_filename, protected_residual_bytes) 

    print("Data sent, waiting for ACK...")

    recv = client.recv(1024)
    
    transmission_successful = False
    if b"{FINISHED_TRANSMISSION}" in recv:
        end = time.time()
        print("[Transmitter] ACK verified! Transmission successful.")
        transmission_successful = True
    else:
        print("[Transmitter] Received weird response or disconnected.")

    client.close()

    transmission_duration = end - start
    print(f"Data transmission finished in {transmission_duration:.4f} seconds")

    # --- WRITE EXPERIMENT TIME WITH DATETIME FILENAME ---
    from datetime import datetime
    
    # Generate a safe, readable filename based on current date & time
    # Example: experiment_2026-07-14_12-53-10.txt
    current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"experiment_{current_time_str}.txt"
    
    try:
        with open(log_filename, "w") as f:
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Image Used: {IMAGE_NAME}\n")
            f.write(f"Mask Size: {mask_size_kb:.2f} KB\n")
            f.write(f"ROI Latent Size: {file_size_kb:.2f} KB\n")
            f.write(f"Total Transmitted: {mask_size_kb + file_size_kb:.2f} KB\n")
            f.write(f"Transmission Time: {transmission_duration:.6f} seconds\n")
            f.write(f"Status: {'SUCCESS' if transmission_successful else 'FAILED'}\n")
        print(f"[Logger] Successfully saved transmission logs to {log_filename}")
    except Exception as e:
        print(f"[Logger] Error writing experiment file: {e}")


if __name__ == "__main__":
    server_side()
