# Experimental Replication Guide: Automated Semantic Communication Testbed

This repository contains the software and automation scripts used to measure real-world performance metrics (transmission delay and RSSI) of the proposed Semantic Communication (SemCom) system over a physical Wi-Fi network.

To decouple neural network inference latency from wireless network transmission performance, the experimental workflow is split into two distinct phases:

1. Payload Generation Phase (Heavy Inference): Run the full model pipelines (`transmitter_ROI.py` / `transmitter_mask_only.py `and their respective receivers) at least once to generate the compressed, encrypted, and error-protected binary payload files (`mask_payload.rsxaz` and `latent_payload.rsxz`).

2. Fast Evaluation Phase (Decoupled Benchmarking): Use lightweight socket drivers (`fast_tx.py` and `fast_rx.py`) driven by an automated master script (`automate.py`) to measure exact network transit time, throughput, and RSSI across multiple runs and channel noise conditions.

## 1. Environment & Dependencies Setup

### Prerequisites
Ensure both local (Server/Receiver) and remote (Target/Transmitter) systems have Python 3.8+ installed along with the required libraries:
- torchvision.
- segmentation-models-pytorch.
- pillow.
- reedsolo.
- cryptography.
- paramiko.
- numpy.

### Required Model Weights

For the full payload generation phase, place the following pre-trained PyTorch weight files in your working directory:  
- `seg_model_efficientnet_deeplab.pth`.
- `seg_to_img_model_efficientnet.pth`.
- `residual_ae.pth`.

## 2. Phase 1: One-Time Payload Generation
Before running fast network benchmarks, you must generate the binary semantic payloads for your target test image.

### A: Configure Script Variables
In `transmitter_ROI.py` or `transmitter_mask_only.py`, set target image path and encryption key, the same must be set in the corresponding receiver script:
```
SECRET_KEY = b"YOUR_32_BYTE_KEY"  # Must match the receiver key
IMAGE_NAME = "IMAGE.PNG"     # Target benchmark image
```
### B: Run Full Model Pipeline
1. Start the Receiver: Run `receiver_ROI.py` (or `receiver_mask_only.py`) on the receiving host.
2. Start the Transmitter: Run `transmitter_ROI.py` (or `transmitter_mask_only.py`) on the transmitting host.

During execution, the pipeline will perform semantic segmentation, base image generation, residual autoencoding, LZMA compression, AES-GCM encryption, and Reed-Solomon ECC protection.

### C: Verify Generated Payloads

The following files will be output to disk:  
- `mask_payload.rsxaz`: LZMA-compressed, AES-GCM encrypted, Reed-Solomon protected semantic mask.
- `latent_payload.rsxz`: LZMA-compressed, Reed-Solomon protected ROI latent residual.

Copy `mask_payload.rsxaz`, `latent_payload.rsxz`, `fast_tx.py`, and the raw `IMAGE_FILENAME` to the target remote directory (e.g., target single-board computer/transmitter device).

## 3. Configuring and Executing `automate.py`
Once payload files are generated, use the decoupled `fast_tx.py` and `fast_rx.py` scripts via `automate.py` to benchmark raw transmission performance over TCP.

The `automate.py` script orchestrates the local receiver and remote SSH transmitter, gathers Wi-Fi RSSI stats from the Access Point/Router, and logs transmission metrics.

### A: Configure Target & Router Details
Open `automate.py` and update the connection details:
```
# Network Nodes
ROUTER_IP    = "192.168.1.1"       # Access Point IP
ROUTER_USER  = "root"
ROUTER_PASS  = "router_password"

ARDUINO_IP   = "192.168.1.50"      # Remote Transmitter IP
ARDUINO_USER = "pi"
ARDUINO_PASS = "pi_password"
REMOTE_DIR   = "/home/pi/semcom/"  # Remote directory containing fast_tx.py & payloads

ARDUINO_MAC  = "aa:bb:cc:dd:ee:ff" # MAC address of remote transmitter
LOCAL_RX_IP  = "192.168.1.100"     # IP of the server

# Experiment Parameters
NUM_RUNS     = 10                  # Number of iterations per mode
USE_NOISE    = False               # Set to True to enable background iperf3 interference
NOISE        = "10M"               # iperf3 bandwidth rate (e.g., 10 Mbit/s)
IMAGE_FILENAME = "image.png"
```

### B: Run Automated Experiments
Run the master script on your local server:
```
python3 automate.py
```
What `automate.py` Does Automatically:
1. Opens SSH sessions to the Router and Remote Target.
2. Verifies remote files (`fast_tx.py` and target image) exist in `REMOTE_DIR`.
3. Optionally starts an iperf3 UDP noise generator to simulate channel congestion (`USE_NOISE = True`).
4. Iterates through all test modes (baseline, roi, mask_only):
   - Cleans up local TCP ports.
   - Launches `fast_rx.py` locally on the required port.
   - Commands the remote target via SSH to execute `fast_tx.py`.
   - Samples router signal strength (`iw dev <iface> station get <mac>`) continuously during transmission.
   - Verifies transmission ACKs (`{FINISHED_TRANSMISSION}`) and records exact network duration.
5. Cleans up background processes and outputs a summary table.

## 5. Experiment Output & Results Interpretation

Upon completing execution, `automate.py` outputs experiment log files and prints a summary table to stdout. Example:
```
=================== FINAL RESULTS ===================
Average BASELINE   | Delay: 0.1245s   | RSSI: -54.20 dBm
Average ROI        | Delay: 0.0312s   | RSSI: -54.15 dBm
Average MASK_ONLY  | Delay: 0.0089s   | RSSI: -54.30 dBm
```
Individual run logs (`experiment_<mode>_<run_num>.txt`) generated on the target device contain timestamped records. Example:

```
Timestamp: 2026-08-04T13:45:00.123456
Files Sent: mask_payload.rsxaz, latent_payload.rsxz
Total Transmitted: 18.42 KB
Transmission Time: 0.031200 seconds
Status: SUCCESS
```

## Acknowledgements & Open-Source Software

This project builds upon several open-source libraries and pretrained models:
* [segmentation-models-pytorch](https://github.com/CSAILVision/semantic-segmentation-pytorch) (BSD 3-Clause License, Copyright (c) 2019, MIT CSAIL Computer Vision)
* [PyTorch](https://pytorch.org/)

Open Datasets & Benchmarks:
- [COCO-Stuff Dataset](https://github.com/nightrome/cocostuff) for segmentation and generative reconstruction training.
- [Kodak Lossless True Color Image Suite](http://r0k.us/graphics/kodak/) for evaluation metrics.
