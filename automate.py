import os
import re
import time
import subprocess
import paramiko

# =====================================================================
# --- CONFIGURATION ---
# =====================================================================
ROUTER_IP = ""
ROUTER_USER = ""
ROUTER_PASS = ""

ARDUINO_IP = ""
ARDUINO_USER = ""
ARDUINO_PASS = ""
REMOTE_DIR = "/path/to/arduino/directory/"

ARDUINO_MAC = ""
LOCAL_RX_IP = ""  # IP of the server device
NUM_RUNS = 10
USE_NOISE = False
NOISE = "10M" # XM: X Mbit/s

IMAGE_FILENAME = "1000073893.jpeg" # Image to transmit

# =====================================================================

def cleanup_local_port(port):
    print(f"[*] Cleaning up any local processes on port {port}...")
    try:
        subprocess.run(f"fuser -k {port}/tcp", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    time.sleep(1)

def verify_remote_files(ssh, remote_dir):
    print("[*] Verifying remote files in directory...")
    required_files = [
        "fast_tx.py",
        IMAGE_FILENAME
    ]
    stdin, stdout, stderr = ssh.exec_command(f"ls -la {remote_dir}")
    files_list = stdout.read().decode('utf-8')
    
    missing_files = [f for f in required_files if f not in files_list]
    if missing_files:
        print(f"[!] WARNING: Missing remote files: {missing_files}")
        return False
    print("[✓] All required remote files verified.")
    return True

def get_router_rssi_robust(router_ssh, mac_address):
    mac_lower = mac_address.lower()
    stdin, stdout, stderr = router_ssh.exec_command("iw dev | grep Interface")
    interfaces = [line.strip().split()[1] for line in stdout if len(line.strip().split()) >= 2]
    if not interfaces:
        interfaces = ["wlan1", "wlan0", "wlan1-1"] # May need to be changed depending on the AP interfaces used to emulate the paper's experiments
        
    for iface in interfaces:
        stdin, stdout, stderr = router_ssh.exec_command(f"iw dev {iface} station get {mac_lower}")
        output = stdout.read().decode('utf-8')
        if "signal:" in output:
            for line in output.split('\n'):
                if "signal:" in line:
                    match = re.search(r"signal:\s+(-?\d+)", line)
                    if match: 
                        return int(match.group(1))
    return None

def run_experiment(run_num, router_ssh, arduino_ssh, mode="baseline"):
    print(f"\n=================== RUN {run_num}/{NUM_RUNS} ({mode.upper()}) ===================")
    
    if mode == "baseline":
        rx_script = "fast_rx.py"
        tx_script = "fast_tx.py"
        port = 9998
        files_to_send = [IMAGE_FILENAME]
        expected_files = 1
    elif mode == "roi":
        rx_script = "fast_rx.py"
        tx_script = "fast_tx.py"
        port = 9999
        files_to_send = ["mask_payload.rsxaz", "latent_payload.rsxz"]
        expected_files = 2
    elif mode == "mask_only":
        rx_script = "fast_rx.py"
        tx_script = "fast_tx.py"
        port = 9997
        files_to_send = ["mask_payload.rsxaz"]
        expected_files = 1
    else:
        raise ValueError(f"[!] Unknown mode specified: {mode}")

    cleanup_local_port(port)
    arduino_ssh.exec_command(f"pkill -f {tx_script}")
    arduino_ssh.exec_command(f"rm -f {REMOTE_DIR}/experiment_*.txt")
    
    print(f"[*] Launching local Receiver ({rx_script}) on port {port}...")
    rx_cmd = ["python3", rx_script, "--port", str(port), "--expected-files", str(expected_files)]
    rx_process = subprocess.Popen(rx_cmd)
    time.sleep(2)
    
    if rx_process.poll() is not None:
        print(f"[!] ERROR: Local Receiver ({rx_script}) crashed on startup!")
        return None

    print(f"[*] Launching remote Transmitter ({tx_script})...")
    transport = arduino_ssh.get_transport()
    tx_channel = transport.open_session()
    
    log_filename = f"experiment_{mode}_{run_num}.txt"
    files_args = " ".join(files_to_send)
    remote_cmd = f"cd {REMOTE_DIR} && python3 {tx_script} --ip {LOCAL_RX_IP} --port {port} --files {files_args} --log-file {log_filename}"
    tx_channel.exec_command(remote_cmd)
    
    rssi_samples = []
    
    while not tx_channel.exit_status_ready():
        rssi = get_router_rssi_robust(router_ssh, ARDUINO_MAC)
        if rssi is not None:
            rssi_samples.append(rssi)
        time.sleep(0.5)
        
    _ = tx_channel.recv_exit_status()
    rx_process.wait()

    # Parse log output from target
    stdin, stdout, stderr = arduino_ssh.exec_command(f"cat {REMOTE_DIR}/{log_filename}")
    log_content = stdout.read().decode('utf-8')

    transmission_delay = None
    if "Status: SUCCESS" in log_content:
        match = re.search(r"Transmission Time:\s+([\d.]+)", log_content)
        if match:
            transmission_delay = float(match.group(1))

    if transmission_delay is None:
        print(f"[!] Run {run_num} failed or log could not be parsed.")
        return None

    avg_rssi = sum(rssi_samples) / len(rssi_samples) if rssi_samples else None
    return avg_rssi, transmission_delay

def main():
    router_ssh = paramiko.SSHClient()
    router_ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    router_ssh.connect(ROUTER_IP, username=ROUTER_USER, password=ROUTER_PASS)
    
    arduino_ssh = paramiko.SSHClient()
    arduino_ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    arduino_ssh.connect(ARDUINO_IP, username=ARDUINO_USER, password=ARDUINO_PASS)
    
    if not verify_remote_files(arduino_ssh, REMOTE_DIR):
        print("[!] Setup failed. Missing required files on target system.")
        router_ssh.close()
        arduino_ssh.close()
        return

    if USE_NOISE:
        arduino_ssh.exec_command("nohup iperf3 -s -p 5202 > /dev/null 2>&1 &")
        time.sleep(1)
        _ = subprocess.Popen(
            ["iperf3", "-c", ARDUINO_IP, "-u", "-b", NOISE, "-t", "360", "-p", "5202"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        time.sleep(1)
        
    modes_to_run = ["baseline", "roi", "mask_only"]
    results = {mode: [] for mode in modes_to_run}
    
    try:
        for mode in modes_to_run:
            print(f"\n==========================================")
            print(f">>> STARTING {mode.upper()} EXPERIMENTS <<<")
            print(f"==========================================")
            for run in range(1, NUM_RUNS + 1):
                res = run_experiment(run, router_ssh, arduino_ssh, mode)
                if res:
                    results[mode].append((run, res[0], res[1]))
                    print(f"[✓] {mode.upper()} Run {run} | RSSI: {res[0]} dBm | Delay: {res[1]}s")
                time.sleep(2)
    finally:
        router_ssh.close()
        arduino_ssh.close()
        if USE_NOISE:
            subprocess.run("pkill -f iperf3", shell=True)
        
        print("\n=================== FINAL RESULTS ===================")
        for mode, data in results.items():
            if data:
                valid_delays = [r[2] for r in data if r[2] is not None]
                avg_delay = sum(valid_delays) / len(valid_delays) if valid_delays else None
                
                valid_rssis = [r[1] for r in data if r[1] is not None]
                avg_rssi = sum(valid_rssis) / len(valid_rssis) if valid_rssis else None
                
                delay_str = f"{avg_delay:.4f}s" if avg_delay is not None else "N/A"
                rssi_str = f"{avg_rssi:.2f} dBm" if avg_rssi is not None else "N/A"
                print(f"Average {mode.upper():<10} | Delay: {delay_str:<9} | RSSI: {rssi_str}")
            else:
                print(f"Average {mode.upper():<10} | No successful runs recorded.")

if __name__ == "__main__":
    main()
