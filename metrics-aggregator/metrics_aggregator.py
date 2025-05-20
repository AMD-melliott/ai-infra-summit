#!/usr/bin/env python3
"""
AMD Instinct GPU Metrics Aggregator

This service collects metrics from AMD Instinct GPUs in different partitioning modes (CPX/SPX),
aggregates them from multiple sources (AMD-SMI, ROCm-SMI, Prometheus), and exposes them
via a REST API for monitoring dashboards.

Key features:
- Multi-source metrics collection with fallback mechanisms
- Support for CPX/SPX partitioning modes
- Automatic GPU type detection (MI300X, MI325X)
- Service metrics aggregation for vLLM instances
- Background metrics collection with configurable update interval

Author: AI Infra Summit Team
License: MIT
"""

import json
import os
import re
import time
import threading
import copy
import urllib.request
import urllib.error
import subprocess
from flask import Flask, jsonify

# Attempt to import psutil and set a flag
try:
    import psutil
    PSUTIL_AVAILABLE = True
    print("psutil library found. Will attempt to trace PIDs for vLLM VRAM attribution.", flush=True)
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil library NOT found. VRAM attribution will sum all processes on a GPU.", flush=True)

app = Flask(__name__)

# --- Global Cache and Lock ---
METRICS_CACHE = {}
cache_lock = threading.Lock()
UPDATE_INTERVAL = 15  # seconds

# --- Configuration ---
CONFIG_FILE_PATH = '/app/config/partition_config.json'
USE_AMD_SMI = True  # Set to True to use amd-smi as additional data source
AMD_SMI_COMMAND = 'amd-smi monitor -q --json'
MAPPER_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gpu_process_mapper.py')

# Default VRAM sizes (in GB) for different GPU types in CPX mode
GPU_VRAM_DEFAULTS = {
    'MI300X': 24.0,  # MI300X has 24GB per partition in CPX mode
    'MI325X': 32.0,  # MI325X has 32GB per partition in CPX mode
    'default': 32.0  # Default fallback if we can't determine GPU type
}

# --- Helper Functions ---
def _get_target_process_info_for_vllm(pid, level=0, max_level=2):
    """
    Internal helper to get command line of a process, tracing up if it's a spawner.
    Returns the command line of the target process (child or grandparent) or None.
    """
    if not PSUTIL_AVAILABLE:
        return pid, None # Target PID, Target CMD line

    if level > max_level:
        return pid, None # Return original PID, but no cmdline found at this depth

    try:
        proc = psutil.Process(pid)
        ppid = proc.ppid()
        cmdline = proc.cmdline()

        if cmdline and any("multiprocessing.spawn" in arg for arg in cmdline) and ppid != 1:
            # print(f"  (L{level} PID {pid} is a spawner, checking parent {ppid} for vLLM server)", flush=True)
            return _get_target_process_info_for_vllm(ppid, level + 1, max_level)
        
        return pid, cmdline # Return current process's PID and its cmdline
    except psutil.NoSuchProcess:
        # print(f"  (L{level} PID {pid} not found by psutil when tracing for vLLM server)", flush=True)
        return pid, None 
    except psutil.AccessDenied:
        # print(f"  (L{level} PID {pid} access denied by psutil when tracing for vLLM server)", flush=True)
        return pid, None
    except Exception as e:
        # print(f"  (L{level} PID {pid} error with psutil: {e} when tracing for vLLM server)", flush=True)
        return pid, None

def find_port_in_cmdline(cmdline):
    """Extracts the port number if --port is present in the command line."""
    if not cmdline:
        return None
    try:
        port_index = cmdline.index('--port')
        if port_index + 1 < len(cmdline):
            # Ensure the port value is a digit before returning
            port_val = cmdline[port_index + 1]
            if port_val.isdigit():
                return port_val
    except ValueError:
        pass # --port not found or port_val not a number
    return None

def map_port_to_partition_id(port, partition_config):
    """Maps a port to a partition ID string using the config."""
    if not port or not partition_config:
        return None
    for part_id_str, config_details in partition_config.items():
        if isinstance(config_details, dict):
            metrics_url = config_details.get('metrics_url', '')
            # Example metrics_url: "http://localhost:8000/metrics"
            # Ensure port is compared as a string
            if f":{str(port)}/metrics" in metrics_url:
                return part_id_str # Returns string like "0", "1", ..., "40"
    return None

def load_partition_config():
    """
    Load the GPU partition configuration from a JSON file.
    Returns an empty dict if the file cannot be found or parsed.
    """
    try:
        with open(CONFIG_FILE_PATH, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading partition config '{CONFIG_FILE_PATH}': {e}", flush=True)
        return {}

def parse_prometheus_metric_line(line):
    """
    Parse a single line of Prometheus metrics output.
    Returns (metric_name, labels, value) tuple or (None, None, None) if parsing fails.
    """
    try:
        line = line.strip()
        if not line or line.startswith('#'):
            return None, None, None
            
        match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)(?:\{([^}]+)\})?\s+(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)$', line)
        if not match:
            return None, None, None
            
        metric_name = match.group(1)
        labels_str = match.group(2)
        value_str = match.group(3)
        
        labels = {}
        if labels_str:
            for part in re.findall(r'([^=,]+)="([^"]*)"', labels_str):
                labels[part[0]] = part[1]
                
        value = float(value_str)
        return metric_name, labels, value
    except Exception as e:
        print(f"Error parsing Prometheus line '{line}': {e}", flush=True)
        return None, None, None

def fetch_prometheus_metrics(url):
    """
    Fetch raw metrics data from a Prometheus endpoint.
    Returns the text content or None if the request fails.
    """
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            if response.status == 200:
                return response.read().decode('utf-8')
            else:
                print(f"Error fetching {url}: Status code {response.status}", flush=True)
                return None
    except urllib.error.URLError as e:
        print(f"Error fetching {url}: {e.reason}", flush=True)
        return None
    except Exception as e:
        print(f"Unexpected error fetching {url}: {e}", flush=True)
        return None

def fetch_amd_smi_data():
    """
    Execute amd-smi command and parse its JSON output.
    Returns parsed data or None if command fails or produces invalid output.
    """
    try:
        result = subprocess.run(AMD_SMI_COMMAND, shell=True, check=True, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                               universal_newlines=True, timeout=20)
        amd_data = json.loads(result.stdout)
        print(f"Successfully fetched data from amd-smi for {len(amd_data)} GPU devices", flush=True)
        return amd_data
    except subprocess.CalledProcessError as e:
        print(f"Error running amd-smi command: {e}", flush=True)
        return None
    except subprocess.TimeoutExpired:
        print(f"Timeout running amd-smi command", flush=True)
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from amd-smi: {e}", flush=True)
        return None
    except Exception as e:
        print(f"Unexpected error with amd-smi: {e}", flush=True)
        return None

def get_metrics_from_rocm_smi_output():
    """
    Executes rocm-smi command and parses its output to extract VRAM and GPU utilization 
    percentages for each GPU device.
    
    Returns:
        dict: Mapping of device ID to dict containing 'vram_percentage' and 'gpu_utilization'
              or None if command fails or output can't be parsed
    """
    try:
        command = ['rocm-smi'] 
        print(f"Executing command: {' '.join(command)}", flush=True)
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
        
        if result.returncode != 0:
            print(f"rocm-smi command failed with return code {result.returncode}. Stderr: {result.stderr}", flush=True)
            # Fallback: Try with sudo if permission denied
            if "permission denied" in result.stderr.lower() or "root privileges" in result.stderr.lower():
                print("rocm-smi failed due to permissions, trying with sudo...", flush=True)
                sudo_command = ['sudo', 'rocm-smi']
                print(f"Executing command: {' '.join(sudo_command)}", flush=True)
                result = subprocess.run(sudo_command, capture_output=True, text=True, check=True, timeout=35)
            else:
                 result.check_returncode()

        output = result.stdout
        metrics = {}
        lines = output.splitlines()
        data_started = False
        
        # Pattern to match the rocm-smi header line
        header_pattern = re.compile(r"^Device\s+Node\s+IDs\s+Temp\s+Power\s+Partitions\s+SCLK\s+MCLK\s+Fan\s+Perf\s+PwrCap\s+VRAM%\s+GPU%$")
        
        for line_idx, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue

            if header_pattern.match(line_stripped):
                data_started = True
                print(f"rocm-smi parser: Header line found at index {line_idx}. Data parsing started.", flush=True)
                continue
            
            if "End of ROCm SMI Log" in line_stripped:
                print(f"rocm-smi parser: End of log found at index {line_idx}.", flush=True)
                data_started = False
                break 

            if data_started:
                if not line_stripped or not line_stripped[0].isdigit():
                    continue
                
                parts = line_stripped.split()
                if len(parts) >= 12:
                    try:
                        device_id_str = parts[0]
                        vram_percent_str = parts[-2].replace('%', '')
                        gpu_util_str = parts[-1].replace('%', '')
                        
                        # Basic validation before conversion
                        if not (vram_percent_str.replace('.', '', 1).isdigit() and gpu_util_str.isdigit()):
                            print(f"rocm-smi parser: Invalid number format in line: '{line_stripped}'. VRAM_str: '{vram_percent_str}', GPU_str: '{gpu_util_str}'", flush=True)
                            continue

                        metrics[device_id_str] = {
                            'vram_percentage': float(vram_percent_str),
                            'gpu_utilization': int(gpu_util_str)
                        }
                    except (ValueError, IndexError) as e:
                        print(f"rocm-smi parser: Error parsing line parts: '{line_stripped}'. Error: {e}", flush=True)
                        continue 
        
        if not metrics:
            print("rocm-smi parser: No metrics successfully parsed. This might be due to unexpected output format or no data lines after header.", flush=True)
            return None
        
        print(f"rocm-smi parser: Successfully parsed metrics for {len(metrics)} devices.", flush=True)
        return metrics

    except subprocess.TimeoutExpired:
        print("Timeout executing rocm-smi.", flush=True)
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error executing rocm-smi (CalledProcessError after potential sudo retry): {e}. Stderr: {e.stderr}", flush=True)
        return None
    except FileNotFoundError:
        print("Error: rocm-smi (or sudo) command not found.", flush=True)
        return None
    except Exception as e:
        print(f"An unexpected error occurred in get_metrics_from_rocm_smi_output: {type(e).__name__} - {e}", flush=True)
        return None

def execute_gpu_mapper_script():
    """
    Executes the gpu_process_mapper.py script and returns its parsed JSON output.
    Returns None if script execution or JSON parsing fails.
    """
    try:
        print(f"Executing GPU mapper script: {MAPPER_SCRIPT_PATH}", flush=True)
        # Try without sudo first
        result = subprocess.run([MAPPER_SCRIPT_PATH], capture_output=True, text=True, check=False, timeout=60)
        
        # If that fails, try with sudo
        if result.returncode != 0:
            print(f"GPU Mapper script failed with return code {result.returncode}, trying with sudo", flush=True)
            result = subprocess.run(['sudo', MAPPER_SCRIPT_PATH], capture_output=True, text=True, check=False, timeout=60)

        if result.returncode != 0:
            print(f"GPU Mapper Script Error (Return Code {result.returncode}):", flush=True)
            print(f"  Stdout: {result.stdout.strip()}", flush=True)
            print(f"  Stderr: {result.stderr.strip()}", flush=True)
            try: # Try to parse stdout as JSON even if error, might contain JSON error msg
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                return {"error": f"GPU Mapper script failed and stdout was not valid JSON."}

        return json.loads(result.stdout)
        
    except subprocess.TimeoutExpired:
        print("Timeout executing GPU mapper script.", flush=True)
        return {"error": "Timeout executing GPU mapper script."}
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from GPU mapper script: {e}", flush=True)
        return {"error": f"Error parsing JSON from GPU mapper script: {e}"}
    except FileNotFoundError:
        print(f"Error: GPU mapper script '{MAPPER_SCRIPT_PATH}' not found.", flush=True)
        return {"error": f"GPU mapper script '{MAPPER_SCRIPT_PATH}' not found."}
    except Exception as e:
        print(f"An unexpected error occurred while executing GPU mapper script: {e}", flush=True)
        return {"error": f"Unexpected error executing GPU mapper script: {e}"}

def detect_gpu_type(amd_smi_data):
    """
    Detect GPU hardware type (MI300X, MI325X, etc.) from AMD SMI data.
    
    Args:
        amd_smi_data: Parsed JSON output from amd-smi command
        
    Returns:
        str: Detected GPU type or None if detection fails
    """
    if not amd_smi_data or not isinstance(amd_smi_data, list) or len(amd_smi_data) == 0:
        print("Cannot detect GPU type: No AMD SMI data available", flush=True)
        return None

    # Try to detect MI325X by the total reported VRAM in each GPU entry
    # MI325X in SPX mode has 256GB of HBM3 memory per GPU
    has_large_vram = False
    for gpu_entry in amd_smi_data:
        vram_total_info = gpu_entry.get('vram_total', {})
        if isinstance(vram_total_info, dict):
            vram_total_val = vram_total_info.get('value', 0.0)
            vram_total_unit = vram_total_info.get('unit', 'GB').upper()
            if vram_total_unit == 'GB' and vram_total_val >= 240.0:  # MI325X has 256GB VRAM
                has_large_vram = True
                print(f"Detected large VRAM GPU (likely MI325X): {vram_total_val} {vram_total_unit}", flush=True)
                break

    # Check product name fields for GPU type identification
    gpu_types_detected = set()
    for gpu_entry in amd_smi_data:
        # Various fields that might contain product name information
        product_name = None
        
        # Different versions of amd-smi might use different field names
        if 'product_name' in gpu_entry:
            product_name = gpu_entry['product_name']
        elif 'chip_name' in gpu_entry:
            product_name = gpu_entry['chip_name']
        elif 'product' in gpu_entry and isinstance(gpu_entry['product'], dict):
            product_name = gpu_entry['product'].get('name')
        elif 'card_series' in gpu_entry:
            product_name = gpu_entry['card_series']
        
        if product_name:
            print(f"Found product name: {product_name}", flush=True)
            if 'MI300X' in product_name:
                gpu_types_detected.add('MI300X')
            elif 'MI325X' in product_name:
                gpu_types_detected.add('MI325X')
            elif 'MI300' in product_name:  # Fallback for general MI300 family
                gpu_types_detected.add('MI300X')
    
    # If we found multiple GPU types, prefer MI325X over MI300X (higher capability)
    detected_type = None
    if 'MI325X' in gpu_types_detected:
        detected_type = 'MI325X'
    elif 'MI300X' in gpu_types_detected:
        detected_type = 'MI300X'
    elif gpu_types_detected:
        detected_type = list(gpu_types_detected)[0]  # Just take the first one
    elif has_large_vram:
        # If we didn't detect any GPU types from product name but we found large VRAM
        detected_type = 'MI325X'
    
    # Final detection output
    print(f"GPU types detected from names: {gpu_types_detected}", flush=True)
    print(f"Final detected GPU type: {detected_type or 'Unknown'}", flush=True)
    
    # Default to MI325X if detection failed but VRAM size suggests it
    if not detected_type and has_large_vram:
        print("Defaulting to MI325X based on VRAM size of 256GB", flush=True)
        detected_type = 'MI325X'
        
    return detected_type

def process_amd_smi_data(amd_data):
    """
    Process raw AMD SMI data to extract GPU hardware details and a list of all processes.
    This version does NOT try to map vLLM VRAM; that's done later.
    
    Args:
        amd_data: JSON data from AMD SMI output
        
    Returns:
        dict: Mapping of amd_smi_gpu_id_str to a dict containing:
              'vram_total_gb', 'utilization', and 'all_processes_reported' (list of dicts)
    """
    if not amd_data:
        return {}
            
    amd_smi_raw_gpu_info = {}
    for gpu_entry in amd_data:
        gpu_id = gpu_entry.get('gpu')
        if gpu_id is None:
            continue
        gpu_id_str = str(gpu_id)

        vram_total_info = gpu_entry.get('vram_total', {})
        vram_total_val = vram_total_info.get('value', 0.0)
        vram_total_unit = vram_total_info.get('unit', 'GB').upper()
        actual_vram_total_gb = 0.0
        if vram_total_val == 0.0:
            actual_vram_total_gb = 0.0
        elif vram_total_unit in ['MB', 'MIB']:
            actual_vram_total_gb = vram_total_val / 1024.0
        elif vram_total_unit == 'B':
            actual_vram_total_gb = vram_total_val / (1024.0 * 1024.0 * 1024.0)
        elif vram_total_unit in ['KB', 'KIB']:
            actual_vram_total_gb = vram_total_val / (1024.0 * 1024.0)
        else: # Assume GB
            actual_vram_total_gb = vram_total_val

        processes_reported_on_this_gpu = []
        process_list = gpu_entry.get('process_list', [])
        for proc_entry in process_list:
            if isinstance(proc_entry, dict) and 'process_info' in proc_entry:
                proc_info = proc_entry['process_info']
                if isinstance(proc_info, dict):
                    child_pid = proc_info.get('pid', 'N/A')
                    name = proc_info.get('name', 'N/A')
                    proc_vram_gb_val = 0.0
                    if 'memory_usage' in proc_info and 'vram_mem' in proc_info['memory_usage']:
                        vram_mem_info = proc_info['memory_usage']['vram_mem']
                        vram_value_bytes = vram_mem_info.get('value', 0)
                        proc_unit = vram_mem_info.get('unit', 'B').upper()
                        if proc_unit == 'B':
                            proc_vram_gb_val = vram_value_bytes / (1024.0**3)
                        elif proc_unit in ['KB', 'KIB']:
                            proc_vram_gb_val = vram_value_bytes / (1024.0**2)
                        elif proc_unit in ['MB', 'MIB']:
                            proc_vram_gb_val = vram_value_bytes / 1024.0
                        elif proc_unit in ['GB', 'GIB']:
                            proc_vram_gb_val = vram_value_bytes
                        else:
                            proc_vram_gb_val = vram_value_bytes / (1024.0**3)
                    
                    if child_pid != 'N/A': # Only add if PID is valid
                        processes_reported_on_this_gpu.append({
                            'pid': str(child_pid), # Ensure PID is string
                            'name': name,
                            'vram_gb': round(proc_vram_gb_val, 2)
                        })
                        
        amd_smi_raw_gpu_info[gpu_id_str] = {
            'vram_total_gb': actual_vram_total_gb,
            'utilization': gpu_entry.get('gfx', {}).get('value', 0),
            'all_processes_reported': processes_reported_on_this_gpu
        }
        
    return amd_smi_raw_gpu_info

def get_service_metrics_data(partition_config):
    """
    Collect metrics from vLLM service instances defined in the partition configuration.
    
    Args:
        partition_config: Parsed partition configuration containing service info
        
    Returns:
        dict: Service metrics aggregated by service type
    """
    services_data = {}
    vllm_total_running = 0
    vllm_total_waiting = 0
    vllm_kv_cache_sum_perc = 0.0
    vllm_instance_count = 0
    vllm_fetch_errors = 0
    
    # Collect metrics from vLLM instances
    for partition_id, config in partition_config.items():
        if isinstance(config, dict) and config.get('service_type') == 'vllm' and config.get('metrics_url'):
            url = config['metrics_url']
            print(f"Fetching metrics for vLLM partition {partition_id} from {url}", flush=True)
            raw_metrics = fetch_prometheus_metrics(url)
            
            if raw_metrics:
                current_running_val = parse_prometheus_metric(raw_metrics, 'vllm:num_requests_running')
                current_waiting_val = parse_prometheus_metric(raw_metrics, 'vllm:num_requests_waiting')
                current_kv_perc_val = parse_prometheus_metric(raw_metrics, 'vllm:gpu_cache_usage_perc')
                
                if current_running_val is not None and current_waiting_val is not None and current_kv_perc_val is not None:
                    vllm_instance_count += 1
                    current_running = int(current_running_val)
                    current_waiting = int(current_waiting_val)
                    current_kv_perc = current_kv_perc_val * 100
                    
                    vllm_total_running += current_running
                    vllm_total_waiting += current_waiting
                    vllm_kv_cache_sum_perc += current_kv_perc
                    
                    print(f"  Parsed metrics for {partition_id}: Run={current_running}, Wait={current_waiting}, KV%={current_kv_perc:.1f}", flush=True)
                else:
                    print(f"  Warning: Could not parse one or more metrics for vLLM partition {partition_id} from {url}", flush=True)
                    vllm_fetch_errors += 1
            else:
                 print(f"  Error: Failed to fetch metrics for vLLM partition {partition_id} from {url}", flush=True)
                 vllm_fetch_errors += 1
    
    # Aggregate vLLM metrics
    if vllm_instance_count > 0:
         vllm_avg_kv_cache_perc = vllm_kv_cache_sum_perc / vllm_instance_count
         services_data['vllm_aggregate'] = {
             'service_type': 'vllm_aggregate',
             'status': 'ok',
             'requests_running': vllm_total_running,
             'requests_waiting': vllm_total_waiting,
             'kv_cache_usage': round(vllm_avg_kv_cache_perc, 1),
             'instance_count': vllm_instance_count,
             'fetch_errors': vllm_fetch_errors
         }
         print(f"Aggregated vLLM metrics: Instances={vllm_instance_count}, Running={vllm_total_running}, Waiting={vllm_total_waiting}, Avg KV%={vllm_avg_kv_cache_perc:.1f}, Errors={vllm_fetch_errors}", flush=True)
    elif vllm_fetch_errors > 0:
         services_data['vllm_aggregate'] = {
             'service_type': 'vllm_aggregate',
             'status': 'error',
             'requests_running': 'Error',
             'requests_waiting': 'Error',
             'kv_cache_usage': 'Error',
             'instance_count': 0,
             'fetch_errors': vllm_fetch_errors,
             'message': 'Failed to fetch/parse metrics for all configured vLLM instances.'
         }
         print("Could not aggregate vLLM metrics due to fetch/parse errors for all configured instances.", flush=True)
    else:
         services_data['vllm_aggregate'] = {
             'service_type': 'vllm_aggregate',
             'status': 'no_instances_configured',
             'requests_running': 'N/A',
             'requests_waiting': 'N/A',
             'kv_cache_usage': 'N/A',
             'instance_count': 0,
             'fetch_errors': 0,
             'message': "No vLLM instances found in config with 'service_type: vllm' and 'metrics_url'."
         }
         print("No vLLM instances found in config with 'service_type: vllm' and 'metrics_url'.", flush=True)
         
    return services_data

def parse_prometheus_metric(text_data, metric_name_to_find):
    """
    Parse a specific metric from Prometheus text format.
    
    This function extracts a single metric value by name from Prometheus exposition format.
    It handles metrics with or without labels and properly parses the numerical value.
    
    Args:
        text_data (str): The raw Prometheus metrics text
        metric_name_to_find (str): The name of the metric to extract
        
    Returns:
        float: The metric value or None if not found or invalid format
    """
    # Pattern to find the metric and its value, accounting for optional labels
    pattern_str = rf"^{re.escape(metric_name_to_find)}(?:\{{[^}}]*\}})?\s+(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)$"
    
    for line in text_data.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        match = re.match(pattern_str, line)
        if match:
            try:
                value = float(match.group(1))  # The value is the first capture group
                return value
            except ValueError as e:
                print(f"Error converting value for metric '{metric_name_to_find}'. Line: '{line}', Error: {e}", flush=True)
                return None
    
    return None

# --- Placeholder for future direct SMI processing logic ---
def build_gpu_metrics_from_smi_source(raw_smi_data, partition_config_map, cpx_part_vram_gb, gpu_type_detected):
    """
    Process AMD SMI JSON output to extract per-partition metrics, applying CPX partitioning rules.
    
    Note: This is a simplified implementation that would be replaced by a more robust
          implementation using the ROCm SMI Python library in production.
    
    Returns:
        dict: Mapping of partition IDs to metrics including VRAM usage and utilization
    """
    print(f"Processing SMI data for {gpu_type_detected}, CPX VRAM: {cpx_part_vram_gb}GB", flush=True)
    processed_metrics = {}
    num_partitions_configured = len(partition_config_map)
    is_cpx = (gpu_type_detected in ['MI300X', 'MI325X'] and num_partitions_configured >= 8) 

    if raw_smi_data and isinstance(raw_smi_data, list) and len(raw_smi_data) > 0:
        # Using simplified metrics for demonstration
        for i in range(num_partitions_configured):
            partition_id_str = str(i)
            
            processed_metrics[partition_id_str] = {
                'vram_used_gb': 0.1,  # Default placeholder value
                'vram_total_gb': cpx_part_vram_gb if is_cpx else GPU_VRAM_DEFAULTS.get(gpu_type_detected, GPU_VRAM_DEFAULTS['default']),
                'utilization': 0,
                '_source': 'amd_smi_processed'
            }
    else:
        # No raw_smi_data, fill with defaults
        print("No raw SMI data available, creating default entries", flush=True)
        for i in range(num_partitions_configured):
            partition_id_str = str(i)
            processed_metrics[partition_id_str] = {
                'vram_used_gb': 0.0,
                'vram_total_gb': cpx_part_vram_gb if is_cpx else GPU_VRAM_DEFAULTS.get(gpu_type_detected, GPU_VRAM_DEFAULTS['default']),
                'utilization': 0,
                '_source': 'amd_smi_nodata'
            }
    
    return processed_metrics
# --- End of Placeholder ---

# --- Core Metrics Aggregation Logic ---
def update_metrics_cache():
    """
    Core function that collects and aggregates metrics from multiple sources:
    
    1. Partition configuration: Defines GPU partitions and their assignments.
    2. Service metrics (e.g., vLLM): Fetched from their Prometheus endpoints.
    3. AMD SMI: Primary source for GPU hardware metrics including process-specific VRAM.
    4. ROCm SMI: Used as an override for VRAM and GPU utilization when enabled.
    
    The aggregation process involves:
    - Collecting metrics from all available sources.
    - Determining GPU partitioning mode (CPX/SPX) primarily from config and AMD SMI data.
    - Applying source priority for GPU hardware (ROCm SMI > AMD SMI > defaults).
    - Normalizing metrics to consistent units and formats.
    - Calculating summary statistics.
    
    The resulting metrics are stored in the global METRICS_CACHE which is
    accessed by API endpoints to serve monitoring dashboards.
    """
    global METRICS_CACHE
    print("update_metrics_cache: Attempting to update metrics cache...", flush=True)
    
    collected_data_successfully = False
    new_collected_data = {}

    try:
        # Load partition configuration
        partition_config = load_partition_config()
        print(f"update_metrics_cache: Loaded partition_config: {bool(partition_config)}", flush=True)

        # Get service-level metrics (vLLM, etc.)
        service_metrics = get_service_metrics_data(partition_config)
        print(f"update_metrics_cache: Fetched service_metrics: {bool(service_metrics)}", flush=True)
    
        partitions_output = []
        total_vram_used_gb = 0.0
        total_vram_available_gb = 0.0
        partitioning_mode = "unknown"
        
        amd_smi_metrics = {} # This will be populated differently or not used directly
        detected_gpu_type = None
        cpx_default_vram_per_partition = GPU_VRAM_DEFAULTS['default']  # Start with default
        amd_smi_raw_gpu_info = {} # To store output from new process_amd_smi_data
        mapper_script_output = None

        # Always try to execute the mapper script first if psutil is available
        # as it provides the crucial vLLM VRAM mapping and SMI GPU ID linkage.
        if PSUTIL_AVAILABLE:
            print("update_metrics_cache: Attempting to execute GPU mapper script...", flush=True)
            mapper_script_output = execute_gpu_mapper_script()
            if mapper_script_output and 'error' in mapper_script_output:
                print(f"update_metrics_cache: GPU Mapper script returned an error: {mapper_script_output['error']}", flush=True)
                # Continue, but parsed_gpu_metrics might be less accurate or default.
                mapper_script_output = None # Nullify to indicate failure for later checks
            elif mapper_script_output:
                print("update_metrics_cache: Successfully received data from GPU mapper script.", flush=True)
                # Log the partition information for better troubleshooting
                vllm_mapping = mapper_script_output.get("vllm_vram_by_true_partition", {})
                smi_mapping = mapper_script_output.get("true_partition_to_smi_gpu_id", {})
                print(f"Mapper script reports VRAM data for {len(vllm_mapping)} partitions and SMI ID mappings for {len(smi_mapping)} partitions", flush=True)
                
                # Log a few sample partitions for validation
                sample_partitions = ["36", "37", "38", "39", "40", "41", "42", "43"]
                for p_id in sample_partitions:
                    if p_id in vllm_mapping:
                        print(f"Partition {p_id} VRAM: {vllm_mapping[p_id]}GB, SMI ID: {smi_mapping.get(p_id, 'Not mapped')}", flush=True)
            else:
                print("update_metrics_cache: GPU mapper script execution failed or returned no data. VRAM attribution will be limited.", flush=True)
        else:
            print("update_metrics_cache: psutil not available, GPU mapper script will not be run. VRAM attribution will be limited.", flush=True)

        # Fetch raw amd-smi data if not already provided by the mapper or if mapper failed
        # The mapper script's all_smi_gpu_details can substitute for this if available.
        if mapper_script_output and mapper_script_output.get('all_smi_gpu_details'):
            print("update_metrics_cache: Using amd-smi base details from mapper script output.", flush=True)
            # Reconstruct a simplified amd_smi_data-like structure for detect_gpu_type if needed
            # For now, detect_gpu_type relies on the full amd-smi dump, so we might still need to fetch it.
            # Let's fetch amd-smi data regardless for detect_gpu_type and for processes not covered by mapper.
            pass # We'll use mapper_script_output['all_smi_gpu_details'] later

        if USE_AMD_SMI: # Still fetch full amd-smi for GPU type detection and non-vLLM process listing
            print("update_metrics_cache: Fetching full AMD SMI data (for GPU type, non-vLLM processes)...", flush=True)
            amd_smi_data_full_dump = fetch_amd_smi_data()
            if amd_smi_data_full_dump:
                detected_gpu_type = detect_gpu_type(amd_smi_data_full_dump)
                if detected_gpu_type and detected_gpu_type in GPU_VRAM_DEFAULTS:
                    cpx_default_vram_per_partition = GPU_VRAM_DEFAULTS[detected_gpu_type]
                
                # Process the full dump to get a list of all processes reported by SMI on each of its GPUs
                # This is useful for debugging or if the mapper doesn't cover something.
                amd_smi_raw_gpu_info = process_amd_smi_data(amd_smi_data_full_dump) # process_amd_smi_data now just extracts raw info
                print(f"update_metrics_cache: Extracted raw process info from AMD SMI full dump for {len(amd_smi_raw_gpu_info)} SMI GPUs.", flush=True)

                if len(amd_smi_raw_gpu_info) == 64 and partitioning_mode != "CPX":
                    partitioning_mode = "CPX"
                elif len(amd_smi_raw_gpu_info) > 0 and len(amd_smi_raw_gpu_info) <= 8 and partitioning_mode != "SPX":
                     partitioning_mode = "SPX"
                print(f"update_metrics_cache: Partitioning mode based on SMI full dump: {partitioning_mode}", flush=True)
            else:
                print("update_metrics_cache: Failed to fetch full AMD SMI dump.", flush=True)
        
        # Extract data from mapper script output if available
        vllm_vram_by_true_partition = {}
        true_partition_to_smi_gpu_id = {}
        smi_gpu_details_from_mapper = {}

        if mapper_script_output:
            vllm_vram_by_true_partition = mapper_script_output.get("vllm_vram_by_true_partition", {})
            true_partition_to_smi_gpu_id = mapper_script_output.get("true_partition_to_smi_gpu_id", {})
            smi_gpu_details_from_mapper = mapper_script_output.get("all_smi_gpu_details", {})
            print(f"update_metrics_cache: Data from mapper - VRAM map entries: {len(vllm_vram_by_true_partition)}, True->SMI map entries: {len(true_partition_to_smi_gpu_id)}, SMI details entries: {len(smi_gpu_details_from_mapper)}", flush=True)

        # Initialize parsed_gpu_metrics for all configured partitions (0-63 or from config)
        parsed_gpu_metrics = {}
        expected_partition_ids = set(partition_config.keys()) if partition_config else set()
        if partitioning_mode == "CPX": # Ensure 0-63 are considered if in CPX mode
            for i in range(64): expected_partition_ids.add(str(i))
        elif not expected_partition_ids and smi_gpu_details_from_mapper: # No config, but have mapper smi data
             for smi_id_key in smi_gpu_details_from_mapper.keys(): expected_partition_ids.add(smi_id_key)
        elif not expected_partition_ids and amd_smi_raw_gpu_info: # No config, but have raw_smi data
             for smi_id_key in amd_smi_raw_gpu_info.keys(): expected_partition_ids.add(smi_id_key)
        
        print(f"update_metrics_cache: Initializing parsed_gpu_metrics for TRUE partition IDs: {sorted(list(expected_partition_ids), key=lambda x: int(x) if x.isdigit() else float('inf'))}", flush=True)
        
        for true_pid_str in expected_partition_ids: # Iterate by TRUE partition ID (e.g. "0" to "63")
            # 1. VRAM Used (primary source: mapper script's vLLM VRAM sum for this true_pid_str)
            vram_used_for_this_true_pid = vllm_vram_by_true_partition.get(true_pid_str, 0.0)
            source_flags = ["init"]

            # 2. Determine the underlying AMD SMI GPU ID for this true_pid_str
            # This SMI ID is used to fetch utilization and total VRAM.
            underlying_smi_gpu_id_str = true_partition_to_smi_gpu_id.get(true_pid_str)
            
            # 3. Get Utilization and Total VRAM for that underlying SMI GPU
            util_for_this_true_pid = 0
            total_vram_for_this_true_pid = 0.0 # This will be adjusted by CPX logic if applicable
            all_processes_on_underlying_smi_gpu = []
            total_process_vram_on_underlying_smi_gpu = 0.0

            if underlying_smi_gpu_id_str:
                source_flags.append("smi_id_mapped")
                # Prefer details from mapper script if available for this SMI ID
                if underlying_smi_gpu_id_str in smi_gpu_details_from_mapper:
                    details = smi_gpu_details_from_mapper[underlying_smi_gpu_id_str]
                    util_for_this_true_pid = details.get('utilization', 0)
                    total_vram_for_this_true_pid = details.get('vram_total_gb', 0.0)
                    source_flags.append("mapper_smi_details")
                # Fallback to amd_smi_raw_gpu_info if mapper didn't have it (shouldn't happen if mapper ran ok)
                elif underlying_smi_gpu_id_str in amd_smi_raw_gpu_info:
                    details = amd_smi_raw_gpu_info[underlying_smi_gpu_id_str]
                    util_for_this_true_pid = details.get('utilization', 0)
                    total_vram_for_this_true_pid = details.get('vram_total_gb', 0.0)
                    source_flags.append("raw_smi_details_fallback")
                
                # Get all processes and sum of their VRAM from the underlying SMI GPU (for debug/info)
                if underlying_smi_gpu_id_str in amd_smi_raw_gpu_info: # Use the full dump for all processes
                    all_processes_on_underlying_smi_gpu = amd_smi_raw_gpu_info[underlying_smi_gpu_id_str].get('all_processes_reported', [])
                    total_process_vram_on_underlying_smi_gpu = sum(p.get('vram_gb', 0.0) for p in all_processes_on_underlying_smi_gpu)
            else:
                # This true_pid_str was not mapped to any SMI GPU ID by the mapper.
                # This could happen for partitions that genuinely have no vLLM server (e.g. Free, sglang).
                # We might still have base SMI data if true_pid_str happens to match an SMI_GPU_ID (e.g. for 0-7 in SPX).
                source_flags.append("no_smi_id_mapped")
                if true_pid_str in smi_gpu_details_from_mapper: # Unlikely if no mapping, but check
                    details = smi_gpu_details_from_mapper[true_pid_str]
                    util_for_this_true_pid = details.get('utilization', 0)
                    total_vram_for_this_true_pid = details.get('vram_total_gb', 0.0)
                    source_flags.append("direct_smi_details_no_map")
                elif true_pid_str in amd_smi_raw_gpu_info:
                    details = amd_smi_raw_gpu_info[true_pid_str]
                    util_for_this_true_pid = details.get('utilization', 0)
                    total_vram_for_this_true_pid = details.get('vram_total_gb', 0.0)
                    all_processes_on_underlying_smi_gpu = details.get('all_processes_reported', [])
                    total_process_vram_on_underlying_smi_gpu = sum(p.get('vram_gb', 0.0) for p in all_processes_on_underlying_smi_gpu)
                    source_flags.append("direct_raw_smi_details_no_map")

            # Fallback for vram_used_gb if no vLLM mapped and ROCM override is off:
            # Use sum of all processes on the *underlying* SMI GPU (if one was found for this true_pid)
            # or sum of processes on the *true_pid itself* if treated as an SMI ID (if no underlying_smi_gpu_id_str)
            if vram_used_for_this_true_pid == 0.0 and not os.getenv('ROCM_SMI_VRAM_OVERRIDE', 'false').lower() == 'true':
                if total_process_vram_on_underlying_smi_gpu > 0.0: #This is sum from the identified SMI GPU
                    vram_used_for_this_true_pid = total_process_vram_on_underlying_smi_gpu
                    source_flags.append("fallback_total_smi_proc_vram_mapped_smi_id")
                    # print(f"  Partition {true_pid_str}: vLLM VRAM 0. Using total process VRAM from mapped SMI GPU {underlying_smi_gpu_id_str if underlying_smi_gpu_id_str else 'N/A'}: {vram_used_for_this_true_pid:.2f} GB.", flush=True)
                # If total_process_vram_on_underlying_smi_gpu is also 0, vram_used remains 0.0
            elif vram_used_for_this_true_pid > 0.0:
                source_flags.append("mapper_vllm_vram_used")

            # Set a reliable default for CPX partitions
            if partitioning_mode == "CPX":
                # Override incorrect 256GB values with the correct CPX partition size
                total_vram_for_this_true_pid = cpx_default_vram_per_partition
            
            parsed_gpu_metrics[true_pid_str] = {
                'partition_id': true_pid_str,
                '_source': "+".join(source_flags),
                'vram_used_gb': round(vram_used_for_this_true_pid, 2),
                'vram_total_gb': total_vram_for_this_true_pid, # Will be adjusted by CPX logic
                'processes': all_processes_on_underlying_smi_gpu, # All processes amd-smi saw on the *underlying SMI GPU*
                'process_vram_gb': round(total_process_vram_on_underlying_smi_gpu, 2), # Sum from *underlying SMI GPU*
                'vllm_specific_process_vram_gb': round(vllm_vram_by_true_partition.get(true_pid_str, 0.0), 2), # Mapped vLLM VRAM
                'utilization': util_for_this_true_pid,
                '_underlying_smi_gpu_id': underlying_smi_gpu_id_str # For debugging
            }
            
            # CPX mode default VRAM total adjustment
            current_pid_type = 'unknown_init'
            if partitioning_mode == "CPX":
                parsed_gpu_metrics[true_pid_str]['vram_total_gb'] = cpx_default_vram_per_partition
                current_pid_type = "cpx_init"
            elif partitioning_mode == "SPX" and total_vram_for_this_true_pid == 0.0 : # SPX but total is 0
                # This can happen if the true_pid_str (0-7) wasn't in SMI output, or had no VRAM total.
                # We need a sensible default for SPX total VRAM.
                default_spx_total = GPU_VRAM_DEFAULTS.get(detected_gpu_type) or GPU_VRAM_DEFAULTS['default']
                if detected_gpu_type == 'MI325X': default_spx_total = 256.0 # More specific default
                elif detected_gpu_type == 'MI300X': default_spx_total = 192.0 # More specific default
                parsed_gpu_metrics[true_pid_str]['vram_total_gb'] = default_spx_total
                parsed_gpu_metrics[true_pid_str]['_source'] += "+spx_total_defaulted"
                current_pid_type = "spx_init_total_defaulted"
            elif partitioning_mode == "SPX":
                 current_pid_type = "spx_init"
            
            parsed_gpu_metrics[true_pid_str]['partition_type'] = current_pid_type
            parsed_gpu_metrics[true_pid_str]['used_mb'] = parsed_gpu_metrics[true_pid_str]['vram_used_gb'] * 1024.0
            parsed_gpu_metrics[true_pid_str]['total_mb'] = parsed_gpu_metrics[true_pid_str]['vram_total_gb'] * 1024.0

        # ROCM_SMI_OVERRIDE logic (operates on parsed_gpu_metrics, which is keyed by true_pid_str)
        rocm_smi_raw_metrics = get_metrics_from_rocm_smi_output() # This returns dict with vram_percentage and gpu_utilization

        if rocm_smi_raw_metrics:
            print(f"update_metrics_cache: ROCm SMI data found for {len(rocm_smi_raw_metrics)} devices. Processing overrides/updates...", flush=True)
            if len(rocm_smi_raw_metrics) == 64 and partitioning_mode != "CPX":
                 partitioning_mode = "CPX"
                 print(f"update_metrics_cache: Confirmed CPX mode from ROCm SMI data (64 devices). Re-adjusting totals in parsed_gpu_metrics if needed.", flush=True)
                 for pid_str_cpx_check in parsed_gpu_metrics.keys():
                     if parsed_gpu_metrics[pid_str_cpx_check].get('partition_type') != "cpx" or parsed_gpu_metrics[pid_str_cpx_check].get('vram_total_gb') != cpx_default_vram_per_partition :
                        parsed_gpu_metrics[pid_str_cpx_check]['vram_total_gb'] = cpx_default_vram_per_partition
                        parsed_gpu_metrics[pid_str_cpx_check]['total_mb'] = cpx_default_vram_per_partition * 1024.0
                        parsed_gpu_metrics[pid_str_cpx_check]['partition_type'] = "cpx"
                        print(f"  CPX Re-check: Partition {pid_str_cpx_check} total VRAM set to {cpx_default_vram_per_partition}GB and type to CPX.", flush=True)


            rocm_override_env_value = os.getenv('ROCM_SMI_VRAM_OVERRIDE', 'false')
            use_rocm_vram_override = rocm_override_env_value.lower() == 'true'
            
            overrides_applied_count = 0
            util_updates_count = 0

            # Ensure parsed_gpu_metrics has entries for all devices seen by rocm-smi, if they were missed by amd-smi
            for device_id_rocm in rocm_smi_raw_metrics.keys():
                if device_id_rocm not in parsed_gpu_metrics:
                    default_total_for_missing = cpx_default_vram_per_partition if partitioning_mode == "CPX" else GPU_VRAM_DEFAULTS['default']
                    parsed_gpu_metrics[device_id_rocm] = {
                        'partition_id': device_id_rocm,
                        '_source': 'rocm_smi_discovered',
                        'vram_used_gb': 0.0,
                        'vram_total_gb': default_total_for_missing,
                        'processes': [], 'process_vram_gb': 0.0, 'utilization': 0,
                        'partition_type': partitioning_mode.lower() if partitioning_mode != "unknown" else "unknown_rocm",
                        'used_mb': 0.0, 'total_mb': default_total_for_missing * 1024.0
                    }
                    print(f"  Added placeholder in parsed_gpu_metrics for Device ID {device_id_rocm} seen by rocm-smi but not amd-smi.", flush=True)


            for partition_id_str, metrics_from_rocm in rocm_smi_raw_metrics.items():
                if partition_id_str in parsed_gpu_metrics:
                    base_metrics_val = parsed_gpu_metrics[partition_id_str]
                    
                    # Always update GPU utilization from rocm-smi
                    new_gpu_util = metrics_from_rocm['gpu_utilization']
                    old_gpu_util = base_metrics_val.get('utilization', 0)
                    base_metrics_val['utilization'] = new_gpu_util
                    if old_gpu_util != new_gpu_util:
                         util_updates_count +=1
                         base_metrics_val['_source'] += '+rocm_util'

                    # Important change: respect vLLM VRAM data from the mapper script
                    # If this partition has vLLM data from the mapper, use that even if ROCM_SMI_VRAM_OVERRIDE is true
                    vllm_vram_from_mapper = vllm_vram_by_true_partition.get(partition_id_str, 0.0)
                    
                    if vllm_vram_from_mapper > 0.0:
                        # Use the mapper's vLLM data if available, overriding rocm-smi
                        if base_metrics_val.get('vram_used_gb', 0.0) != vllm_vram_from_mapper:
                            print(f"Partition {partition_id_str}: Using vLLM VRAM from mapper ({vllm_vram_from_mapper:.2f}GB) instead of current value ({base_metrics_val.get('vram_used_gb', 0.0):.2f}GB)", flush=True)
                            base_metrics_val['vram_used_gb'] = vllm_vram_from_mapper
                            base_metrics_val['used_mb'] = vllm_vram_from_mapper * 1024.0
                            base_metrics_val['_source'] += '+mapper_vllm_override'
                    elif use_rocm_vram_override:
                        # Calculate VRAM used based on rocm-smi percentage only if no vLLM data
                        current_partition_total_vram_gb = base_metrics_val.get('vram_total_gb', 0.0)
                        # Ensure total_vram_gb is sensible for percentage calculation
                        if current_partition_total_vram_gb == 0.0:
                            if partitioning_mode == "CPX":
                                current_partition_total_vram_gb = cpx_default_vram_per_partition
                            else: # SPX or Unknown, use a broad default if total is still 0
                                current_partition_total_vram_gb = GPU_VRAM_DEFAULTS.get(detected_gpu_type, GPU_VRAM_DEFAULTS['default'])
                            print(f"Partition {partition_id_str}: Warning - vram_total_gb was 0 for rocm-smi % calc. Using {current_partition_total_vram_gb}GB.", flush=True)
                        
                        new_vram_used_gb_from_rocm_pct = (metrics_from_rocm['vram_percentage'] / 100.0) * current_partition_total_vram_gb
                        old_vram_used_gb = base_metrics_val.get('vram_used_gb', 0.0)
                        
                        base_metrics_val['vram_used_gb'] = round(new_vram_used_gb_from_rocm_pct, 2)
                        base_metrics_val['used_mb'] = base_metrics_val['vram_used_gb'] * 1024.0 # Update mb too
                        base_metrics_val['_source'] += '+rocm_vram_override'
                        overrides_applied_count += 1
                        print(f"Partition {partition_id_str}: ROCM_SMI_VRAM_OVERRIDE true. VRAM Used: {old_vram_used_gb:.2f}GB -> {new_vram_used_gb_from_rocm_pct:.2f}GB. GPU Util: {old_gpu_util}% -> {new_gpu_util}%. Source: {base_metrics_val['_source']}", flush=True)
                    else:
                        # ROCM_SMI_VRAM_OVERRIDE is false, VRAM from amd-smi ('process_vram_gb') is kept.
                        # GPU util was already updated.
                        if old_gpu_util != new_gpu_util: # only print if util changed
                            print(f"Partition {partition_id_str}: ROCM_SMI_VRAM_OVERRIDE false. VRAM from amd-smi used. GPU Util updated: {old_gpu_util}% -> {new_gpu_util}%. Source: {base_metrics_val['_source']}", flush=True)

            print(f"Finished processing ROCm SMI data. VRAM overrides: {overrides_applied_count}, Util updates: {util_updates_count}.", flush=True)
        else:
            print("update_metrics_cache: No ROCm SMI metrics data found. GPU utilization might be missing or stale.", flush=True)
        # ROCM_SMI_OVERRIDE logic ends here

        # --- Start of partitions_output generation ---
        total_vram_used_gb = 0.0
        total_vram_available_gb = 0.0 # This will be refined, especially for CPX

        if partition_config:
            print(f"update_metrics_cache: Processing {len(partition_config)} partitions from configuration (using potentially overridden parsed_gpu_metrics)...", flush=True)
            
            config_sample = {}
            for i, (pid, cfg) in enumerate(partition_config.items()):
                if i < 5 or (i >= 35 and i <= 45) or (i >= 55 and i <= 63):
                    config_sample[pid] = cfg
            print(f"update_metrics_cache: Partition config sample (for partitions_output loop): {json.dumps(config_sample)}", flush=True)
            
            for partition_id_str, config_entry in partition_config.items():
                name = f"GPU {partition_id_str}"
                color = None
                is_free_partition = False
                if isinstance(config_entry, dict):
                    name = config_entry.get('name', name)
                    color = config_entry.get('bg_color')
                elif isinstance(config_entry, str): name = config_entry
                
                is_free_partition = "Free Partition" in name
                
                partition_vram_used_gb = 0.0
                partition_vram_total_gb = 0.0 
                partition_gpu_utilization = 0.0
                current_metric_source = "config_default"

                partition_metrics = parsed_gpu_metrics.get(partition_id_str) 
                
                if partition_metrics:
                    current_metric_source = partition_metrics.get('_source', 'unknown_in_parsed')
                    partition_vram_used_gb = partition_metrics.get('vram_used_gb', 0.0)
                    partition_vram_total_gb = partition_metrics.get('vram_total_gb', 0.0)
                    partition_gpu_utilization = partition_metrics.get('utilization', 0.0)
                    
                    # Final check for total VRAM in CPX mode
                    if partitioning_mode == "CPX":
                        if partition_vram_total_gb != cpx_default_vram_per_partition:
                             print(f"  Partition {partition_id_str} ('{name}'): Final CPX check, correcting total VRAM from {partition_vram_total_gb:.2f} to {cpx_default_vram_per_partition:.2f} GB.", flush=True)
                             partition_vram_total_gb = cpx_default_vram_per_partition
                    elif partition_vram_total_gb == 0: # If total is still 0 for non-CPX (or CPX missed somehow)
                        # Fallback if total is somehow still zero
                        if detected_gpu_type and detected_gpu_type in GPU_VRAM_DEFAULTS:
                            partition_vram_total_gb = GPU_VRAM_DEFAULTS[detected_gpu_type] # SPX total for that type, or CPX part if type known
                        else:
                            partition_vram_total_gb = GPU_VRAM_DEFAULTS['default'] # Generic default
                        current_metric_source += "_total_final_fallback"
                        print(f"  Partition {partition_id_str} ('{name}'): Total VRAM was 0, applied final fallback to {partition_vram_total_gb:.2f}GB.", flush=True)


                    print(f"  Partition {partition_id_str} ('{name}'): Data from parsed_gpu_metrics (source: {current_metric_source}) -> VRAM Used: {partition_vram_used_gb:.2f} GB, VRAM Total: {partition_vram_total_gb:.2f} GB, Util: {partition_gpu_utilization}%", flush=True)

                else: 
                    print(f"  Warning: No metrics in parsed_gpu_metrics for configured partition {partition_id_str} (Name: {name}). Using defaults.", flush=True)
                    current_metric_source = "no_parsed_metrics"

                if is_free_partition and partition_vram_used_gb > 0.0:
                    print(f"  Zeroing VRAM usage for Free Partition {partition_id_str} (was {partition_vram_used_gb:.2f} GB, source: {current_metric_source})", flush=True)
                    partition_vram_used_gb = 0.0
                
                if (not is_free_partition and
                    "Partition" not in name and
                    partition_vram_used_gb == 0.0): # MODIFIED: Removed 'and current_metric_source != 'rocm_smi_override''
                    print(f"  Applying 0.1GB placeholder for model partition {partition_id_str} (Name: {name}, VRAM was 0.0, Source: {current_metric_source})", flush=True)
                    partition_vram_used_gb = 0.1
                
                total_vram_used_gb += partition_vram_used_gb
                total_vram_available_gb += partition_vram_total_gb 

                partitions_output.append({
                    "id": partition_id_str, "name": name,
                    "vram_used": round(partition_vram_used_gb, 1), 
                    "vram_total": round(partition_vram_total_gb, 1), # This will be corrected by CPX logic later if needed
                    "gpu_utilization": round(float(partition_gpu_utilization), 1), # Ensure float before round
                    "bg_color": color,
                    "_debug_source": current_metric_source # Add source for easier debugging in output
                })
        else: 
            print("update_metrics_cache: No partition configuration. Using detected metrics directly from parsed_gpu_metrics...", flush=True)
            if partitioning_mode == "UNKNOWN" and amd_smi_metrics:
                if len(amd_smi_metrics) >= 64 : partitioning_mode = "CPX"
                elif len(amd_smi_metrics) > 0 and len(amd_smi_metrics) <= 8 : partitioning_mode = "SPX" # e.g. 8 physical GPUs
            print(f"update_metrics_cache: Inferred/Current partitioning_mode for no-config path: {partitioning_mode}", flush=True)

            for pid, metrics_from_parsed in parsed_gpu_metrics.items():
                default_total_gb_no_config = 0.0
                display_name = f"{'CPX Part ' if partitioning_mode == 'CPX' else 'GPU '}{pid}"
                
                if partitioning_mode == "CPX":
                    default_total_gb_no_config = cpx_default_vram_per_partition
                elif partitioning_mode == "SPX": 
                    # Use amd-smi total if available, otherwise specific SPX defaults
                    default_total_gb_no_config = metrics_from_parsed.get('vram_total_gb', 0.0)
                    if default_total_gb_no_config == 0.0 : # If amd-smi didn't provide or was missing for this pid
                        default_total_gb_no_config = GPU_VRAM_DEFAULTS.get(detected_gpu_type, 256.0 if detected_gpu_type == 'MI325X' else 192.0)
                else: # Unknown
                    default_total_gb_no_config = metrics_from_parsed.get('vram_total_gb', GPU_VRAM_DEFAULTS['default'])
                
                vram_used_gb_no_config = metrics_from_parsed.get('vram_used_gb', 0.0)
                vram_total_gb_no_config = metrics_from_parsed.get('vram_total_gb', 0.0)
                if vram_total_gb_no_config == 0: vram_total_gb_no_config = default_total_gb_no_config

                gpu_util_no_config = metrics_from_parsed.get('utilization', 0.0)
                source_no_config = metrics_from_parsed.get('_source', 'unknown_no_conf')
                processes_no_config = metrics_from_parsed.get('processes', [])


                if (pid != "0" and # heuristic: not the system partition
                    vram_used_gb_no_config == 0.0 and 
                    len(processes_no_config) == 0 and # no processes reported by amd-smi
                    (not use_rocm_vram_override or (rocm_smi_raw_metrics and rocm_smi_raw_metrics.get(pid,{}).get('vram_percentage',1) == 0) ) # and rocm-smi also shows 0% or override is off
                    and (not PSUTIL_AVAILABLE or metrics_from_parsed.get('vllm_specific_process_vram_gb', 1.0) == 0.0) # and if psutil is on, no vllm specific vram
                    ):
                    # If it's not partition 0, has 0 VRAM used from amd-smi processes, no processes,
                    # and (either rocm_vram_override is OFF OR rocm-smi also shows 0% VRAM)
                    # and (if psutil is available, no vLLM specific VRAM was found for this GPU by psutil)
                    # then apply placeholder. This avoids placeholder if rocm_override IS on and rocm reports usage,
                    # or if psutil is on and found vLLM specific VRAM (even if total process_vram was initially 0 which is unlikely).
                    print(f"  No-config: Applying 0.1GB placeholder for partition {pid} (VRAM was 0.0, Source: {source_no_config})", flush=True)
                    vram_used_gb_no_config = 0.1

                total_vram_used_gb += vram_used_gb_no_config
                total_vram_available_gb += vram_total_gb_no_config
                partitions_output.append({
                    "id": pid, "name": display_name,
                    "vram_used": round(vram_used_gb_no_config, 1), 
                    "vram_total": round(vram_total_gb_no_config, 1),
                    "gpu_utilization": round(float(gpu_util_no_config),1), # Ensure float before round
                    "bg_color": color,
                    "_debug_source": source_no_config
                })

        if partitioning_mode == "CPX": 
            print(f"update_metrics_cache: CPX mode post-processing for partitions_output. Verifying 64 partitions and {cpx_default_vram_per_partition}GB total for each.", flush=True)
            expected_partition_ids = {str(i) for i in range(64)}
            current_partition_ids_in_output = {p['id'] for p in partitions_output}
            
            # Add missing partitions if any (should ideally not happen if partition_config has all 64)
            for i in range(64):
                pid_str = str(i)
                if pid_str not in current_partition_ids_in_output:
                    config_name_cpx_missing = f"GPU {pid_str}"
                    debug_source_cpx_missing = "cpx_added_missing"
                    if partition_config and partition_config.get(pid_str):
                        entry = partition_config[pid_str]
                        if isinstance(entry, dict): config_name_cpx_missing = entry.get('name', config_name_cpx_missing)
                        elif isinstance(entry, str): config_name_cpx_missing = entry
                    
                    partitions_output.append({
                        "id": pid_str, "name": config_name_cpx_missing if config_name_cpx_missing != f"GPU {pid_str}" else "Free Partition CPX", 
                        "vram_used": 0.0, "vram_total": cpx_default_vram_per_partition, 
                        "gpu_utilization": 0.0, "bg_color": "#FFFFFF",
                        "_debug_source": debug_source_cpx_missing
                    })
                    print(f"update_metrics_cache: Added missing CPX partition {pid_str} to partitions_output with {cpx_default_vram_per_partition}GB total.", flush=True)
            
            temp_total_vram_available_gb_cpx = 0 # Recalculate available for CPX based on fixed partition sizes
            temp_total_vram_used_gb_cpx = 0 # Recalculate used VRAM based on final values in partitions_output

            for p_idx, p_entry in enumerate(partitions_output):
                # Ensure we only consider the 64 CPX partitions for this specific CPX adjustment block
                if p_entry['id'] in expected_partition_ids:
                    if p_entry.get('vram_total') != cpx_default_vram_per_partition:
                        print(f"update_metrics_cache: CPX Correction: Partition {p_entry['id']} total VRAM from {p_entry.get('vram_total')} to {cpx_default_vram_per_partition} GB.", flush=True)
                        partitions_output[p_idx]['vram_total'] = cpx_default_vram_per_partition # Directly modify the list item
                    
                    if "Free Partition" in p_entry.get('name', ''): # Second pass for Free Partitions in CPX context
                        if p_entry.get('vram_used', 0.0) > 0.0:
                            print(f"update_metrics_cache: CPX Correction: Zeroing VRAM for Free Partition {p_entry['id']} (was {p_entry.get('vram_used', 0.0)} GB, source: {p_entry.get('_debug_source')})", flush=True)
                            partitions_output[p_idx]['vram_used'] = 0.0 # Directly modify
                    
                    temp_total_vram_available_gb_cpx += partitions_output[p_idx]['vram_total'] # Use the (potentially corrected) total
                    temp_total_vram_used_gb_cpx += partitions_output[p_idx]['vram_used'] # Use the (potentially corrected) used
            
            # Update overall totals if they were based on pre-CPX-correction values
            if total_vram_available_gb != temp_total_vram_available_gb_cpx and temp_total_vram_available_gb_cpx > 0:
                 print(f"update_metrics_cache: CPX total_vram_available_gb corrected from {total_vram_available_gb} to {temp_total_vram_available_gb_cpx}.", flush=True)
                 total_vram_available_gb = temp_total_vram_available_gb_cpx
            
            if total_vram_used_gb != temp_total_vram_used_gb_cpx: # Update total used based on CPX pass
                 print(f"update_metrics_cache: CPX total_vram_used_gb corrected from {total_vram_used_gb} to {temp_total_vram_used_gb_cpx}.", flush=True)
                 total_vram_used_gb = temp_total_vram_used_gb_cpx

            expected_cpx_total_overall_final = 64 * cpx_default_vram_per_partition
            if total_vram_available_gb != expected_cpx_total_overall_final:
                print(f"update_metrics_cache: Final CPX total_vram_available_gb sanity check. Correcting from {total_vram_available_gb} to {expected_cpx_total_overall_final} GB.", flush=True)
                total_vram_available_gb = expected_cpx_total_overall_final
        # --- End of moved block ---
        
        partitions_output.sort(key=lambda p: int(p['id']) if p['id'].isdigit() else float('inf'))
        
        new_collected_data = {
            "status": "ok", # Mark as successfully collected
            "timestamp": time.time(),
            "partitions": partitions_output,
            "summary": {
                "total_vram_used": round(total_vram_used_gb, 1),
                "total_vram_available": round(total_vram_available_gb, 1),
                "total_vram_free": round(max(0, total_vram_available_gb - total_vram_used_gb), 1) # Ensure free is not negative
            },
            "services": service_metrics
        }
        collected_data_successfully = True
        print("update_metrics_cache: Successfully prepared new metrics data.", flush=True)

    except Exception as e:
        print(f"update_metrics_cache: CRITICAL ERROR during metrics collection: {e}", flush=True)
        # Do not update METRICS_CACHE if a critical error occurred in the collection logic.
        # The existing (potentially stale) cache will be preserved if it was 'ok'.
        # If it was 'initializing', mark it as a collection failure.
        with cache_lock:
            if METRICS_CACHE.get("status") == "initializing":
                METRICS_CACHE["status"] = "initialization_collection_failed"
                METRICS_CACHE["error_details"] = f"Critical error during initial collection: {str(e)}"
                METRICS_CACHE["timestamp"] = time.time()
        return # Exit the function

    # After all data collection attempts, decide if cache should be updated
    if collected_data_successfully:
        has_prometheus_gpu_data = bool(parsed_gpu_metrics) # True if any Prometheus GPU partitions were parsed
        has_amd_smi_gpu_data = bool(amd_smi_metrics)    # True if AMD SMI returned any GPU data
        has_rocm_smi_gpu_data = bool(rocm_smi_raw_metrics) # True if ROCm SMI returned any data

        if not has_prometheus_gpu_data and not has_amd_smi_gpu_data and not has_rocm_smi_gpu_data: # Check all sources
            print("update_metrics_cache: No actual GPU metrics obtained from Prometheus, AMD SMI, or ROCm SMI.", flush=True)
            with cache_lock:
                if METRICS_CACHE.get("status") == "ok":
                    print("update_metrics_cache: Preserving previous 'ok' cache as current collection lacks GPU data.", flush=True)
                    # Optionally, update a 'last_attempt_failed_timestamp' or similar in METRICS_CACHE
                    METRICS_CACHE["last_successful_update"] = METRICS_CACHE.get("timestamp", time.time()) # Preserve old timestamp
                    METRICS_CACHE["timestamp"] = time.time() # Mark time of this failed attempt
                    METRICS_CACHE["status_detail"] = "Failed to fetch new GPU data, using stale."
                    # Don't overwrite METRICS_CACHE with new_collected_data if it's "empty"
                    return # Skip updating the main cache with potentially empty/default data
                else: # Cache was 'initializing' or some error state, update with current (failed) collection attempt
                    print("update_metrics_cache: Cache was not 'ok'. Updating with current collection (which lacks GPU data).", flush=True)
                    # This will result in METRICS_CACHE being new_collected_data, which might show all defaults
                    # but its 'status' will be 'ok' from new_collected_data.
                    # We should mark new_collected_data with a different status if it's not good.
                    new_collected_data["status"] = "ok_no_gpu_data" # A more specific status
                    new_collected_data["status_detail"] = "Data collected, but no specific GPU metrics from Prometheus/AMD-SMI/ROCm-SMI were available in this cycle."

        with cache_lock:
            METRICS_CACHE = new_collected_data # new_collected_data has 'status: "ok"' or "ok_no_gpu_data"
            print(f"update_metrics_cache: METRICS_CACHE updated. New status: {METRICS_CACHE.get('status')}.", flush=True)
    else: # collected_data_successfully is False (shouldn't happen if critical error above returns)
        print("update_metrics_cache: Metrics collection flagged as unsuccessful (collected_data_successfully=False), METRICS_CACHE not updated.", flush=True)


def background_metrics_updater():
    """
    Background thread function that periodically updates the metrics cache.
    
    This runs continuously, sleeping for UPDATE_INTERVAL seconds between updates.
    """
    print("Background metrics updater thread started", flush=True)
    loop_count = 0
    
    while True:
        loop_count += 1
        print(f"Background metrics updater: Loop iteration {loop_count} starting", flush=True)
        
        try:
            update_metrics_cache()
        except Exception as e:
            print(f"Error in background metrics updater loop: {e}", flush=True)
            
        print(f"Background updater sleeping for {UPDATE_INTERVAL} seconds...", flush=True)
        time.sleep(UPDATE_INTERVAL)

# --- Flask Routes ---    
@app.route('/metrics')
def get_metrics():
    """
    API endpoint that returns the current metrics cache.
    
    Returns:
        JSON containing GPU metrics and status information
    """
    with cache_lock:
        # Return a deep copy to prevent modification of the cache by callers
        current_cache_status = METRICS_CACHE.get("status")
        if current_cache_status == "ok" or current_cache_status == "ok_no_gpu_data":
            return jsonify(copy.deepcopy(METRICS_CACHE))
        elif current_cache_status == "initializing":
            return jsonify({
                "error": "Metrics data is initializing. Please try again shortly.",
                "status": current_cache_status,
                "details": METRICS_CACHE # Send current cache for debugging
            }), 503
        else: # Covers other error states
            return jsonify({
                "error": "Metrics data is not available or an error occurred during collection.",
                "status": current_cache_status,
                "cache_content": METRICS_CACHE # Send current cache for debugging
            }), 503

@app.route('/api/config')
def get_partition_config():
    """
    API endpoint that returns the current partition configuration.
    
    Returns:
        JSON containing partition configuration or error
    """
    config = load_partition_config()
    if not config:
        return jsonify({"error": "Partition configuration could not be loaded."}), 500
    return jsonify(config)

# --- Application Initialization ---
def initialize_app():
    """Initialize the metrics cache and start the background updater thread."""
    global METRICS_CACHE
    
    with cache_lock:
        METRICS_CACHE = {
            "status": "initializing",
            "timestamp": time.time(),
            "partitions": [],
            "summary": {"message": "Initial data collection pending..."},
            "services": {}
        }
    print("Initial METRICS_CACHE state set to 'initializing'", flush=True)

    # Populate the cache for the first time
    try:
        update_metrics_cache()
        with cache_lock:
            if METRICS_CACHE.get("status") == "ok" or METRICS_CACHE.get("status") == "ok_no_gpu_data":
                print("Initial metrics cache populated successfully", flush=True)
            else:
                print(f"Initial metrics collection may have failed. Cache status: {METRICS_CACHE.get('status')}", flush=True)
    except Exception as e:
        print(f"CRITICAL Error during initial metrics collection: {e}", flush=True)
        with cache_lock:
            METRICS_CACHE["status"] = "critical_initialization_failure"
            METRICS_CACHE["error_message"] = str(e)

    # Start background updater thread
    updater_thread = threading.Thread(target=background_metrics_updater, daemon=True)
    updater_thread.start()
    print("Background metrics updater thread started", flush=True)
    
    return app

# Initialize the application
app = initialize_app()

if __name__ == "__main__":
    print("Starting Flask application on port 5001", flush=True)
    app.run(host='0.0.0.0', port=5001, threaded=True) 
