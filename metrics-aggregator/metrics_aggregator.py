#!/usr/bin/env python3
"""
Simplified AMD Instinct GPU Metrics Aggregator

This service collects metrics from AMD Instinct GPUs using the gpu_process_mapper.py script
as its primary source, and exposes them via a REST API for monitoring dashboards.

Key features:
- Direct use of gpu_process_mapper.py output for reliable VRAM attribution
- Simple configuration through partition_config.json
- Optional GPU utilization via rocm-smi
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
import subprocess
import urllib.request
import urllib.error
from flask import Flask, jsonify

app = Flask(__name__)

# --- Global Cache and Lock ---
METRICS_CACHE = {}
cache_lock = threading.Lock()
UPDATE_INTERVAL = 15  # seconds

# --- Configuration ---
CONFIG_FILE_PATH = '/app/config/partition_config.json'
MAPPER_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gpu_process_mapper.py')
USE_ROCM_SMI_FOR_UTILIZATION = True  # Set to True to use rocm-smi for GPU utilization

# Default VRAM size in GB for each partition
DEFAULT_VRAM_TOTAL_GB = 32.0  # Default for MI325X in CPX mode

# --- Helper Functions ---
def load_partition_config():
    """
    Load the GPU partition configuration from a JSON file.
    Returns an empty dict if the file cannot be found or parsed.
    """
    try:
        config_paths = [
            CONFIG_FILE_PATH,
            'config/partition_config.json',  # For local development
        ]
        
        for path in config_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return json.load(f)
        
        print(f"Error: Partition config file not found in any of {config_paths}", flush=True)
        return {}
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading partition config: {e}", flush=True)
        return {}

def execute_gpu_mapper_script():
    """
    Executes the gpu_process_mapper.py script and returns its parsed JSON output.
    Returns None if script execution or JSON parsing fails.
    """
    try:
        # Get absolute path to script for better error reporting
        script_path = os.path.abspath(MAPPER_SCRIPT_PATH)
        print(f"Executing GPU mapper script: {script_path}", flush=True)
        
        # Check if script exists
        if not os.path.exists(script_path):
            print(f"Error: GPU mapper script does not exist at {script_path}", flush=True)
            
            # Try to find the script in current directory
            if os.path.exists('./gpu_process_mapper.py'):
                script_path = './gpu_process_mapper.py'
                print(f"Found script at {script_path} instead", flush=True)
            else:
                print("Could not find gpu_process_mapper.py in current directory either", flush=True)
                return None
        
        # Try with sudo first since we know this works outside the container
        print(f"Running script with sudo first: {script_path}", flush=True)
        result = subprocess.run(['sudo', script_path], capture_output=True, text=True, check=False, timeout=60)
        
        # If that fails, try without sudo
        if result.returncode != 0:
            print(f"GPU Mapper script with sudo failed with return code {result.returncode}, trying without sudo", flush=True)
            print(f"Script stdout: {result.stdout[:200]}...", flush=True)
            print(f"Script stderr: {result.stderr[:200]}...", flush=True)
            result = subprocess.run([script_path], capture_output=True, text=True, check=False, timeout=60)

        if result.returncode != 0:
            print(f"GPU Mapper Script failed (Return Code {result.returncode}):", flush=True)
            print(f"  Stdout: {result.stdout.strip()}", flush=True)
            print(f"  Stderr: {result.stderr.strip()}", flush=True)
            return None

        try:
            parsed_data = json.loads(result.stdout)
            print(f"Successfully parsed JSON from GPU mapper script", flush=True)
            # Debug print the VRAM data to ensure it's being captured correctly
            vram_data = parsed_data.get('vllm_vram_by_true_partition', {})
            if vram_data:
                print(f"VRAM data found for partitions: {list(vram_data.keys())}", flush=True)
                for part_id, vram in sorted(vram_data.items(), key=lambda x: int(x[0])):
                    print(f"  Partition {part_id}: {vram} GB", flush=True)
            else:
                print("WARNING: No VRAM data found in mapper output", flush=True)
            return parsed_data
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from GPU mapper script output: {e}", flush=True)
            print(f"First 200 chars of output: {result.stdout[:200]}", flush=True)
            return None
        
    except subprocess.TimeoutExpired:
        print("Timeout executing GPU mapper script.", flush=True)
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from GPU mapper script: {e}", flush=True)
        return None
    except FileNotFoundError:
        print(f"Error: GPU mapper script '{MAPPER_SCRIPT_PATH}' not found.", flush=True)
        return None
    except Exception as e:
        print(f"An unexpected error occurred while executing GPU mapper script: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None

def get_rocm_smi_utilization():
    """
    Executes rocm-smi command to get GPU utilization data.
    Returns a dict mapping device ID to utilization percentage.
    """
    if not USE_ROCM_SMI_FOR_UTILIZATION:
        return {}
        
    try:
        command = ['rocm-smi']
        print(f"Executing rocm-smi for GPU utilization", flush=True)
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
        
        if result.returncode != 0:
            print(f"rocm-smi command failed, trying with sudo", flush=True)
            result = subprocess.run(['sudo', 'rocm-smi'], capture_output=True, text=True, check=False, timeout=30)
            
        if result.returncode != 0:
            print(f"rocm-smi command failed even with sudo", flush=True)
            return {}
            
        output = result.stdout
        utilization_by_gpu = {}
        
        # Pattern to match the rocm-smi header line
        header_pattern = re.compile(r"^Device\s+Node\s+IDs\s+Temp\s+Power\s+Partitions\s+SCLK\s+MCLK\s+Fan\s+Perf\s+PwrCap\s+VRAM%\s+GPU%$")
        
        lines = output.splitlines()
        data_started = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if header_pattern.match(line):
                data_started = True
                continue
                
            if "End of ROCm SMI Log" in line:
                break
                
            if data_started and line[0].isdigit():
                parts = line.split()
                if len(parts) >= 12:
                    try:
                        device_id = parts[0]
                        gpu_util = parts[-1].replace('%', '')
                        
                        if gpu_util.isdigit():
                            utilization_by_gpu[device_id] = int(gpu_util)
                    except (ValueError, IndexError):
                        continue
        
        return utilization_by_gpu
    
    except Exception as e:
        print(f"Error getting GPU utilization from rocm-smi: {e}", flush=True)
        return {}

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

def parse_prometheus_metric(text_data, metric_name_to_find):
    """
    Parse a specific metric from Prometheus text format.
    
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

def get_service_metrics(partition_config):
    """
    Collect metrics from vLLM service instances defined in the partition configuration.
    """
    services_data = {'vllm_aggregate': {
        'service_type': 'vllm_aggregate',
        'status': 'ok',
        'requests_running': 0,
        'requests_waiting': 0,
        'kv_cache_usage': 0.0,
        'instance_count': 0,
        'fetch_errors': 0
    }}
    
    # Count and get metrics from vLLM instances
    vllm_total_running = 0
    vllm_total_waiting = 0
    vllm_kv_cache_sum_perc = 0.0
    vllm_instance_count = 0
    vllm_fetch_errors = 0
    
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
    
    # Update vLLM aggregate data
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

# --- Core Metrics Aggregation Logic ---
def update_metrics_cache():
    """
    Core function that collects and aggregates metrics from:
    1. Partition configuration (defines partitions and their assignments)
    2. gpu_process_mapper.py (primary source for VRAM usage)
    3. rocm-smi (optional source for GPU utilization)
    """
    global METRICS_CACHE
    print("update_metrics_cache: Updating metrics cache...", flush=True)
    
    try:
        # Load partition configuration
        partition_config = load_partition_config()
        if not partition_config:
            print("Warning: No partition configuration loaded. Continuing with empty config.", flush=True)
            # We'll continue with an empty config rather than returning, 
            # so we can still get data from the mapper script
        
        # Get VRAM data from gpu_process_mapper.py
        mapper_data = execute_gpu_mapper_script()
        if not mapper_data:
            print("Warning: No data from GPU mapper script. Using placeholder values.", flush=True)
            mapper_data = {
                'vllm_vram_by_true_partition': {},
                'true_partition_to_smi_gpu_id': {},
                'all_smi_gpu_details': {}
            }
        
        # Extract the actual VRAM usage data by partition
        vram_by_partition = mapper_data.get('vllm_vram_by_true_partition', {})
        gpu_to_partition = mapper_data.get('true_partition_to_smi_gpu_id', {})
        smi_gpu_details = mapper_data.get('all_smi_gpu_details', {})
        
        print(f"Found VRAM data for {len(vram_by_partition)} partitions", flush=True)
        
        # Get GPU utilization from rocm-smi
        gpu_utilization = get_rocm_smi_utilization()
        
        # Build partitions output
        partitions_output = []
        total_vram_used_gb = 0.0
        total_vram_available_gb = 0.0
        
        # Process all partitions from config
        all_partition_ids = set(partition_config.keys())
        
        # Ensure all partitions from mapper are included
        all_partition_ids.update(vram_by_partition.keys())
        
        # If we still have no partitions, use the ones from the config file
        if not all_partition_ids and partition_config:
            all_partition_ids = set(partition_config.keys())
            print(f"Using {len(all_partition_ids)} partition IDs from config file only", flush=True)
        
        # Sort partition IDs numerically
        sorted_partition_ids = sorted(all_partition_ids, key=lambda x: int(x) if x.isdigit() else float('inf'))
        print(f"Processing {len(sorted_partition_ids)} partitions", flush=True)
        
        for partition_id in sorted_partition_ids:
            # Get configuration for this partition
            config = partition_config.get(partition_id, {})
            
            # Determine partition name and color
            if isinstance(config, dict):
                name = config.get('name', f"GPU {partition_id}")
                color = config.get('bg_color', '#FFFFFF')
            elif isinstance(config, str):
                name = config
                color = '#FFFFFF'
            else:
                name = f"GPU {partition_id}"
                color = '#FFFFFF'
                
            # Determine if this is a free partition (not allocated to any model)
            is_free = "Free Partition" in name
            
            # Get VRAM usage for this partition from the mapper data
            # This is the key change - directly use the actual VRAM values from mapper
            vram_used = vram_by_partition.get(partition_id, 0.0)
            print(f"Partition {partition_id} ({name}): VRAM usage from mapper: {vram_used} GB", flush=True)
            
            # Free partitions should show 0 VRAM usage
            if is_free:
                vram_used = 0.0
            
            # Apply 0.1GB placeholder for model partitions with 0 VRAM
            # but only if they're not "Free Partition" and have no actual usage
            if not is_free and vram_used == 0.0:
                print(f"Applying 0.1GB placeholder for partition {partition_id} ({name})", flush=True)
                vram_used = 0.1
                
            # Default VRAM total is 32GB per partition
            vram_total = DEFAULT_VRAM_TOTAL_GB
            
            # Get GPU utilization - try to map from physical GPU to partition
            gpu_util = 0.0
            smi_gpu_id = gpu_to_partition.get(partition_id)
            
            # First check if we have utilization from rocm-smi for the SMI GPU ID
            if smi_gpu_id and smi_gpu_id in gpu_utilization:
                gpu_util = gpu_utilization[smi_gpu_id]
            # Then check if we have utilization directly for the partition ID
            elif partition_id in gpu_utilization:
                gpu_util = gpu_utilization[partition_id]
                
            # Add to totals
            total_vram_used_gb += vram_used
            total_vram_available_gb += vram_total
            
            # Add partition to output
            partitions_output.append({
                "id": partition_id,
                "name": name,
                "vram_used": round(vram_used, 1),
                "vram_total": round(vram_total, 1),
                "gpu_utilization": round(gpu_util, 1),
                "bg_color": color
            })
        
        # Get service metrics (including vLLM metrics)
        service_metrics = get_service_metrics(partition_config)
        
        # Create final metrics data
        new_metrics = {
            "status": "ok",
            "timestamp": time.time(),
            "partitions": partitions_output,
            "summary": {
                "total_vram_used": round(total_vram_used_gb, 1),
                "total_vram_available": round(total_vram_available_gb, 1),
                "total_vram_free": round(max(0, total_vram_available_gb - total_vram_used_gb), 1)
            },
            "services": service_metrics
        }
        
        # Update the cache with new metrics
        with cache_lock:
            METRICS_CACHE = new_metrics
            
        print(f"update_metrics_cache: Successfully updated metrics. Partitions: {len(partitions_output)}, Total VRAM: {total_vram_available_gb}GB, Used: {total_vram_used_gb}GB", flush=True)
        
    except Exception as e:
        print(f"update_metrics_cache: Error updating metrics: {e}", flush=True)
        import traceback
        traceback.print_exc()

def background_metrics_updater():
    """
    Background thread function that periodically updates the metrics cache.
    """
    print("Background metrics updater thread started", flush=True)
    
    while True:
        try:
            update_metrics_cache()
        except Exception as e:
            print(f"Error in background metrics updater: {e}", flush=True)
            
        print(f"Background updater sleeping for {UPDATE_INTERVAL} seconds...", flush=True)
        time.sleep(UPDATE_INTERVAL)

# --- Flask Routes ---    
@app.route('/metrics')
def get_metrics():
    """
    API endpoint that returns the current metrics cache.
    """
    with cache_lock:
        current_cache = copy.deepcopy(METRICS_CACHE)
        
    if not current_cache:
        return jsonify({
            "error": "Metrics data is not available yet.",
            "status": "initializing"
        }), 503
        
    return jsonify(current_cache)

@app.route('/api/config')
def get_partition_config():
    """
    API endpoint that returns the current partition configuration.
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
    
    # Populate the cache for the first time
    try:
        update_metrics_cache()
    except Exception as e:
        print(f"Error during initial metrics collection: {e}", flush=True)

    # Start background updater thread
    updater_thread = threading.Thread(target=background_metrics_updater, daemon=True)
    updater_thread.start()
    
    return app

# Initialize the application
app = initialize_app()

if __name__ == "__main__":
    print("Starting Flask application on port 5001", flush=True)
    # Explicitly set threaded=True and configure Flask for production use
    app.run(host='0.0.0.0', port=5001, threaded=True, debug=False) 