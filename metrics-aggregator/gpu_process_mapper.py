#!/usr/bin/env python3
"""
GPU Process to Inference Server Mapper Utility - JSON Output Mode

This script fetches data from 'amd-smi', identifies parent processes
of those reported by amd-smi, checks if the parent is a vLLM or SGLang inference server,
maps it to a partition defined in 'config/partition_config.json',
and outputs a JSON object containing:
1. vllm_vram_by_true_partition: {true_partition_id: summed_vram_gb}
2. true_partition_to_smi_gpu_id: {true_partition_id: amd_smi_gpu_id_where_vllm_runs}
3. all_smi_gpu_details: {smi_gpu_id: {'utilization': %, 'vram_total_gb': GB}}
"""
import subprocess
import json
import os
import re
from collections import defaultdict

try:
    import psutil
except ImportError:
    print(json.dumps({"error": "psutil library is required. Please install it using 'pip install psutil'"}))
    exit(1)

CONFIG_FILE_PATH = 'config/partition_config.json' # Relative to where script is run
AMD_SMI_COMMAND = 'sudo amd-smi monitor -q --json'

def load_partition_config():
    """Loads the partition configuration."""
    try:
        # Try to load from /app/config/ first (for container)
        # then from ./config/ (for local dev)
        config_paths_to_try = [
            '/app/config/partition_config.json',
            'config/partition_config.json'
        ]
        loaded_path = None
        for path_to_try in config_paths_to_try:
            if os.path.exists(path_to_try):
                with open(path_to_try, 'r') as f:
                    # print(f"DEBUG_MAPPER: Loading config from {path_to_try}", file=sys.stderr)
                    return json.load(f)
        # If neither path worked
        print(json.dumps({"error": f"Partition config file not found at expected locations: {config_paths_to_try}"}))
        return None
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Could not decode JSON from partition config. Error: {e}"}))
        return None

def fetch_amd_smi_data():
    """Executes amd-smi and returns its JSON output."""
    try:
        result = subprocess.run(AMD_SMI_COMMAND, shell=True, check=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True, timeout=45)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        return {"error": f"Error running amd-smi: {e}, Stderr: {e.stderr}"}
    except subprocess.TimeoutExpired:
        return {"error": "Timeout running amd-smi command after 45 seconds."}
    except json.JSONDecodeError as e:
        return {"error": f"Error parsing JSON from amd-smi: {e}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred with amd-smi: {e}"}

def get_process_details(pid, level=0, max_level=2):
    """Gets command line of a process, tracing up if it's a spawner."""
    if level > max_level:
        return pid, None
    try:
        proc = psutil.Process(pid)
        ppid = proc.ppid()
        cmdline = proc.cmdline()
        
        # Handle empty cmdline case
        if not cmdline:
            return pid, None
            
        # Handle multiprocessing spawn parent trace
        if any("multiprocessing.spawn" in arg for arg in cmdline) and ppid != 1:
            return get_process_details(ppid, level + 1, max_level)
        
        # Handle python launcher case - look at vLLM or SGLang process specifically
        if cmdline and len(cmdline) > 1 and "python" in cmdline[0].lower():
            # Check if any argument contains vLLM or SGLang
            if any('vllm.entrypoints.openai.api_server' in arg for arg in cmdline) or \
               any('sglang.launch_server' in arg for arg in cmdline):
                return pid, cmdline
                
        return pid, cmdline
    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        # More specific handling of psutil exceptions
        return pid, None
    except Exception as e:
        # General exception fallback
        return pid, None

def find_port_in_cmdline(cmdline):
    """Extracts the port number if --port is present in the command line."""
    if not cmdline: return None
    try:
        port_index = cmdline.index('--port')
        if port_index + 1 < len(cmdline) and cmdline[port_index + 1].isdigit():
            return cmdline[port_index + 1]
    except ValueError: pass
    return None

def map_port_to_partition_id(port, partition_config, server_type):
    """Maps a port to a partition ID and name using the config."""
    if not port or not partition_config: return None
    
    # For vLLM servers, check metrics_url
    if server_type == "vllm":
        for part_id, config in partition_config.items():
            if isinstance(config, dict) and f":{port}/metrics" in config.get('metrics_url', ''):
                return part_id
    # For SGLang servers, use port 9100 + offset to match
    elif server_type == "sglang":
        try:
            # SGLang servers use ports 9101-9108 for partitions 24-31
            port_num = int(port)
            if 9101 <= port_num <= 9108:
                # Map from SGLang port to partition ID
                # 9101 -> 24, 9102 -> 25, etc.
                partition_id = str(port_num - 9101 + 24)
                
                # Verify this is a SGLang partition in config
                if partition_id in partition_config:
                    config = partition_config[partition_id]
                    if isinstance(config, dict) and config.get('service_type') == 'sglang':
                        return partition_id
        except ValueError:
            pass  # Port is not a valid integer
    
    return None

def main():
    partition_config = load_partition_config()
    if not partition_config or 'error' in partition_config:
        # Error already printed as JSON by load_partition_config if it failed
        if not partition_config: # If it returned None truly
             print(json.dumps({"error": "Partition config could not be loaded (returned None)."}))
        return

    amd_smi_data = fetch_amd_smi_data()
    if not amd_smi_data or 'error' in amd_smi_data:
        print(json.dumps(amd_smi_data if isinstance(amd_smi_data, dict) else {"error": "amd-smi data fetching failed (not dict)"}))
        return

    if not isinstance(amd_smi_data, list):
        print(json.dumps({"error": "amd-smi data is not a list as expected."}))
        return
        
    vllm_vram_by_true_partition = defaultdict(float)
    true_partition_to_smi_gpu_id = {}
    all_smi_gpu_details = {}

    for gpu_entry in amd_smi_data:
        smi_gpu_id = gpu_entry.get('gpu')
        if smi_gpu_id is None: continue
        smi_gpu_id_str = str(smi_gpu_id)

        # Store base details for all SMI GPUs
        vram_total_info = gpu_entry.get('vram_total', {})
        vram_total_val = vram_total_info.get('value', 0.0)
        vram_total_unit = vram_total_info.get('unit', 'GB').upper()
        actual_vram_total_gb = 0.0
        if vram_total_unit in ['MB', 'MIB']: actual_vram_total_gb = vram_total_val / 1024.0
        elif vram_total_unit == 'B': actual_vram_total_gb = vram_total_val / (1024.0**3)
        elif vram_total_unit in ['KB', 'KIB']: actual_vram_total_gb = vram_total_val / (1024.0**2)
        else: actual_vram_total_gb = vram_total_val
        
        all_smi_gpu_details[smi_gpu_id_str] = {
            'utilization': gpu_entry.get('gfx', {}).get('value', 0),
            'vram_total_gb': round(actual_vram_total_gb, 2)
        }

        process_list = gpu_entry.get('process_list', [])
        for proc_entry in process_list:
            if not (isinstance(proc_entry, dict) and 'process_info' in proc_entry):
                continue
            proc_info = proc_entry['process_info']
            if not isinstance(proc_info, dict): continue

            child_pid = proc_info.get('pid')
            if child_pid is None: continue

            child_vram_gb = 0.0
            if 'memory_usage' in proc_info and 'vram_mem' in proc_info['memory_usage']:
                vram_mem_info = proc_info['memory_usage']['vram_mem']
                vram_val_bytes = vram_mem_info.get('value', 0)
                unit = vram_mem_info.get('unit', 'B').upper()
                if unit == 'B': child_vram_gb = vram_val_bytes / (1024.0**3)
                elif unit in ['KB', 'KIB']: child_vram_gb = vram_val_bytes / (1024.0**2)
                elif unit in ['MB', 'MIB']: child_vram_gb = vram_val_bytes / 1024.0
                elif unit in ['GB', 'GIB']: child_vram_gb = vram_val_bytes
            
            traced_pid, traced_cmdline = get_process_details(int(child_pid))
            
            # Check for vLLM servers
            if traced_cmdline and any('vllm.entrypoints.openai.api_server' in arg for arg in traced_cmdline):
                port = find_port_in_cmdline(traced_cmdline)
                true_part_id = map_port_to_partition_id(port, partition_config, "vllm")
                if true_part_id:
                    vllm_vram_by_true_partition[true_part_id] += child_vram_gb
                    if true_part_id not in true_partition_to_smi_gpu_id:
                        true_partition_to_smi_gpu_id[true_part_id] = smi_gpu_id_str
            
            # Check for SGLang servers
            elif traced_cmdline and any('sglang.launch_server' in arg for arg in traced_cmdline):
                port = find_port_in_cmdline(traced_cmdline)
                true_part_id = map_port_to_partition_id(port, partition_config, "sglang")
                if true_part_id:
                    vllm_vram_by_true_partition[true_part_id] += child_vram_gb
                    if true_part_id not in true_partition_to_smi_gpu_id:
                        true_partition_to_smi_gpu_id[true_part_id] = smi_gpu_id_str

    # Round VRAM sums
    for k, v in vllm_vram_by_true_partition.items():
        vllm_vram_by_true_partition[k] = round(v, 2)

    output_data = {
        "vllm_vram_by_true_partition": dict(vllm_vram_by_true_partition),
        "true_partition_to_smi_gpu_id": true_partition_to_smi_gpu_id,
        "all_smi_gpu_details": all_smi_gpu_details
    }
    print(json.dumps(output_data, indent=2))

if __name__ == '__main__':
    # This script is intended to be called by other Python scripts and output JSON to stdout.
    # Error messages will also be JSON for easier parsing by the caller.
    # Debug prints should go to stderr if absolutely necessary for standalone debugging.
    import sys
    # Optional: redirect stderr to a file for debugging when run by other script
    # sys.stderr = open('debug_mapper_stderr.log', 'w') 
    main() 