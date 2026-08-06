# lib/simulator/lookup_tables.py
#
# Hardware-Aware Automated Quantization (HAQ) & Project Omnia
# Hardware Simulator / Latency Estimator Module
#
# This module provides a Roofline-based analytical latency estimation
# for quantized edge accelerators (e.g., FPGA with Custom HLS datapath).
# It replaces the parameter-count fallback in entropy_quantize.py with 
# realistic hardware execution metrics.

import torch
import torch.nn as nn
from lib.utils.quantize_utils import QConv2d, QLinear

class LatencyEstimator:
    def __init__(self, target_hardware='fpga_edge_custom_hls'):
        """
        Initializes the analytical hardware simulator.
        Default profile is an edge FPGA running a custom INT4/INT8 
        mixed-precision HLS datapath.
        """
        self.target_hardware = target_hardware
        
        # Target hardware clock frequency
        self.clock_freq_mhz = 200.0
        
        # Peak MACs per cycle (compute bound)
        # Custom datapath packs two INT4 ops into one INT8 multiplier
        self.peak_macs_per_cycle_int8 = 1024 
        self.peak_macs_per_cycle_int4 = 2048 

        # Memory bandwidth (bytes per cycle) (memory bound)
        self.bandwidth_bytes_per_cycle = 16

    def _get_module_macs_and_bytes(self, module):
        """Extracts operation and parameter counts from QModules."""
        macs = 0
        params = 0
        
        if isinstance(module, QConv2d):
            # Proxy calculation: standard average spatial dimension (28x28) 
            # for mid-network feature maps in MobileNet/ResNet
            spatial_area = 28 * 28 
            macs = module.weight.numel() * spatial_area
            params = module.weight.numel()
        elif isinstance(module, QLinear):
            macs = module.weight.numel()
            params = module.weight.numel()
            
        return macs, params

    def estimate(self, model, strategy):
        """
        Estimates the latency of the model under the given bit-width strategy.
        
        Args:
            model: The PyTorch model (containing QConv2d / QLinear).
            strategy: dict of {layer_idx: (w_bit, a_bit)}
            
        Returns:
            dict containing detailed hardware performance estimates.
        """
        total_macs = 0
        total_cycles = 0
        total_memory_bytes = 0
        
        idx_to_module = {i: m for i, m in enumerate(model.modules())}
        
        for idx, bits in strategy.items():
            if idx not in idx_to_module:
                continue
                
            m = idx_to_module[idx]
            w_bit, a_bit = bits
            
            macs, params = self._get_module_macs_and_bytes(m)
            total_macs += macs
            
            # Calculate memory bottleneck delay
            weight_bytes = params * (w_bit / 8.0)
            total_memory_bytes += weight_bytes
            memory_cycles = weight_bytes / self.bandwidth_bytes_per_cycle
            
            # Calculate compute bottleneck delay
            if w_bit == 4 and a_bit == 4:
                compute_cycles = macs / self.peak_macs_per_cycle_int4
            else:
                # INT8 or mixed defaults to the standard INT8 datapath speed
                compute_cycles = macs / self.peak_macs_per_cycle_int8
                
            # Roofline estimate: layer latency is the max of compute and memory delays
            layer_cycles = max(compute_cycles, memory_cycles)
            total_cycles += layer_cycles

        # Base overhead for unquantized layers (stem, pool, BN) and memory access latency
        total_cycles += 150000 
        
        # Convert total cycles to milliseconds
        estimated_latency_ms = (total_cycles / (self.clock_freq_mhz * 1e6)) * 1000.0

        return {
            'total_macs_millions': round(total_macs / 1e6, 2),
            'total_cycles': int(total_cycles),
            'estimated_latency_ms': round(estimated_latency_ms, 3),
            'bottleneck': 'Compute' if (total_macs / self.peak_macs_per_cycle_int8) > (total_memory_bytes / self.bandwidth_bytes_per_cycle) else 'Memory',
            'hardware_profile': self.target_hardware
        }
