# Hardware Acceleration Optimization for MacOS M4 Chip
# This file contains recommendations and optimizations for running on Apple Silicon

import os
import multiprocessing as mp

def configure_hardware_acceleration():
    """
    Configure hardware acceleration settings for MacOS M4 chip
    """
    # Set environment variables for Apple Silicon optimization
    os.environ['OMP_NUM_THREADS'] = str(mp.cpu_count())
    os.environ['OPENBLAS_NUM_THREADS'] = str(mp.cpu_count())
    os.environ['MKL_NUM_THREADS'] = str(mp.cpu_count())
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(mp.cpu_count())
    os.environ['NUMEXPR_NUM_THREADS'] = str(mp.cpu_count())
    
    # For neural network libraries (if used), prefer Metal backend on Apple Silicon
    # os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # For PyTorch on M1/M2/M3/M4
    
    print(f"Hardware acceleration configured for {mp.cpu_count()} CPU cores")


def get_optimal_workers():
    """
    Determine optimal number of worker processes based on hardware
    """
    cpu_count = mp.cpu_count()
    # For I/O heavy tasks, we can use more workers
    # For CPU heavy tasks, use fewer to avoid overhead
    return min(cpu_count, 8)  # Reasonable default for most applications


if __name__ == "__main__":
    configure_hardware_acceleration()
    print(f"Optimal workers: {get_optimal_workers()}")