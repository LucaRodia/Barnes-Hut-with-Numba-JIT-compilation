import time
import numpy as np
import matplotlib.pyplot as plt
import os
from main import (
    initialize_system, reset_tree, build_tree, fill_leaf_data, 
    compute_mass_moments, compute_forces, integrate, DT, MAX_NODES_MULTIPLIER
)

# Configuration
RUN_NUMBER = "001"
SETUP_TYPE = "benchmark_headless"
OUTPUT_DIR = f"{SETUP_TYPE}_{RUN_NUMBER}"

def run_benchmark():
    n_values = [1000, 2000, 5000, 10000, 20000, 30000]
    steps = 100 
    times = []
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Results will be saved to: {OUTPUT_DIR}")
    
    print(f"Starting Benchmark (Steps={steps} per N)...")
    
    for N in n_values:
        print(f"Testing N={N}...", end="", flush=True)
        
        # Init
        # Use a simple config for benchmarking, or make it configurable?
        # User asked for "diverse quantità di particelle", didn't specify config type for benchmark.
        # "binary_static_random" is good because it doesn't require heavy setup calc.
        config_name = "binary_static_random" 
        
        # Safe MAX_NODES
        MAX_NODES = int(N * MAX_NODES_MULTIPLIER)
        
        # Arrays
        pos, vel, masses = initialize_system(config_name, N)
        forces = np.zeros((N, 2), dtype=np.float64)
        
        tree_child_idx = np.zeros((MAX_NODES, 4), dtype=np.int32)
        tree_mass = np.zeros(MAX_NODES, dtype=np.float64)
        tree_com = np.zeros((MAX_NODES, 2), dtype=np.float64)
        tree_leaf_p = np.zeros(MAX_NODES, dtype=np.int32)
        
        # Warmup / JIT Compile Trigger
        # Run one step to ensure everything is compiled
        reset_tree(tree_child_idx, tree_mass, tree_com, tree_leaf_p)
        nb, x1, x2, y1, y2 = build_tree(pos, masses, tree_child_idx, tree_mass, tree_com, tree_leaf_p)
        fill_leaf_data(nb, tree_leaf_p, tree_mass, tree_com, pos, masses)
        compute_mass_moments(0, tree_child_idx, tree_mass, tree_com, tree_leaf_p)
        compute_forces(pos, tree_child_idx, tree_mass, tree_com, tree_leaf_p, x1, x2, y1, y2, forces)
        integrate(pos, vel, forces, masses, DT)
        
        # Measurement Loop
        start_time = time.time()
        
        for _ in range(steps):
             reset_tree(tree_child_idx, tree_mass, tree_com, tree_leaf_p)
             n_nodes, xmin, xmax, ymin, ymax = build_tree(pos, masses, tree_child_idx, tree_mass, tree_com, tree_leaf_p)
             
             if n_nodes == -1:
                 print(" Error: Tree Node Overflow")
                 break
                 
             fill_leaf_data(n_nodes, tree_leaf_p, tree_mass, tree_com, pos, masses)
             compute_mass_moments(0, tree_child_idx, tree_mass, tree_com, tree_leaf_p)
             compute_forces(pos, tree_child_idx, tree_mass, tree_com, tree_leaf_p, xmin, xmax, ymin, ymax, forces)
             integrate(pos, vel, forces, masses, DT)
             
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_ms = (total_time / steps) * 1000
        times.append(avg_time_ms)
        print(f" {avg_time_ms:.2f} ms/step")
        
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, times, 'o-', linewidth=2, label='Mean Step Time')
    plt.xlabel("Number of Particles (N)")
    plt.ylabel("Time (ms)")
    plt.title("Barnes-Hut Simulation Scalability")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "benchmark_linear.png"))
    
    # Log-Log
    plt.figure(figsize=(10, 6))
    plt.loglog(n_values, times, 'o-', linewidth=2, label='Mean Step Time')
    plt.xlabel("Number of Particles (N)")
    plt.ylabel("Time (ms)")
    plt.title("Barnes-Hut Simulation Scalability (Log-Log)")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "benchmark_loglog.png"))
    
    # Save Data
    np.savetxt(os.path.join(OUTPUT_DIR, "results.csv"), 
               np.column_stack((n_values, times)), 
               delimiter=",", 
               header="N,Time_ms", 
               comments="")
               
    print("Benchmark completed.")

if __name__ == "__main__":
    run_benchmark()
