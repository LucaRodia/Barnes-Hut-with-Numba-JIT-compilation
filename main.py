import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import njit, prange
import time

# --- Constants ---
G = 1.0  # Gravitational constant
THETA = 0.5  # Barnes-Hut criterion
DT = 0.01  # Time step
SOFTENING = 0.1  # Softening parameter
MAX_NODES_MULTIPLIER = 8  # Safety buffer for QuadTree nodes

# --- Helper Functions ---

@njit
def get_bounds(pos):
    """
    Compute the bounding box of all particles.
    Returns: x_min, x_max, y_min, y_max
    """
    min_x = np.min(pos[:, 0])
    max_x = np.max(pos[:, 0])
    min_y = np.min(pos[:, 1])
    max_y = np.max(pos[:, 1])

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0

    size_x = max_x - min_x
    size_y = max_y - min_y
    
    # Square box with 10% margin
    size = max(size_x, size_y) * 1.1 + 1e-9
    half_size = size / 2.0

    return center_x - half_size, center_x + half_size, center_y - half_size, center_y + half_size

# --- QuadTree Implementation with Numba ---

@njit
def reset_tree(child_idx, node_mass, node_com, leaf_particle_idx):
    """Reset tree arrays for the next frame."""
    child_idx[:] = -1
    node_mass[:] = 0.0
    node_com[:] = 0.0
    leaf_particle_idx[:] = -1

@njit
def build_tree(pos, masses, child_idx, node_mass, node_com, leaf_particle_idx):
    """
    Builds the QuadTree.
    Returns: 
        n_nodes (int): Number of nodes used.
        xmin, xmax, ymin, ymax: Bounds used.
    """
    N = pos.shape[0]
    max_nodes = child_idx.shape[0]
    
    # 1. Reset Arrays (Assuming they are passed in)
    # We assume they are reset outside or we reset them here?
    # To be safe and self-contained, let's assume they might be dirty.
    # But for speed, 'fill' is fast.
    reset_tree(child_idx, node_mass, node_com, leaf_particle_idx)
    
    n_nodes = 1  # Root is node 0
    
    # 2. Compute Root Bounds
    xmin, xmax, ymin, ymax = get_bounds(pos)
    root_center_x = (xmin + xmax) / 2.0
    root_center_y = (ymin + ymax) / 2.0
    root_size = xmax - xmin

    # 3. Insert Particles
    for i in range(N):
        p_pos = pos[i]
        
        # Start at Root
        curr = 0
        cx = root_center_x
        cy = root_center_y
        size = root_size
        
        # Infinite loop for traversal until insertion
        while True:
            # Check for Leaf Collision first
            existing_p = leaf_particle_idx[curr]
            
            if existing_p != -1:
                # Collision! We are at a leaf 'curr' which holds 'existing_p'.
                # We need to split 'curr' until 'existing_p' and 'i' (new particle) separate.
                
                p1 = existing_p
                p2 = i
                
                # Mark 'curr' as internal (no longer a leaf with particle)
                leaf_particle_idx[curr] = -1
                
                # Push both down
                # For p1 (existing), we keep it in the loop logic?
                # No, we handle the split locally here until they diverge.
                
                # We need a sub-loop to push them down together while they share quadrants
                
                while True:
                    # Check safety
                    if n_nodes >= max_nodes - 8:
                        # Error / Overflow
                        return -1, xmin, xmax, ymin, ymax

                    # Determine quadrants for p1 and p2 in the current 'curr' frame
                    # 0: NW, 1: NE, 2: SW, 3: SE
                    q1 = 0
                    if pos[p1, 0] >= cx: q1 += 1
                    if pos[p1, 1] < cy: q1 += 2
                    
                    q2 = 0
                    if pos[p2, 0] >= cx: q2 += 1
                    if pos[p2, 1] < cy: q2 += 2
                    
                    if q1 == q2:
                        # They are in the same quadrant.
                        # Create a specific child node for them if it doesn't exist
                        # (It shouldn't exist because we just converted curr from leaf)
                        
                        # Note: In standard insertion, provided we move *one by one*, 
                        # child slots are empty when we convert a leaf.
                        
                        if child_idx[curr, q1] == -1:
                             new_node = n_nodes
                             n_nodes += 1
                             child_idx[curr, q1] = new_node
                             
                        # Move down to this new node
                        curr = child_idx[curr, q1]
                        
                        # Update geometric properties for next level
                        size *= 0.5
                        step = size / 2.0
                        cx += step if (q1 & 1) else -step
                        cy += step if (q1 & 2)==0 else -step
                        
                        continue # Continue pushing both down
                        
                    else:
                        # They split!
                        # Create child for p1
                        if child_idx[curr, q1] == -1:
                            new_node_1 = n_nodes
                            n_nodes += 1
                            child_idx[curr, q1] = new_node_1
                            leaf_particle_idx[new_node_1] = p1
                        else:
                            # Should not happen if we just converted leaf, 
                            # unless we had existing structure? 
                            # But if curr WAS a leaf, it had NO children.
                            # So this branch is safe.
                            pass

                        # Create child for p2
                        if child_idx[curr, q2] == -1:
                            new_node_2 = n_nodes
                            n_nodes += 1
                            child_idx[curr, q2] = new_node_2
                            leaf_particle_idx[new_node_2] = p2
                            
                        # Done inserting both!
                        break
                
                # Break outer loop (particle i inserted)
                break
            
            else:
                # 'curr' is internal (or empty root).
                # Determine quadrant for p = i
                q = 0
                if p_pos[0] >= cx: q += 1
                if p_pos[1] < cy: q += 2
                
                child = child_idx[curr, q]
                
                if child != -1:
                    # Go deeper
                    curr = child
                    # Update metrics
                    size *= 0.5
                    step = size / 2.0
                    cx += step if (q & 1) else -step
                    cy += step if (q & 2)==0 else -step
                    continue
                else:
                    # Found empty slot.
                    if n_nodes >= max_nodes:
                        return -1, xmin, xmax, ymin, ymax
                        
                    new_node = n_nodes
                    n_nodes += 1
                    child_idx[curr, q] = new_node
                    leaf_particle_idx[new_node] = i
                    # Done
                    break

    return n_nodes, xmin, xmax, ymin, ymax

@njit
def compute_mass_moments(curr, child_idx, node_mass, node_com, leaf_particle_idx):
    """
    Recursively compute mass and COM.
    """
    # If leaf with particle
    p_idx = leaf_particle_idx[curr]
    if p_idx != -1:
        # It's a leaf with a particle. (Mass/Pos are not stored in tree arrays yet?
        # We need to access global 'masses' and 'pos' arrays?
        # WAIT. We can't access 'pos' directly if we don't pass it.
        # BUT: node_mass/com were pre-allocated.
        # We should iterate and set them?
        # Better: pass 'pos' and 'masses' to this function.
        pass # Logic handled below
        return node_mass[curr], node_com[curr, 0], node_com[curr, 1]

    # If internal
    m_total = 0.0
    com_x = 0.0
    com_y = 0.0
    
    for i in range(4):
        child = child_idx[curr, i]
        if child != -1:
            m, cx, cy = compute_mass_moments(child, child_idx, node_mass, node_com, leaf_particle_idx)
            m_total += m
            com_x += m * cx
            com_y += m * cy
            
    if m_total > 0:
        node_mass[curr] = m_total
        node_com[curr, 0] = com_x / m_total
        node_com[curr, 1] = com_y / m_total
    else:
        node_mass[curr] = 0.0
        node_com[curr, 0] = 0.0
        node_com[curr, 1] = 0.0 # Should not happen for active nodes
        
    return m_total, node_com[curr, 0], node_com[curr, 1]

@njit
def fill_leaf_data(n_nodes, leaf_particle_idx, node_mass, node_com, pos, masses):
    """
    Helper to fill leaf mass/com from particle data before reduction.
    """
    for i in range(n_nodes):
        p_idx = leaf_particle_idx[i]
        if p_idx != -1:
            node_mass[i] = masses[p_idx]
            node_com[i, 0] = pos[p_idx, 0]
            node_com[i, 1] = pos[p_idx, 1]

# --- Force Calculation ---

@njit(parallel=True)
def compute_forces(pos, child_idx, node_mass, node_com, leaf_particle_idx, xmin, xmax, ymin, ymax, forces_out):
    N = pos.shape[0]
    # Reset forces
    forces_out[:] = 0.0
    
    root_size = max(xmax - xmin, ymax - ymin)
    root_center_x = (xmin + xmax) / 2.0
    root_center_y = (ymin + ymax) / 2.0

    for i in prange(N):
        p_pos = pos[i]
        
        # Traverse tree
        stack_nodes = np.zeros(256, dtype=np.int32)
        stack_sizes = np.zeros(256, dtype=np.float64)
        stack_cx = np.zeros(256, dtype=np.float64)
        stack_cy = np.zeros(256, dtype=np.float64)
        
        stack_nodes[0] = 0
        stack_sizes[0] = root_size
        stack_cx[0] = root_center_x
        stack_cy[0] = root_center_y
        sp = 1
        
        fx = 0.0
        fy = 0.0
        
        while sp > 0:
            sp -= 1
            curr = stack_nodes[sp]
            size = stack_sizes[sp]
            cx = stack_cx[sp]
            cy = stack_cy[sp]
            
            # Skip empty nodes (shouldn't be in tree usually, but safe check)
            if node_mass[curr] == 0:
                continue
                
            dx = node_com[curr, 0] - p_pos[0]
            dy = node_com[curr, 1] - p_pos[1]
            r2 = dx*dx + dy*dy
            r = np.sqrt(r2)
            
            # Check if leaf
            # In our structure, leaf has children all -1 
            # Or simplified: if leaf_particle_idx != -1, it's definitely a leaf.
            # But we might have empty internal nodes? No.
            # However: node_com of a leaf is the particle position.
            
            # Determine if we can approximate
            # BH Criterion: s/d < theta
            
            is_leaf = (leaf_particle_idx[curr] != -1) or (child_idx[curr, 0] == -1 and child_idx[curr, 1] == -1 and child_idx[curr, 2] == -1 and child_idx[curr, 3] == -1)
            
            if is_leaf:
                leaf_p = leaf_particle_idx[curr]
                if leaf_p != i: # Avoid self
                    inv_dist3 = 1.0 / ((r2 + SOFTENING**2)**1.5)
                    f = G * node_mass[curr] * inv_dist3
                    fx += f * dx
                    fy += f * dy
            else:
                if (size / r) < THETA:
                    # Approx
                    inv_dist3 = 1.0 / ((r2 + SOFTENING**2)**1.5)
                    f = G * node_mass[curr] * inv_dist3
                    fx += f * dx
                    fy += f * dy
                else:
                    # Refine
                    for q in range(4):
                        child = child_idx[curr, q]
                        if child != -1:
                            stack_nodes[sp] = child
                            next_size = size * 0.5
                            stack_sizes[sp] = next_size
                            
                            step = next_size / 2.0
                            stack_cx[sp] = cx + (step if (q & 1) else -step)
                            stack_cy[sp] = cy + (step if (q & 2)==0 else -step)
                            sp += 1
                            
        forces_out[i, 0] = fx
        forces_out[i, 1] = fy

# --- Integration ---

@njit
def integrate(pos, vel, forces, masses, dt):
    # Leapfrog / Semi-implicit
    # v(t+1) = v(t) + a(t)*dt
    # x(t+1) = x(t) + v(t+1)*dt
    
    # Reshape mass for broadcasting
    inv_mass = 1.0 / masses.reshape(-1, 1)
    acc = forces * inv_mass
    
    vel += acc * dt
    pos += vel * dt

# --- Energy Calculations ---

@njit(parallel=True)
def compute_potential_energy(pos, masses, child_idx, node_mass, node_com, leaf_particle_idx, xmin, xmax, ymin, ymax):
    """
    Compute total potential energy using Barnes-Hut.
    U = 0.5 * sum(m_i * phi_i)
    phi_i = -G * sum(m_j / r_ij)
    """
    N = pos.shape[0]
    potentials = np.zeros(N, dtype=np.float64)
    root_size = max(xmax - xmin, ymax - ymin)
    root_center_x = (xmin + xmax) / 2.0
    root_center_y = (ymin + ymax) / 2.0
    
    for i in prange(N):
        p_pos = pos[i]
        
        # Traverse tree (iterative stack)
        stack_nodes = np.zeros(256, dtype=np.int32)
        stack_sizes = np.zeros(256, dtype=np.float64)
        stack_cx = np.zeros(256, dtype=np.float64)
        stack_cy = np.zeros(256, dtype=np.float64)
        
        stack_nodes[0] = 0
        stack_sizes[0] = root_size
        stack_cx[0] = root_center_x
        stack_cy[0] = root_center_y
        sp = 1
        
        phi = 0.0
        
        while sp > 0:
            sp -= 1
            curr = stack_nodes[sp]
            size = stack_sizes[sp]
            cx = stack_cx[sp]
            cy = stack_cy[sp]
            
            if node_mass[curr] == 0:
                continue
                
            dx = node_com[curr, 0] - p_pos[0]
            dy = node_com[curr, 1] - p_pos[1]
            r2 = dx*dx + dy*dy
            r = np.sqrt(r2)
            
            # Check if leaf
            is_leaf = (leaf_particle_idx[curr] != -1) or (child_idx[curr, 0] == -1 and child_idx[curr, 1] == -1 and child_idx[curr, 2] == -1 and child_idx[curr, 3] == -1)
            
            if is_leaf:
                leaf_p = leaf_particle_idx[curr]
                if leaf_p != i: # Avoid self
                    inv_dist = 1.0 / np.sqrt(r2 + SOFTENING**2)
                    phi -= G * node_mass[curr] * inv_dist
            else:
                if (size / r) < THETA:
                    # Approx
                    inv_dist = 1.0 / np.sqrt(r2 + SOFTENING**2)
                    phi -= G * node_mass[curr] * inv_dist
                else:
                    # Refine
                    for q in range(4):
                        child = child_idx[curr, q]
                        if child != -1:
                            stack_nodes[sp] = child
                            next_size = size * 0.5
                            stack_sizes[sp] = next_size
                            step = next_size / 2.0
                            stack_cx[sp] = cx + (step if (q & 1) else -step)
                            stack_cy[sp] = cy + (step if (q & 2)==0 else -step)
                            sp += 1
        potentials[i] = phi
        
    total_potential = 0.5 * np.sum(potentials * masses)
    return total_potential

@njit
def compute_kinetic_energy(masses, vel):
    # T = 0.5 * sum(m * v^2)
    v2 = np.sum(vel**2, axis=1)
    return 0.5 * np.sum(masses * v2)

# --- Initialization ---

def initialize_system(config_name, N, **kwargs):
    pos = np.zeros((N, 2), dtype=np.float64)
    vel = np.zeros((N, 2), dtype=np.float64)
    masses = np.ones(N, dtype=np.float64)
    
    if config_name == "binary_static_random":
        D = kwargs.get('D', 150.0)
        M = kwargs.get('M', 5000.0)
        
        pos[0] = [-D, 0]
        pos[1] = [D, 0]
        masses[0] = M
        masses[1] = M
        vel[0] = [0, 0] # Static
        vel[1] = [0, 0] 
        
        # Random cloud
        for i in range(2, N):
            pos[i] = np.random.uniform(-D*2, D*2, 2)
            masses[i] = np.random.uniform(0.1, 1.0)
            vel[i] = [0, 0] # Collapse scenario
            
    elif config_name == "galaxy_gaussian_hole":
        center_mass = kwargs.get('center_mass', 10000.0)
        r_min = kwargs.get('r_min', 50.0)
        r_scale = kwargs.get('r_scale', 100.0)
        
        pos[0] = [0, 0]
        masses[0] = center_mass
        vel[0] = [0, 0]
        
        for i in range(1, N):
            # Rejection sampling for hole
            while True:
                # Gaussian distribution
                x = np.random.normal(0, r_scale)
                y = np.random.normal(0, r_scale)
                r = np.sqrt(x*x + y*y)
                if r > r_min:
                    pos[i] = [x, y]
                    break
            
            # Keplerian Velocity
            r = np.sqrt(np.sum(pos[i]**2))
            v = np.sqrt(G * center_mass / r)
            angle = np.arctan2(pos[i, 1], pos[i, 0])
            vel[i] = [-v * np.sin(angle), v * np.cos(angle)]
            masses[i] = np.random.uniform(0.5, 2.0)
            
    elif config_name == "binary_orbiting":
        D = kwargs.get('D', 100.0) # Half-separation
        M = kwargs.get('M', 5000.0)
        
        # Binary Stars
        pos[0] = [-D, 0]
        pos[1] = [D, 0]
        masses[0] = M
        masses[1] = M
        
        # V for circular orbit of binary
        # a = v^2 / r, F = G M M / (2r)^2
        # r here is D. Distance between masses is 2D.
        # F = G M^2 / 4D^2 = M v^2 / D  => v^2 = G M / 4D
        v_star = np.sqrt(G * M / (4 * D))
        
        vel[0] = [0, v_star]  # Counter-clockwise
        vel[1] = [0, -v_star]
        
        # Clouds around stars
        for i in range(2, N):
            star_idx = 0 if i % 2 == 0 else 1
            star_pos = pos[star_idx]
            star_vel = vel[star_idx]
            
            # Local cloud
            angle = np.random.uniform(0, 2*np.pi)
            dist = np.random.uniform(10, D/2)
            local_pos = np.array([dist * np.cos(angle), dist * np.sin(angle)])
            pos[i] = star_pos + local_pos
            
            # Orbital velocity around the star
            v_local_mag = np.sqrt(G * M / dist)
            v_local = np.array([-v_local_mag * np.sin(angle), v_local_mag * np.cos(angle)])
            
            vel[i] = star_vel + v_local
            masses[i] = np.random.uniform(0.1, 0.5)
            
    else:
        # Fallback to random uniform
        print(f"Unknown config '{config_name}', using random uniform.")
        pos = np.random.uniform(-200, 200, (N, 2))
        vel = np.zeros((N, 2))
        
    return pos, vel, masses

# --- Main ---

def run_simulation():
    print("Select Initial Configuration:")
    print("1. binary_static_random")
    print("2. galaxy_gaussian_hole")
    print("3. binary_orbiting")
    
    choice = input("Enter choice (1-3) or name: ").strip()
    
    config_map = {
        "1": "binary_static_random",
        "2": "galaxy_gaussian_hole",
        "3": "binary_orbiting"
    }
    
    config_name = config_map.get(choice, choice)
    if config_name not in config_map.values():
        print("Invalid choice, defaulting to galaxy_gaussian_hole")
        config_name = "galaxy_gaussian_hole"

    N_input = input("Enter number of particles (default 2000): ").strip()
    try:
        N = int(N_input)
    except:
        N = 2000
    
    # Calculate MAX_NODES safely
    # For very large N, we might need more buffer, but 8x is usually enough.
    MAX_NODES = int(N * MAX_NODES_MULTIPLIER)
    print(f"Initializing {config_name} with N={N}, MaxNodes={MAX_NODES}")
    
    pos, vel, masses = initialize_system(config_name, N)
    forces = np.zeros((N, 2), dtype=np.float64)
    
    # Tree Arrays
    tree_child_idx = np.zeros((MAX_NODES, 4), dtype=np.int32)
    tree_mass = np.zeros(MAX_NODES, dtype=np.float64)
    tree_com = np.zeros((MAX_NODES, 2), dtype=np.float64)
    tree_leaf_p = np.zeros(MAX_NODES, dtype=np.int32)
    
    # JIT Compile
    print("Compiling JIT functions...")
    start_compile = time.time()
    reset_tree(tree_child_idx, tree_mass, tree_com, tree_leaf_p)
    nb, x1, x2, y1, y2 = build_tree(pos, masses, tree_child_idx, tree_mass, tree_com, tree_leaf_p)
    fill_leaf_data(nb, tree_leaf_p, tree_mass, tree_com, pos, masses)
    compute_mass_moments(0, tree_child_idx, tree_mass, tree_com, tree_leaf_p)
    compute_forces(pos, tree_child_idx, tree_mass, tree_com, tree_leaf_p, x1, x2, y1, y2, forces)
    integrate(pos, vel, forces, masses, DT)
    compute_kinetic_energy(masses, vel)
    compute_potential_energy(pos, masses, tree_child_idx, tree_mass, tree_com, tree_leaf_p, x1, x2, y1, y2)
    print(f"Compilation finished in {time.time() - start_compile:.2f}s")
    
    # Viz
    fig = plt.figure(figsize=(14, 6), facecolor='black')
    ax = fig.add_axes([0.05, 0.1, 0.4, 0.8], facecolor='black') # Simulation
    ax_graph = fig.add_axes([0.55, 0.1, 0.4, 0.8], facecolor='black') # Energy Graph
    
    points, = ax.plot([], [], 'o', markersize=0.8, color='cyan', alpha=0.7)
    text_info = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white', family='monospace')
    
    # Energy Lines
    line_T, = ax_graph.plot([], [], label='Kinetic (T)', color='red')
    line_U, = ax_graph.plot([], [], label='Potential (U)', color='blue')
    line_E, = ax_graph.plot([], [], label='Total (E)', color='green')
    ax_graph.legend(loc='upper right')
    ax_graph.set_title("Energy Evolution", color='white')
    ax_graph.tick_params(colors='white', which='both')
    for spine in ax_graph.spines.values(): spine.set_edgecolor('white')
    ax_graph.set_facecolor('black')
    ax_graph.grid(True, color='gray', alpha=0.3)
    
    energy_history = {'T': [], 'U': [], 'E': [], 't': []}
    perf_history = []
    
    def init():
        return points, text_info, line_T, line_U, line_E

    def update(frame):
        t0 = time.time()
        
        # Physics Step
        
        # 1. Reset
        reset_tree(tree_child_idx, tree_mass, tree_com, tree_leaf_p)
        
        # 2. Build
        n_nodes, xmin, xmax, ymin, ymax = build_tree(pos, masses, tree_child_idx, tree_mass, tree_com, tree_leaf_p)
        
        if n_nodes == -1:
            print("Error: Tree Node Overflow")
            return points, text_info, line_T, line_U, line_E
            
        # 3. Moments
        fill_leaf_data(n_nodes, tree_leaf_p, tree_mass, tree_com, pos, masses)
        compute_mass_moments(0, tree_child_idx, tree_mass, tree_com, tree_leaf_p)
        
        # 4. Forces
        compute_forces(pos, tree_child_idx, tree_mass, tree_com, tree_leaf_p, xmin, xmax, ymin, ymax, forces)
        
        # 5. Integrate
        integrate(pos, vel, forces, masses, DT)
        
        # Energy Calc
        T = compute_kinetic_energy(masses, vel)
        U = compute_potential_energy(pos, masses, tree_child_idx, tree_mass, tree_com, tree_leaf_p, xmin, xmax, ymin, ymax)
        E = T + U
        
        energy_history['T'].append(T)
        energy_history['U'].append(U)
        energy_history['E'].append(E)
        energy_history['t'].append(frame)
        
        t1 = time.time()
        dt_frame = (t1 - t0) * 1000
        perf_history.append(dt_frame)
        
        # Draw Sim
        points.set_data(pos[:, 0], pos[:, 1])
        box_size = max(xmax - xmin, ymax - ymin)
        margin = box_size * 0.1
        cx = (xmax + xmin) / 2
        cy = (ymax + ymin) / 2
        
        # Smooth camera or just jump? Jump is fine for now.
        if not np.isnan(cx) and not np.isnan(box_size):
             ax.set_xlim(cx - box_size/2 - margin, cx + box_size/2 + margin)
             ax.set_ylim(cy - box_size/2 - margin, cy + box_size/2 + margin)
        
        text_info.set_text(f'N: {N} | Nodes: {n_nodes}\nTime: {dt_frame:.1f} ms')
        
        # Update Graph (windowed)
        window = 300
        start_idx = max(0, len(energy_history['t']) - window)
        
        ts = energy_history['t'][start_idx:]
        Ts = energy_history['T'][start_idx:]
        Us = energy_history['U'][start_idx:]
        Es = energy_history['E'][start_idx:]
        
        if ts:
            ax_graph.set_xlim(min(ts), max(ts)+1)
            all_vals = Ts + Us + Es
            if all_vals:
                min_v = min(all_vals)
                max_v = max(all_vals)
                margin_v = (max_v - min_v) * 0.1 if max_v != min_v else 1.0
                ax_graph.set_ylim(min_v - margin_v, max_v + margin_v)
            
            line_T.set_data(ts, Ts)
            line_U.set_data(ts, Us)
            line_E.set_data(ts, Es)
        
        return points, text_info, line_T, line_U, line_E

    ani = animation.FuncAnimation(fig, update, frames=None, init_func=init, blit=False, interval=1)
    
    try:
        plt.show()
    except Exception as e:
        print(f"Visualization error: {e}")
        
    # Plot Performance on exit
    if perf_history:
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(perf_history)
            plt.title(f"Frame Computation Time (ms) - N={N} {config_name}")
            plt.ylabel("Time (ms)")
            plt.xlabel("Frame")
            plt.grid(True)
            filename = f"perf_{config_name}_N{N}_{int(time.time())}.png"
            plt.savefig(filename)
            print(f"Saved performance plot to {filename}")
        except:
            pass

if __name__ == "__main__":
    run_simulation()
