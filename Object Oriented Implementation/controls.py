def compute_cinetic_energy(bodies):
    """Energia cinetica totale (dovrebbe conservarsi circa)"""
    E = 0
    for body in bodies:
        v2 = body.speed[0]**2 + body.speed[1]**2
        E += 0.5 * body.mass * v2
    return E

def compute_potential_energy(bodies, k=-200):
    E_pot = 0.0
    for i, b1 in enumerate(bodies):
        for b2 in bodies[i+1:]:
            dx = b1.pos[0] - b2.pos[0]
            dy = b1.pos[1] - b2.pos[1]
            r = (dx**2 + dy**2)**0.5
            if r > 1e-6:
                E_pot += k / r  # k negativo → E_pot negativa (bound system)
    return E_pot

def check_bounds(bodies, x_min, x_max, y_min, y_max):
    """Verifica corpi fuori bounds"""
    out_of_bounds = 0
    for body in bodies:
        if not (x_min <= body.pos[0] <= x_max and y_min <= body.pos[1] <= y_max):
            out_of_bounds += 1
    return out_of_bounds

def compute_total_momentum(bodies):
    px = sum(b.mass * b.speed[0] for b in bodies)
    py = sum(b.mass * b.speed[1] for b in bodies)
    return px, py