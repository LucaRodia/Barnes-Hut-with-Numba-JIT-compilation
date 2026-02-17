import matplotlib.pyplot as plt
import pickle
import numpy as np

def setup_plot(num_bodies):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Inizializziamo con zeri per allocare la memoria dello scatter
    dummy_x = np.zeros(num_bodies)
    dummy_y = np.zeros(num_bodies)
    
    scatter = ax.scatter(dummy_x, dummy_y, s=15, alpha=0.7, c='blue')
    title = ax.set_title('N-Body Simulation - Epoch 0')
    ax.grid(True, alpha=0.3)
    
    plt.ion()
    plt.show(block=False)
    return fig, ax, scatter, title

def plot_bodies(bodies, scatter, title, epoch):
    # 1. Estrazione dati veloce
    if hasattr(bodies[0], 'pos'):
        x = [b.pos[0] for b in bodies]
        y = [b.pos[1] for b in bodies]
    else:
        x, y = bodies[:, 0], bodies[:, 1]
    
    # 2. Aggiorna i punti (operazione leggera)
    data = np.column_stack((x, y))
    scatter.set_offsets(data)
    
    # 3. Aggiorna titolo
    title.set_text(f'Epoch {epoch}')
    
    # 4. Auto-Zoom (Il collo di bottiglia è qui, ma serve se i corpi scappano)
    # Lo rendiamo più semplice possibile
    ax = scatter.axes
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    
    margin = 50 # Margine fisso per evitare tremolii
    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(min_y - margin, max_y + margin)
    
    # 5. Pausa brevissima per aggiornare la GUI
    plt.pause(0.0001)

#plot the graphs with the data saved

def plot_energy_graphs():
    with open("/home/luca/Barnes-Hut Algorithm/code/dati.pkl", 'rb') as f:
        data = pickle.load(f)

        cinetic_energy_list = data["cinetic_energy"]
        potential_energy_list = data["potential_energy"]
        CM_list = data["body_CM"]
        bodies_outside_list = data['bodies_outside']
        momentum_list = data["total_momentum"]

        total_energy_list = []
        for i in range(len(cinetic_energy_list)):
            total_energy_list.append(cinetic_energy_list[i] + potential_energy_list[i])

        X_cm = [p[0] for p in CM_list]
        Y_cm = [p[1] for p in CM_list]

        X_mom = [p[0] for p in momentum_list]
        Y_mom = [p[0] for p in momentum_list]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (10, 8))

        ax1.plot(cinetic_energy_list, 'o-', color="green")
        ax1.plot(potential_energy_list, 'o-', color="yellow")
        ax1.plot(total_energy_list, 'o-', color="violet")
        ax1.set_title("Energy. cinetic (green), potential (yellow), total (violet)")

        ax2.plot(bodies_outside_list, 'o-', color="orange")
        ax2.set_title("Bodies Outside Grid")

        ax3.plot(X_cm, 'o-', color="red")
        ax3.plot(Y_cm, 'o-', color="blue")
        ax3.set_title("CM coordinates. X (red), Y (blue)")

        ax4.plot(X_mom, 'o-', color="pink")
        ax4.plot(Y_mom, 'o-', color="green")
        ax4.set_title("Momentum. X (pink), Y (green)")

        plt.tight_layout()  # Evita sovrapposizioni
        plt.show()

plot_energy_graphs()