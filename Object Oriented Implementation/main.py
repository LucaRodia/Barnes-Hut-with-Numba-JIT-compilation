import pickle
import random
import matplotlib.pyplot as plt

from Tree import Tree
from Node import Node
from Body import Body
from controls import compute_cinetic_energy, check_bounds, compute_potential_energy, compute_total_momentum
from plot import setup_plot, plot_bodies, plot_energy_graphs

random.seed(42)

epochs = 1000
num_bodies= 100
x_min, y_min = 0, 0
x_max, y_max = 200, 200 
theta = 0.5
dt = 0.01   		#seconds

#BH_Tree = Tree(x_min, y_min, x_max, y_max, theta, dt)

bodies = [Body(mass=1.0, pos=[random.uniform(0,200), random.uniform(0,200)]) 
          for _ in range(num_bodies)]
#for body in bodies:
#	body.speed = [-0.05*body.pos[1], 0.05*body.pos[0]]

fig, ax, scatter, title = setup_plot()


cinetic_energy_list = []
potential_energy_list = []
center_of_mass_list = []
bodies_outside_list = []
total_momentum_list = []


i = 0
for epoch in range(epochs):

	BH_Tree = Tree(x_min-10, y_min-10, x_max+10, y_max+10, theta, dt)

	BH_Tree.build_tree(bodies)
	BH_Tree.compute_force_tree(bodies)

	f_mag = (bodies[0].force[0]**2 + bodies[0].force[1]**2)**0.5

	BH_Tree.apply_force_tree(bodies)

	x_coords = [b.pos[0] for b in bodies]
	y_coords = [b.pos[1] for b in bodies]

	x_min, x_max = min(x_coords), max(x_coords)
	y_min, y_max = min(y_coords), max(y_coords)

	plot_bodies(x_coords, y_coords, ax, scatter, title, epoch, bounds=[x_min, x_max, y_min, y_max])
	if epoch == 0:
		print(f"Body 0 force: {bodies[0].force}")
		print(f"Body 0 speed before: {bodies[0].speed}")
		
	if epoch%10 == 0:
		print(f"Epoch {epoch} - Body 0 Force magnitude: {f_mag}")
		
		cinetic_energy_list.append(compute_cinetic_energy(bodies))
		potential_energy_list.append(compute_potential_energy(bodies))
		center_of_mass_list.append(BH_Tree.root.center_of_mass)
		bodies_outside_list.append(check_bounds(bodies, 0, 200, 0, 200))
		total_momentum_list.append(compute_total_momentum(bodies))

		print(f"Epoch {epoch}: cinetic={cinetic_energy_list[i]}, "
              f"potential={potential_energy_list[i]}"
			  f"CM={center_of_mass_list[i]}, "
              f"Out={bodies_outside_list[i]}"
			  f"momentum={total_momentum_list[i]}")
		
		i+= 1


	BH_Tree.reset_tree()
	
with open('dati.pkl', 'wb') as f:
    pickle.dump({'cinetic_energy': cinetic_energy_list, "potential_energy": potential_energy_list, "total_momentum": total_momentum_list, 'bodies_outside': bodies_outside_list, 'body_CM': center_of_mass_list}, f)
