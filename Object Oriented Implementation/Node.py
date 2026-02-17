from math import sqrt
import numpy as np
from numba import njit

class Node:
	def __init__(self, bounds):

		#per indicare il quadrante uso i vertici opposti rispetto all'origine, in basso a sinistra e in alto a destra
		self.x_min, self.y_min = bounds[0]
		self.x_max, self.y_max = bounds[1]
		
		self.mass = 0
		self.center_of_mass = [0.0, 0.0]   #0.0 lo mette come float
		
		self.body = None  #None/puntatore (8-bit) a un corpo (oggetto)
			
		self.children = None
		
	def is_leaf(self): #non dipende dal tipo di self.children (dict, list, etc...)
		return self.children is None
		
	def is_empty(self):	
		return self.children is None and self.body is None

	def subdivide(self): 
		x_mid = (self.x_min + self.x_max)/2
		y_mid = (self.y_min + self.y_max)/2
		
		self.children ={
			"SW": Node([(self.x_min, self.y_min), (x_mid, y_mid)]), #SW
			"NW": Node([(self.x_min, y_mid), (x_mid, self.y_max)]), #NW
			"NE": Node([(x_mid, y_mid), (self.x_max, self.y_max)]), #NE
			"SE": Node([(x_mid, self.y_min), (self.x_max, y_mid)])  #SE
		}
	
	def get_quadrant(self, pos):
		x_mid, y_mid = (self.x_min + self.x_max)/2, (self.y_min + self.y_max)/2
		x, y = pos
		
		if y > y_mid:
			return "NE" if x>x_mid else "NW"
		else:
			return "SE" if x>x_mid else "SW"
		

	def insert_body(self, body, depth = 0):
		if depth >= 30: 
			if self.body is None:
            # Trasforma in nodo "multi-corpo" (non suddiviso)
				self.body = body
				self.total_mass = body.mass
				self.center_of_mass = list(body.pos)
			else:
				self.total_mass += body.mass
				self.update_params(body.mass, body.pos)
				self.body = None  # segna come non-foglia senza children
			return
		
		#il nodo è vuoto (quindi anche una foglia)
		if self.is_empty(): 
			self.body = body
			self.mass = body.mass
			self.center_of_mass = body.pos

		#the node is a leaf
		elif self.body != None and self.children == None: #questa è equivalente a self.is_leaf()?

			Q1 = self.get_quadrant(self.center_of_mass) #it's the only mass there
			Q2 = self.get_quadrant(body.pos)
			
			self.subdivide()
			self.children[Q1].insert_body(self.body, depth + 1)
			self.children[Q2].insert_body(body, depth + 1)
			
			self.update_params(body.mass, body.pos)
			
			self.body = None

		#non è una foglia. è un nodo interno
		elif self.is_leaf() == False: #qui può servire self.is_leaf() così non devo mettere ==dict
			self.update_params(body.mass, body.pos)
			Q1 = self.get_quadrant(body.pos)
			
			self.children[Q1].insert_body(body, depth + 1)

		
	def update_params(self, mass_body, pos_body):
		x_body, y_body = pos_body
		total_mass = self.mass + mass_body
		
		#Weighted sum. New COM. 
		new_x = (self.mass * self.center_of_mass[0] + mass_body * x_body)/total_mass
		new_y = (self.mass * self.center_of_mass[0] + mass_body * y_body)/total_mass
		
		self.center_of_mass = [new_x, new_y]
		self.mass = total_mass
		
	def get_distance(self, x_body, y_body):
		#this will get the distance between node's center of mass and body position
		dx = self.center_of_mass[0] - x_body
		dy = self.center_of_mass[1] - y_body
		
		d = sqrt(dx**2 + dy**2)
		return d 
	
	def compute_force(self, body, theta): 		
		s = max(self.x_max - self.x_min, self.y_max - self.y_min)   #meglio, funziona anche se non è un quadrato 
		x_body, y_body = body.pos  #CHECK
			
		d = self.get_distance(x_body, y_body)
		
		if self.is_empty():
			return [0.0, 0.0]
		
		elif self.is_leaf() and self.body == body: #unico punto dentro al nodo --> no forza
			return [0.0, 0.0] 	    #ritorniamo [0.0, 0.0] così non ho problemi di mancato return

		elif self.is_leaf(): #implicit "and self.body != body. Voglio evitare il caso in cui il criterio non è rispettato ma la foglia non ha figli --> NoneType not iterable
			force = force_between_bodies(self.center_of_mass, self.mass, body.pos, body.mass)
			return force
		
		elif s/d < theta:
			force = force_between_bodies(self.center_of_mass, self.mass, body.pos, body.mass)
			return force
		
		else: 
			force = [0.0, 0.0]
			for child in self.children:
				child_force = self.children[child].compute_force(body, theta)
				force[0] = force[0] + child_force[0]
				force[1] = force[1] + child_force[1]
			return force
	
@njit(fastmath=True)
def force_between_bodies(pos0, mass0, pos1, mass1):
    # Costanti hardcoded per performance o passate come argomenti
    k = 200.0 
    epsilon = 2.0 # Aumentato per stabilità
    
    # Calcolo vettoriale rapido
    dx = pos0[0] - pos1[0]
    dy = pos0[1] - pos1[1]

    dist_sq = dx*dx + dy*dy
    dist_soft = dist_sq + epsilon*epsilon
    
    inv_dist_cube = dist_soft**(-1.5)
    
    factor = k * mass0 * mass1 * inv_dist_cube
    
    return factor * dx, factor * dy # Ritorna tupla o array