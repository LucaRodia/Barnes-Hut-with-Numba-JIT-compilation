from Node import Node

class Tree:
	def __init__(self, x_min, y_min, x_max, y_max, theta, dt):
	
		self.bounds = [[x_min, y_min],[x_max, y_max]]
		self.root = Node(self.bounds)
		self.theta = theta	
		self.dt = dt
		
	def build_tree(self, bodies):
		for body in bodies:
			self.insert_body_tree(body)	
							
	def reset_tree(self):
		self.root = Node(self.bounds)
		
		
	def insert_body_tree(self, body):
		self.root.insert_body(body)
		
	def compute_force_tree(self, bodies):
		for body in bodies:
			force = self.root.compute_force(body, self.theta)
			body.force[0], body.force[1] = force[0], force[1]
			
	
	def apply_force_tree(self, bodies):
		for body in bodies:
			body.apply_force(self.dt)