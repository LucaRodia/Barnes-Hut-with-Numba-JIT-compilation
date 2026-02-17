class Body:
	def __init__(self, mass, pos, speed = [0.0, 0.0]):
		
		self.mass = mass
		self.pos = pos
		self.speed = speed
		self.force = [0.0, 0.0]

	def apply_force(self, dt): 

		ax, ay = self.force[0]/self.mass, self.force[1]/self.mass	

		# Using Leapfrog 
		self.speed[0] += ax*dt
		self.speed[1] += ay*dt

		new_x = self.pos[0] + self.speed[0]*dt
		new_y = self.pos[1] + self.speed[1]*dt 

		self.pos = [new_x, new_y]
		self.force = [0.0, 0.0]



