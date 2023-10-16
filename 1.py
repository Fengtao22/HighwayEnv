import gymnasium as gym
import highway_env
#highway_env.register_highway_envs()
import time, warnings
warnings.filterwarnings("ignore") 

from matplotlib import pyplot as plt



env = gym.make('highway-v0', render_mode='human')
env.configure({
	"observation": {
		"type": "GrayscaleObservation",
		"observation_shape": (256, 64),
		"stack_size": 4,
		"weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
		"scaling": 5,
	},
})

obs, info = env.reset()
terminated = truncated = False
steps = 0
while not (terminated or truncated):
	env.render()
	#print('type of obs: ', type(obs), obs[0].shape, obs[1].shape)
	#fig, axes = plt.subplots(ncols=4, figsize=(12, 5))
	#for i, ax in enumerate(axes.flat):
	#	ax.imshow(obs[i, ...].T, cmap=plt.get_cmap('gray'))
	#plt.show()

	print(obs[1])

	plt.figure()
	plt.imshow(obs[0][-1].T)
	plt.savefig("0_{}.png".format(steps)) 
	plt.show()
	print('Current frame saved...')	
	action = 2 #env.action_space.sample()  ### 0 up, 1 idle, 2 down
	obs, reward, terminated, truncated, info = env.step(action)
	steps += 1
	#time.sleep(20)
env.close()



'''
import pygame

class Screen(pygame.sprite.Sprite):
	def __init__(self, image, location):
		pygame.sprite.Sprite.__init__(self)
		self.image = image
		self.pos = location
		self.rect = self.image.get_rect(center = self.pos)	## create a rectangle for the Surface centered at a given position.

	def rotate(self, angle, camera):
		self.image = pygame.transform.rotate(self.image, angle) # Rotate the image
		offset = Vector2(camera)
		rotated_offset = offset.rotate(-angle)  # Rotate the offset vector.
		# Add the offset vector to the center/pivot point to shift the rect.
		###### careful about the plus or minus sign
		self.rect = self.image.get_rect(center=self.pos+rotated_offset)
		self.mask = pygame.mask.from_surface(self.image)

'''