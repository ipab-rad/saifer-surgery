from vae import vae
import numpy as np
import cv2
import theano.tensor as tt
import pymc3 as pm
import glob

class cosine_reward_model:
	
	def __init__(self,fol_path='../data/*',vae_path='./logs/'):

		self.log_path = vae_path
		self.vae_model = vae()
		self.vae_model.load(path=vae_path)
		self.w = self.vae_model.w
		self.h = self.vae_model.h  
		
		folder = sorted(glob.glob(fol_path))[0]
		im_file = sorted(glob.glob(folder+'/*.jpg'))[-1]
		print('Loading '+im_file)
		im = np.mean(cv2.resize(cv2.imread(im_file)[180:700,500:1020,:],(self.w,self.h)),axis=-1)
		self.target_latent = self.vae_model.encoder.predict(im.reshape(-1,self.w,self.h,1)/255.0)[0]

	def cosine_sim(self,latent):
		return np.dot(self.target_latent.ravel(), latent.ravel())/(np.linalg.norm(self.target_latent) * np.linalg.norm(latent))
