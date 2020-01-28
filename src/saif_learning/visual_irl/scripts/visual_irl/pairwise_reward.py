from vae import vae
import numpy as np
import cv2
import theano.tensor as tt
import pymc3 as pm
import glob
import os.path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import pickle

class pairwise_reward_model:

	def __init__(self,vae_path='./logs/'):
		
		self.log_path = vae_path
		self.vae_model = vae()
		print('Loading model '+vae_path)
		self.vae_model.load(path=vae_path)
		self.vae_model.encoder._make_predict_function()
		self.w = self.vae_model.w
		self.h = self.vae_model.h  
		self.Ni = 100

	def generate_pairs(self,latent_list):

		latent = np.vstack(latent_list)
		W = np.arange(latent.shape[0]).astype(int)
		Nt = 50000
   
		start_bins = [0]
		for j in range(len(latent_list)):
			start_bins.append(start_bins[-1]+latent_list[j].shape[0])

		G_list = []
		for i in range(Nt):
			 demo_idx = np.random.randint(len(latent_list))
	
			 traj = latent_list[demo_idx]
	
			 b1 = np.random.randint(traj.shape[0])
			 b2 = np.random.randint(traj.shape[0])
	
			 if b1 >= b2:
				 G = np.array([b1 + start_bins[demo_idx],b2 + start_bins[demo_idx]])
			 else:
				 G = np.array([b2 + start_bins[demo_idx],b1 + start_bins[demo_idx]])
			 # Generate some comparisons
			 G_list.append(G)
		G = np.vstack(G_list).astype(int)

		return G

	def train(self,fol_path='../data/*'):

		fol_list = glob.glob(fol_path)
		print(fol_list)
		seq_list = []
		for fol in fol_list:
			f_list = glob.glob(fol+'/*.jpg')
			im_list = []
			for f in sorted(f_list):
				#Crop to ultrasound active area
				im = np.mean(cv2.resize(cv2.imread(f)[180:700,500:1020,:],(self.w,self.h)),axis=-1)
				im_list.append(im)
			seq_list.append(np.array(im_list))

		# Get latent states
		self.latent_list = []
		for s in seq_list[:-1]:
			self.latent_list.append(self.vae_model.encoder.predict(s.reshape(-1,self.w,self.h,1)/255.0)[0]) 
		self.latent = np.vstack(self.latent_list)
		
		np.savetxt(self.log_path+'latent.txt',self.latent)
		
		#Generate training pairs
		print('Generating training pairs')  
		G = self.generate_pairs(self.latent_list)
		W = np.arange(self.latent.shape[0]).astype(int)

		Gt = tt.as_tensor(G)
		W = W.astype(int)
		Xt = tt.as_tensor(self.latent)

		with pm.Model() as reward_model:
	
			l = pm.Gamma("l", alpha=2.0, beta=0.5)
	
			cov_func = pm.gp.cov.Matern32(self.latent.shape[1], ls=l)
	
			Xu = pm.gp.util.kmeans_inducing_points(self.Ni, self.latent)
	
			sig = pm.HalfCauchy("sig", beta=np.ones((self.latent.shape[0],)),shape=self.latent.shape[0])
	
			gp = pm.gp.MarginalSparse(cov_func=cov_func)
	
			f = gp.marginal_likelihood('reward', Xt, Xu, shape=self.latent.shape[0], y=None, noise=sig, is_observed=False)
	
			diff = f[Gt[:,0]] - f[Gt[:,1]]
	
			p = pm.math.sigmoid(diff)
	
			wl = pm.Bernoulli('observed wl', p=p, observed=np.ones((G.shape[0],)),total_size=self.latent.shape[0])
			inference = pm.ADVI()
	
			train_probs = inference.approx.sample_node(p)


		train_accuracy = (train_probs>0.5).mean(-1)
		eval_tracker = pm.callbacks.Tracker(train_accuracy=train_accuracy.eval)
		approx = inference.fit(1000,obj_optimizer=pm.adam(learning_rate=0.1), callbacks=[eval_tracker]);

		trace = approx.sample(5000)
		l = np.mean(trace['l'])
		sig =  np.mean(trace['sig'])
		reward = np.mean(trace['reward'],axis=0)
		np.savetxt('./logs/l.txt',np.array([l]))
		np.savetxt('./logs/sig.txt',np.array([sig]))
		np.savetxt('./logs/reward.txt',reward)

		print('Saved trained reward parameters')
		return l,sig,reward

	def build_map_model(self,load=True):

		if load:
			l = np.genfromtxt(self.log_path+'l.txt')
			sig = np.genfromtxt(self.log_path+'sig.txt')
			reward = np.genfromtxt(self.log_path+'reward.txt')
			self.latent = np.genfromtxt(self.log_path+'latent.txt')
			filename = self.log_path+'pairwise_model.sav'
			self.gp = pickle.load(open(filename, 'rb'))
			print(self.latent.shape)
		else:
			l, sig, reward = self.train()
		
			self.gp = GaussianProcessRegressor(kernel=Matern(length_scale=l),alpha=sig)
			self.gp.fit(self.latent,reward)
			filename = self.log_path+'pairwise_model.sav'
			pickle.dump(self.gp, open(filename, 'wb'))
	

if __name__ == '__main__':
	pair_reward = pairwise_reward_model()
	pair_reward.train()
