from vae import vae
import numpy as np
import cv2
import theano.tensor as tt
import pymc3 as pm
import glob

class max_ent_reward_model:

    def __init__(self,fol_path='../data/*',vae_path='./logs/'):

        self.log_path = vae_path
	self.vae_model = vae()
        self.vae_model.load(path=vae_path)
	self.w = self.vae_model.w
	self.h = self.vae_model.h  
	self.Ni = 10

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
        self.bins_list = []
	count = 0
        for s in seq_list[:-1]:
            self.latent_list.append(self.vae_model.encoder.predict(s.reshape(-1,self.w,self.h,1)/255.0)[0])
	    self.bins_list.append(np.arange(s.shape[0])+count)
	    count = self.bins_list[-1][-1] 

        self.latent = np.vstack(self.latent_list)

    def train(self,fol_path='../data/*'):

        Xt = tt.as_tensor(self.latent)

        bins_t = []
        for t in self.bins_list:
            bins_t.append(tt.as_tensor(t))

        def exp(reward):
            traj_reward = pm.math.sum(reward)
            return traj_reward

        with pm.Model() as reward_model:
    
            l = pm.Gamma("l", alpha=2.0, beta=0.5)
    
            cov_func = pm.gp.cov.Matern32(self.latent.shape[1], ls=l)
    
            Xu = pm.gp.util.kmeans_inducing_points(self.Ni, self.latent)
    
            sig = pm.HalfCauchy("sig", beta=np.ones((self.latent.shape[0],)),shape=self.latent.shape[0])
    
            gp = pm.gp.MarginalSparse(cov_func=cov_func)
    
            f = gp.marginal_likelihood('reward', Xt, Xu, shape=self.latent.shape[0], y=None, noise=sig, is_observed=False)
  
            exp_list = []
            for i,bins in enumerate(bins_t):
                exp_list.append(pm.DensityDist('me_%d'%i, exp, observed={'reward':f[bins]}))

            inference = pm.ADVI()
        approx = inference.fit(1000,obj_optimizer=pm.adam(learning_rate=0.1))
    
        trace = approx.sample(5000)
        l = np.mean(trace['l'])
        sig =  np.mean(trace['sig'])
        reward = np.mean(trace['reward'],axis=0)

        np.savetxt('./logs/l_me.txt',np.array([l]))
        np.savetxt('./logs/sig_me.txt',np.array([sig]))
        np.savetxt('./logs/reward_me.txt',reward)

	print('Saved trained reward parameters')
	return l,sig,reward

    def build_map_model(self,load=True):

        if load:
            l = np.genfromtxt(self.log_path+'l_me.txt')
            sig = np.genfromtxt(self.log_path+'sig_me.txt')
            reward = np.genfromtxt(self.log_path+'reward_me.txt')
        else:
            l, sig, reward = self.train()

        Xt = tt.as_tensor(self.latent)
	
        with pm.Model() as map_model:
    
            cov_func = pm.gp.cov.Matern32(self.latent.shape[1], ls=l)
    
            Xu = pm.gp.util.kmeans_inducing_points(self.Ni, self.latent)
    
            self.gp = pm.gp.MarginalSparse(cov_func=cov_func)
    
            y = self.gp.marginal_likelihood('reward', Xt, Xu, shape=self.latent.shape[0], y=reward, noise=sig)

if __name__ == '__main__':
    me_reward = max_ent_reward_model()
    me_reward.train()
