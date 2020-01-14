import numpy as np 

def cosine_sim(latent, target_latent):
    return np.dot(target_latent, latent)/(np.linalg.norm(target_latent) * np.linalg.norm(latent))