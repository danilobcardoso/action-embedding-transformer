import numpy as np
from skeleton_models import ntu_rgbd, ntu_ss_1, ntu_ss_2, ntu_ss_3
from render import save_animation

sample = np.load('ntu_sample.npy')
save_animation(sample - sample[0,0], ntu_rgbd, 'teste.gif')