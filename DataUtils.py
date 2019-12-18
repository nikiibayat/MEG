import numpy as np
import scipy.io as sio


MEG = np.load('./Subjects/Subject' + str(1) + '/Magnet_Normalized_RDM_Euclidean_Final.npy')
fMRI0 = np.load('./fMRI_RDMs/rdm_cor_no_nn/' + str(1) + '_0.npy')
fMRI1 = np.load('./fMRI_RDMs/rdm_cor_no_nn/' + str(1) + '_1.npy')

directory = 'C:/Users/nbayat5/Desktop/Fusion/'
sio.savemat(directory + 'MEGsubject1.mat', {'vect': MEG})
sio.savemat(directory + 'fMRI0subject1.mat', {'vect': fMRI0})
sio.savemat(directory + 'fMRI1subject1.mat', {'vect': fMRI1})
