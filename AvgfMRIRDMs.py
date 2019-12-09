import numpy as np
from matplotlib.pyplot import imshow, close, colorbar, title, show
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy import stats

def plot_RDM(RDM, title, dir):
    temp_rdm = squareform(stats.rankdata(squareform(RDM), method='average'))
    imshow(temp_rdm / np.max(temp_rdm))
    # colorbar()
    # plt.title(title)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.savefig('./saved_rdm/'+dir+'/'+title)
    close()

def plot_mds(RDM, title, dir):
    RDM = squareform(stats.rankdata(squareform(RDM), method='average'))
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0)  # Create the MDS object
    results = mds.fit(RDM)  # Use the dissimilarity matrix
    coords = results.embedding_
    leg = []
    plt.figure(figsize=(4, 4))
    leg.append(plt.scatter(coords[0:27, 0], coords[0:27, 1],marker='o', s=15, c='brown'))
    leg.append(plt.scatter(coords[28:63, 0], coords[28:63, 1],marker='o', s=15, c='blue'))
    leg.append(plt.scatter(coords[64:99, 0], coords[64:99, 1],marker='o', s=15, c='green'))
    leg.append(plt.scatter(coords[100:123, 0], coords[100:123, 1],marker='o', s=15, c='orange'))
    leg.append(plt.scatter(coords[124:155, 0], coords[124:155, 1],marker='o', s=15, c='red'))
    # plt.legend(leg[0:5], ['Animals', 'Objects', 'Scenes', 'People','Faces'], loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5), fontsize='xx-small',markerscale = 0.5)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.savefig('./saved_mds/'+dir+'/'+title)
    plt.close()

# fMRI_RDM_corr_loc0_avg = []
# fMRI_RDM_corr_loc1_avg = []
# fMRI_RDM_euc_loc0_avg = []
# fMRI_RDM_euc_loc1_avg = []
# fMRI_RDM_mah_loc0_avg = []
# fMRI_RDM_mah_loc1_avg = []
#
# for i in range(1, 16):
#     RDM_corr_fMRI_loc0 = np.load(
#                     './fMRI_RDMs/rdm_cor_no_nn/' + str(i) + '_0.npy')
#     fMRI_RDM_corr_loc0_avg.append(RDM_corr_fMRI_loc0)
#     RDM_euclidean_fMRI_loc0 = np.load(
#                     './fMRI_RDMs/rdm_euclidean_no_nn/' + str(i) + '_0.npy')
#     fMRI_RDM_euc_loc0_avg.append(RDM_euclidean_fMRI_loc0)
#     RDM_mahalanobis_fMRI_loc0 = np.load(
#                     './fMRI_RDMs/rdm_mahalanobis_nn/' + str(i) + '_0.npy')
#     fMRI_RDM_mah_loc0_avg.append(RDM_mahalanobis_fMRI_loc0)
#
#     RDM_corr_fMRI_loc1 = np.load(
#                 './fMRI_RDMs/rdm_cor_no_nn/' + str(i) + '_1.npy')
#     fMRI_RDM_corr_loc1_avg.append(RDM_corr_fMRI_loc1)
#     RDM_euclidean_fMRI_loc1 = np.load(
#                 './fMRI_RDMs/rdm_euclidean_no_nn/' + str(i) + '_1.npy')
#     fMRI_RDM_euc_loc1_avg.append(RDM_euclidean_fMRI_loc1)
#     RDM_mahalanobis_fMRI_loc1 = np.load(
#                 './fMRI_RDMs/rdm_mahalanobis_nn/' + str(i) + '_1.npy')
#     fMRI_RDM_mah_loc1_avg.append(RDM_mahalanobis_fMRI_loc1)
#
# fMRI_RDM_corr_loc0_avg = np.asarray(fMRI_RDM_corr_loc0_avg)
# fMRI_RDM_corr_loc1_avg = np.asarray(fMRI_RDM_corr_loc1_avg)
# fMRI_RDM_euc_loc0_avg = np.asarray(fMRI_RDM_euc_loc0_avg)
# fMRI_RDM_euc_loc1_avg = np.asarray(fMRI_RDM_euc_loc1_avg)
# fMRI_RDM_mah_loc0_avg = np.asarray(fMRI_RDM_mah_loc0_avg)
# fMRI_RDM_mah_loc1_avg = np.asarray(fMRI_RDM_mah_loc1_avg)
#
# np.save('./fMRI_RDMs/rdm_cor_no_nn/corr_loc0_avg',np.mean(
# fMRI_RDM_corr_loc0_avg, axis=0))
# np.save('./fMRI_RDMs/rdm_cor_no_nn/corr_loc1_avg',np.mean(
# fMRI_RDM_corr_loc1_avg, axis=0))
# np.save('./fMRI_RDMs/rdm_euclidean_no_nn/euc_loc0_avg',np.mean(
# fMRI_RDM_euc_loc0_avg, axis=0))
# np.save('./fMRI_RDMs/rdm_euclidean_no_nn/euc_loc1_avg',np.mean(
# fMRI_RDM_euc_loc1_avg, axis=0))
# np.save('./fMRI_RDMs/rdm_mahalanobis_nn/mah_loc0_avg',np.mean(
# fMRI_RDM_mah_loc0_avg, axis=0))
# np.save('./fMRI_RDMs/rdm_mahalanobis_nn/mah_loc1_avg',np.mean(
# fMRI_RDM_mah_loc1_avg, axis=0))

# fMRI_RDM_corr_loc0_avg = np.load('./fMRI_RDMs/rdm_cor_no_nn/corr_loc0_avg.npy')
# fMRI_RDM_corr_loc1_avg = np.load('./fMRI_RDMs/rdm_cor_no_nn/corr_loc1_avg.npy')
# fMRI_RDM_euc_loc0_avg = np.load(
#     './fMRI_RDMs/rdm_euclidean_no_nn/euc_loc0_avg.npy')
# fMRI_RDM_euc_loc1_avg = np.load(
#     './fMRI_RDMs/rdm_euclidean_no_nn/euc_loc1_avg.npy')
# fMRI_RDM_mah_loc0_avg = np.load(
#     './fMRI_RDMs/rdm_mahalanobis_nn/mah_loc0_avg.npy')
# fMRI_RDM_mah_loc1_avg = np.load(
#     './fMRI_RDMs/rdm_mahalanobis_nn/mah_loc1_avg.npy')



# MEG_RDM_corr_avg = []
# MEG_RDM_euc_avg = []
# MEG_RDM_mah_avg = []
#
# for i in range(1, 16):
#     subject = "Subject" + str(i)
#     RDM_corr_MEG = np.load(
#             './Subjects/' + subject + '/RDM_Correlation_Final.npy')
#     MEG_RDM_corr_avg.append(RDM_corr_MEG)
#     RDM_euclidean_MEG = np.load(
#             './Subjects/' + subject + '/RDM_Euclidean_Final.npy')
#     MEG_RDM_euc_avg.append(RDM_euclidean_MEG)
#     RDM_mahalanobis_MEG = np.load(
#             './Subjects/' + subject + '/RDM_Mahalanobis_Final.npy')
#     MEG_RDM_mah_avg.append(RDM_mahalanobis_MEG)
#
# MEG_RDM_corr_avg = np.asarray(MEG_RDM_corr_avg)
# MEG_RDM_euc_avg = np.asarray(MEG_RDM_euc_avg)
# MEG_RDM_mah_avg = np.asarray(MEG_RDM_mah_avg)
#
#
# np.save('./Subjects/MEG_Corr_avg_oversubjects',np.mean(MEG_RDM_corr_avg, axis=0))
# np.save('./Subjects/MEG_Euc_avg_oversubjects',np.mean(MEG_RDM_euc_avg, axis=0))
# np.save('./Subjects/MEG_Mah_avg_oversubjects',np.mean(MEG_RDM_mah_avg, axis=0))

# MEG_RDM_corr_avg = np.load('./Subjects/MEG_Corr_avg_oversubjects.npy')
# MEG_RDM_euc_avg = np.load('./Subjects/MEG_Euc_avg_oversubjects.npy')
# MEG_RDM_mah_avg = np.load('./Subjects/MEG_Mah_avg_oversubjects.npy')
#
# for i in range(0, 1201, 50):
#     plot_mds(MEG_RDM_corr_avg[i, :, :], "1-Correlation "+str(i-200),"corr")
#     plot_mds(MEG_RDM_euc_avg[i, :, :], "Euclidean "+str(i-200),"euc")
#     plot_mds(MEG_RDM_mah_avg[i, :, :], "Mahalanobis "+str(i-200),"mah")
#
#     plot_RDM(MEG_RDM_corr_avg[i, :, :], "1-Correlation "+str(i-200),"corr")
#     plot_RDM(MEG_RDM_euc_avg[i, :, :], "Euclidean "+str(i-200),"euc")
#     plot_RDM(MEG_RDM_mah_avg[i, :, :], "Mahalanobis "+str(i-200),"mah")
