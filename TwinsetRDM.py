import hdf5storage
import numpy as np
from matplotlib.pyplot import imshow, show, colorbar, title, savefig
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt


def create_twin_RDMs():
    data = hdf5storage.loadmat('twinset_indices.mat')
    twinset1_indices = np.subtract(data['twinset1_indices'][0], 1)
    twinset2_indices = np.subtract(data['twinset2_indices'][0], 1)

    RDM_corr_twin1 = np.load('./Saved_RDMs/RDM_Correlation_Final.npy')
    RDM_euclidean_twin1 = np.load('./Saved_RDMs/RDM_Euclidean_Final.npy')
    RDM_mahalanobis_twin1 = np.load('./Saved_RDMs/RDM_Mahalanobis_Final.npy')

    RDM_corr_twin2 = RDM_corr_twin1.copy()
    RDM_euclidean_twin2 = RDM_euclidean_twin1.copy()
    RDM_mahalanobis_twin2 = RDM_mahalanobis_twin1.copy()

    RDM_corr_twin1 = np.delete(RDM_corr_twin1, twinset2_indices, axis=1)
    RDM_corr_twin1 = np.delete(RDM_corr_twin1, twinset2_indices, axis=2)
    RDM_euclidean_twin1 = np.delete(RDM_euclidean_twin1, twinset2_indices,
                                    axis=1)
    RDM_euclidean_twin1 = np.delete(RDM_euclidean_twin1, twinset2_indices,
                                    axis=2)
    RDM_mahalanobis_twin1 = np.delete(RDM_mahalanobis_twin1, twinset2_indices,
                                      axis=1)
    RDM_mahalanobis_twin1 = np.delete(RDM_mahalanobis_twin1, twinset2_indices,
                                      axis=2)

    RDM_corr_twin2 = np.delete(RDM_corr_twin2, twinset1_indices, axis=1)
    RDM_corr_twin2 = np.delete(RDM_corr_twin2, twinset1_indices, axis=2)
    RDM_euclidean_twin2 = np.delete(RDM_euclidean_twin2, twinset1_indices,
                                    axis=1)
    RDM_euclidean_twin2 = np.delete(RDM_euclidean_twin2, twinset1_indices,
                                    axis=2)
    RDM_mahalanobis_twin2 = np.delete(RDM_mahalanobis_twin2, twinset1_indices,
                                      axis=1)
    RDM_mahalanobis_twin2 = np.delete(RDM_mahalanobis_twin2, twinset1_indices,
                                      axis=2)
    return RDM_corr_twin1, RDM_corr_twin2, RDM_euclidean_twin1, \
           RDM_euclidean_twin2, RDM_mahalanobis_twin1, RDM_mahalanobis_twin2


def plot_RDM(RDM, timepoint):
    imshow(RDM[timepoint, :, :])
    colorbar()
    show()


def correlation(img1, img2):
    return np.corrcoef(img1, img2)[0, 1]


def plot_corr(c1, c2, c3):
    time = np.arange(1201)
    plt.plot(time, c1, marker='o', markerfacecolor='blue', markersize=2,
             color='skyblue', linewidth=2, label="1-Correlation")
    plt.plot(time, c2, marker='', color='olive', linewidth=2, label="Euclidean")
    plt.plot(time, c3, marker='', color='red', linewidth=2, linestyle='dashed',
             label="Mahalanobis")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    RDM_corr_twin1, RDM_corr_twin2, RDM_euclidean_twin1, RDM_euclidean_twin2, \
    RDM_mahalanobis_twin1, RDM_mahalanobis_twin2 = create_twin_RDMs()
    # plot_RDM(RDM_corr_twin1,100)

    RDM_Corr_twin1_flatten = np.zeros((1201, 3003))
    RDM_Euc_twin1_flatten = np.zeros((1201, 3003))
    RDM_Mah_twin1_flatten = np.zeros((1201, 3003))
    RDM_Corr_twin2_flatten = np.zeros((1201, 3003))
    RDM_Euc_twin2_flatten = np.zeros((1201, 3003))
    RDM_Mah_twin2_flatten = np.zeros((1201, 3003))

    for t in range(1201):
        RDM_Corr_twin1_flatten[t] = squareform(RDM_corr_twin1[t])
        RDM_Euc_twin1_flatten[t] = squareform(RDM_euclidean_twin1[t])
        RDM_Mah_twin1_flatten[t] = squareform(RDM_mahalanobis_twin1[t])
        RDM_Corr_twin2_flatten[t] = squareform(RDM_corr_twin2[t])
        RDM_Euc_twin2_flatten[t] = squareform(RDM_euclidean_twin2[t])
        RDM_Mah_twin2_flatten[t] = squareform(RDM_mahalanobis_twin2[t])

    Correlation_corr = np.zeros(1201)
    Euclidean_corr = np.zeros(1201)
    Mahalanobis_corr = np.zeros(1201)
    for t in range(1201):
        Correlation_corr[t] = correlation(RDM_Corr_twin1_flatten[t],
                                          RDM_Corr_twin2_flatten[t])
        Euclidean_corr[t] = correlation(RDM_Euc_twin1_flatten[t],
                                        RDM_Euc_twin2_flatten[t])
        Mahalanobis_corr[t] = correlation(RDM_Mah_twin1_flatten[t],
                                          RDM_Mah_twin2_flatten[t])

    plot_corr(Correlation_corr, Euclidean_corr, Mahalanobis_corr)