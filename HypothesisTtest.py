import hdf5storage
import numpy as np
from matplotlib.pyplot import imshow, show, colorbar, title, savefig
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests


def create_twin_RDMs(RDM_corr, RDM_euclidean, RDM_mahalanobis):
    dims = len(RDM_corr.shape)
    data = hdf5storage.loadmat('twinset_indices.mat')
    twinset1_indices = np.subtract(data['twinset1_indices'][0], 1)
    twinset2_indices = np.subtract(data['twinset2_indices'][0], 1)

    RDM_corr_twin1 = RDM_corr
    RDM_euclidean_twin1 = RDM_euclidean
    RDM_mahalanobis_twin1 = RDM_mahalanobis

    RDM_corr_twin2 = RDM_corr_twin1.copy()
    RDM_euclidean_twin2 = RDM_euclidean_twin1.copy()
    RDM_mahalanobis_twin2 = RDM_mahalanobis_twin1.copy()

    RDM_corr_twin1 = np.delete(RDM_corr_twin1, twinset2_indices, axis=dims-2)
    RDM_corr_twin1 = np.delete(RDM_corr_twin1, twinset2_indices, axis=dims-1)

    RDM_euclidean_twin1 = np.delete(RDM_euclidean_twin1, twinset2_indices,
                                    axis=dims-2)
    RDM_euclidean_twin1 = np.delete(RDM_euclidean_twin1, twinset2_indices,
                                    axis=dims-1)
    RDM_mahalanobis_twin1 = np.delete(RDM_mahalanobis_twin1, twinset2_indices,
                                      axis=dims-2)
    RDM_mahalanobis_twin1 = np.delete(RDM_mahalanobis_twin1, twinset2_indices,
                                      axis=dims-1)
    RDM_corr_twin2 = np.delete(RDM_corr_twin2, twinset1_indices, axis=dims-2)
    RDM_corr_twin2 = np.delete(RDM_corr_twin2, twinset1_indices, axis=dims-1)
    RDM_euclidean_twin2 = np.delete(RDM_euclidean_twin2, twinset1_indices,
                                    axis=dims-2)
    RDM_euclidean_twin2 = np.delete(RDM_euclidean_twin2, twinset1_indices,
                                    axis=dims-1)
    RDM_mahalanobis_twin2 = np.delete(RDM_mahalanobis_twin2, twinset1_indices,
                                      axis=dims-2)
    RDM_mahalanobis_twin2 = np.delete(RDM_mahalanobis_twin2, twinset1_indices,
                                      axis=dims-1)
    return RDM_corr_twin1, RDM_corr_twin2, RDM_euclidean_twin1, \
           RDM_euclidean_twin2, RDM_mahalanobis_twin1, RDM_mahalanobis_twin2


def plot_RDM(RDM, timepoint):
    imshow(RDM[timepoint, :, :])
    colorbar()
    show()


def correlation(img1, img2):
    return np.corrcoef(img1, img2)[0, 1]


def plot_corr(c1, c2, c3, title):
    print("using plot corr from Hypothesis test file")
    time = np.arange(-200, 1001)
    # plt.title(title)
    plt.plot(time, c1, marker='o', markerfacecolor='blue', markersize=2,
             color='skyblue', linewidth=2, label="1-Correlation")
    plt.plot(time, c2, marker='', color='olive', linewidth=2, label="Euclidean")
    plt.plot(time, c3, marker='', color='red', linewidth=2, linestyle='dashed',
             label="Mahalanobis")
    plt.legend()
    plt.savefig('./fusion_plots/'+title)
    plt.close()


def plot_ttest(c1, c2, c3, title):
    time = np.arange(-200, 1001)
    plt.title("Mahalanobis - 1-Correlation")
    plt.plot(time, c1, color='skyblue', label="Mah - 1-Corr")
    plt.axhline(y=0.05, color='black')
    plt.savefig(title + " Mah - 1-Corr")
    plt.close()
    plt.title(" Mah - Euc")
    plt.plot(time, c2, color='green', label=" Mah - Euc")
    plt.axhline(y=0.05, color='black')
    plt.savefig(title + " Mah - Euc")
    plt.close()
    plt.title(" 1-Corr - Euc")
    plt.plot(time, c3, color='red', label="1-Corr - Euc")
    plt.axhline(y=0.05, color='black')
    plt.savefig(title + " 1-Corr - Euc")
    plt.close()


def plot_fdr(c1, c2, c3, title, t1, t2, t3):
    time = np.arange(-200, 1001)
    plt.title("Mahalanobis - 1-Correlation")
    plt.plot(time, c1, color='skyblue')
    plt.axhline(y=t1, color='red', label="FDR threshold = "+str("%.3f" % t1))
    plt.axhline(y=0.05, color='black', label="threshold = 0.050")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0., 0.5, 0.5))
    plt.savefig('./Hypothesis_ttest/'+title+" Mah - 1-Corr")
    plt.close()
    plt.title("Mahalanobis - Euclidean")
    plt.plot(time, c2, color='green')
    plt.axhline(y=t2, color='red', label="FDR threshold = "+str("%.3f" % t2))
    plt.axhline(y=0.05, color='black', label="threshold = 0.050")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0., 0.5, 0.5))
    plt.savefig('./Hypothesis_ttest/'+title + " Mah - Euc")
    plt.close()
    plt.title("1-Correlation - Euclidean")
    plt.plot(time, c3, color='darkblue')
    plt.axhline(y=t3, color='red', label="FDR threshold = "+str("%.3f" % t3))
    plt.axhline(y=0.05, color='black', label="threshold = 0.050")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0., 0.5, 0.5))
    plt.savefig('./Hypothesis_ttest/'+title + " 1-Corr - Euc")
    plt.close()


def fdr_thresh(PH1, PH2, PH3):
    idx = np.arange(1201)

    # Line to be used as cutoff
    thrline = idx * 0.029 / (len(PH1) * 1)

    # Find the largest pval, still under the line
    thr1 = max([PH1[i] for i in np.nonzero((PH1 <= thrline))[0]])
    thr2 = max([PH2[i] for i in np.nonzero((PH2 <= thrline))[0]])
    thr3 = max([PH3[i] for i in np.nonzero((PH3 <= thrline))[0]])

    # Deal with the case when all the points under the line
    # are equal to zero, and other points are above the line
    if thr1 == 0:
        thr1 = max([thrline[i] for i in np.nonzero((PH1 <= thrline))[0]])
    if thr2 == 0:
        thr2 = max([thrline[i] for i in np.nonzero((PH2 <= thrline))[0]])
    if thr3 == 0:
        thr3 = max([thrline[i] for i in np.nonzero((PH3 <= thrline))[0]])

    # Case when it does not cross
    if thr1 == None:
        thr1 = 0
    if thr2 == None:
        thr2 = 0
    if thr3 == None:
        thr3 = 0

    return thr1, thr2, thr3


def hypothesis_ttest(c1, c2, c3):
    c1 = np.asarray(c1).reshape((-1, 1201))
    c2 = np.asarray(c2).reshape((-1, 1201))
    c3 = np.asarray(c3).reshape((-1, 1201))
    PH1 = []
    PH2 = []
    PH3 = []
    for t in range(1201):
        t1, p1 = stats.ttest_1samp(c3[:, t] - c1[:, t], 0)
        PH1.append(p1)
        t2, p2 = stats.ttest_1samp(c3[:, t] - c2[:, t], 0)
        PH2.append(p2)
        t3, p3 = stats.ttest_1samp(c1[:, t] - c2[:, t], 0)
        PH3.append(p3)

    rej1, PH1_corrected = multipletests(PH1, alpha=0.05, method='fdr_bh')[:2]
    rej2, PH2_corrected = multipletests(PH2, alpha=0.05, method='fdr_bh')[:2]
    rej3, PH3_corrected = multipletests(PH3, alpha=0.05, method='fdr_bh')[:2]

    PH1_count = []
    PH2_count = []
    PH3_count = []
    for i in range(len(PH1_corrected)):
        if rej1[i]:
            PH1_count.append(i-200)
        if rej2[i]:
            PH2.append(i-200)
        if rej3[i]:
            PH3_count.append(i-200)

    return PH1_count, PH2_count, PH3_count


if __name__ == '__main__':
    c1 = []
    c2 = []
    c3 = []
    for i in range(1, 16):
        subject = "Subject" + str(i)
        RDM_corr_MEG = np.load(
            './Subjects/' + subject + '/RDM_Correlation_Final.npy')
        RDM_euclidean_MEG = np.load(
            './Subjects/' + subject + '/RDM_Euclidean_Final.npy')
        RDM_mahalanobis_MEG = np.load(
            './Subjects/' + subject + '/RDM_Mahalanobis_Final.npy')
        RDM_corr_twin1, RDM_corr_twin2, RDM_euclidean_twin1, \
        RDM_euclidean_twin2, \
        RDM_mahalanobis_twin1, RDM_mahalanobis_twin2 = create_twin_RDMs(RDM_corr_MEG,
                                                                        RDM_euclidean_MEG,
                                                                        RDM_mahalanobis_MEG)

        Correlation_corr = np.zeros(1201)
        Euclidean_corr = np.zeros(1201)
        Mahalanobis_corr = np.zeros(1201)
        for t in range(1201):
            Correlation_corr[t] = correlation(squareform(RDM_corr_twin1[t]),
                                              squareform(RDM_corr_twin2[t]))
            Euclidean_corr[t] = correlation(squareform(RDM_euclidean_twin1[t]),
                                            squareform(RDM_euclidean_twin2[t]))
            Mahalanobis_corr[t] = correlation(squareform(RDM_mahalanobis_twin1[t]),
                                              squareform(RDM_mahalanobis_twin2[t]))

        c1.append(Correlation_corr)
        c2.append(Euclidean_corr)
        c3.append(Mahalanobis_corr)

    # Average of correlation between twin set 1 and twin set 2 for all subjects
    # c1 = np.mean(np.asarray(c1).reshape((-1, 1201)), axis=0)
    # c2 = np.mean(np.asarray(c2).reshape((-1, 1201)), axis=0)
    # c3 = np.mean(np.asarray(c3).reshape((-1, 1201)), axis=0)
    # plot_corr(c1, c2, c3,"Average over all subjects")

    hypothesis_ttest(c1, c2, c3, "MEG")
