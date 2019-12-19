import HypothesisTtest as ttest
from scipy.stats import spearmanr
import scipy.io as sio
from TwinsetRDM import *
from scipy import stats
from statsmodels.stats.multitest import multipletests


def plot_comparison(c1, c2, c3, t1, t2, t3, title):
    time = np.arange(-200, 1001)
    # plt.title(title)
    plt.plot(time, c1, color='#66c2a5', label="Mahalanobis - 1-Correlation")
    plt.plot(time, c2, color='#fc8d62', label="Mahalanobis - Euclidean")
    plt.plot(time, c3, color='#8da0cb', linestyle='dashed',
             label="1-Correlation - Euclidean")
    horiz_line_data = np.array([-0.09 for i in range(len(t1))])
    plt.scatter(t1, horiz_line_data, color='#66c2a5', s=5)
    horiz_line_data = np.array([-0.1 for i in range(len(t2))])
    plt.scatter(t2, horiz_line_data, color='#fc8d62', s=5)
    horiz_line_data = np.array([-0.07 for i in range(len(t3))])
    plt.scatter(t3, horiz_line_data, color='#8da0cb', s=5)

    plt.legend(loc="upper right", prop={'size': 6})
    plt.savefig('./Hypothesis_ttest/' + title)
    plt.close()


def twin_ttest(c1, c2):
    c1 = np.asarray(c1).reshape((-1, 1201))
    c2 = np.asarray(c2).reshape((-1, 1201))
    PH1 = []

    for t in range(1201):
        t1, p1 = stats.ttest_1samp(c2[:, t] - c1[:, t], 0)
        PH1.append(p1)

    rej1, PH1_corrected = multipletests(PH1, alpha=0.05, method='fdr_bh')[:2]

    PH1_count = []
    for i in range(len(PH1_corrected)):
        if rej1[i]:
            PH1_count.append(i-200)

    return PH1_count


def plot_twin_corr(c1, c2, t1, title):
    time = np.arange(-200, 1001)
    plt.title(title)
    plt.plot(time, c1, color='#66c2a5', label="TwinSet1")
    plt.plot(time, c2, color='#fc8d62', label="TwinSet2")

    horiz_line_data = np.array([-0.05 for i in range(len(t1))])
    plt.scatter(t1, horiz_line_data, color='red', s=5)

    plt.legend()
    plt.savefig('./fusion_plots/' + title)
    plt.close()


def fusion():
    total_corr_loc0 = []
    total_corr_loc1 = []
    total_euc_loc0 = []
    total_euc_loc1 = []
    total_mah_loc0 = []
    total_mah_loc1 = []

    # Load fMRI
    try:
        RDM_corr_fMRI_loc0 = np.load(
            './fMRI_RDMs/rdm_cor_no_nn/corr_loc0_avg.npy')
        RDM_euclidean_fMRI_loc0 = np.load(
            './fMRI_RDMs/rdm_euclidean_no_nn/euc_loc0_avg.npy')
        RDM_mahalanobis_fMRI_loc0 = np.load(
            './fMRI_RDMs/rdm_mahalanobis_nn/mah_loc0_avg.npy')

        RDM_corr_fMRI_loc1 = np.load(
            './fMRI_RDMs/rdm_cor_no_nn/corr_loc1_avg.npy')
        RDM_euclidean_fMRI_loc1 = np.load(
            './fMRI_RDMs/rdm_euclidean_no_nn/euc_loc1_avg.npy')
        RDM_mahalanobis_fMRI_loc1 = np.load(
            './fMRI_RDMs/rdm_mahalanobis_nn/mah_loc1_avg.npy')

    except:
        print("Error in loading average fMRI RDMs")

    for i in range(1, 16):
        subject = "Subject" + str(i)
        print(subject)
        # Load MEG
        try:
            RDM_corr_MEG = np.load(
                './Subjects/' + subject +
                '/Normalized_RDM_Correlation_Final.npy')
            RDM_euclidean_MEG = np.load(
                './Subjects/' + subject +
                '/Normalized_RDM_Euclidean_Final.npy')
            RDM_mahalanobis_MEG = np.load(
                './Subjects/' + subject +
                '/RDM_Mahalanobis_Final.npy')

        except:
            print("Error in Loading RDMs")

        corr_loc0 = []
        corr_loc1 = []
        euc_loc0 = []
        euc_loc1 = []
        mah_loc0 = []
        mah_loc1 = []

        for t in range(1201):
            corr_loc0.append(spearmanr(squareform(RDM_corr_MEG[t]),
                squareform(RDM_corr_fMRI_loc0))[0])
            corr_loc1.append(spearmanr(squareform(RDM_corr_MEG[t]),
                squareform(RDM_corr_fMRI_loc1))[0])

            euc_loc0.append(spearmanr(squareform(RDM_euclidean_MEG[t]),
                squareform(RDM_euclidean_fMRI_loc0))[0])
            euc_loc1.append(spearmanr(squareform(RDM_euclidean_MEG[t]),
                squareform(RDM_euclidean_fMRI_loc1))[0])

            mah_loc0.append(spearmanr(squareform(RDM_mahalanobis_MEG[t]),
                squareform(RDM_mahalanobis_fMRI_loc0))[0])
            mah_loc1.append(spearmanr(squareform(RDM_mahalanobis_MEG[t]),
                squareform(RDM_mahalanobis_fMRI_loc1))[0])

        total_corr_loc0.append(corr_loc0)
        total_corr_loc1.append(corr_loc1)
        total_euc_loc0.append(euc_loc0)
        total_euc_loc1.append(euc_loc1)
        total_mah_loc0.append(mah_loc0)
        total_mah_loc1.append(mah_loc1)

    directory = 'C:/Users/nbayat5/Desktop/Fusion/'
    sio.savemat(directory + 'total_corr_loc0.mat', {'vect': total_corr_loc0})
    sio.savemat(directory + 'total_corr_loc1.mat', {'vect': total_corr_loc1})
    sio.savemat(directory + 'total_euc_loc0.mat', {'vect': total_euc_loc0})
    sio.savemat(directory + 'total_euc_loc1.mat', {'vect': total_euc_loc1})
    sio.savemat(directory + 'total_mah_loc0.mat', {'vect': total_mah_loc0})
    sio.savemat(directory + 'total_mah_loc1.mat', {'vect': total_mah_loc1})

    # save plot average over subjects per fMRI location

    avg_corr_loc0 = np.mean(np.asarray(total_corr_loc0).reshape((-1, 1201)),
        axis=0)
    avg_corr_loc1 = np.mean(np.asarray(total_corr_loc1).reshape((-1, 1201)),
        axis=0)
    avg_euc_loc0 = np.mean(np.asarray(total_euc_loc0).reshape((-1, 1201)),
        axis=0)
    avg_euc_loc1 = np.mean(np.asarray(total_euc_loc1).reshape((-1, 1201)),
        axis=0)
    avg_mah_loc0 = np.mean(np.asarray(total_mah_loc0).reshape((-1, 1201)),
        axis=0)
    avg_mah_loc1 = np.mean(np.asarray(total_mah_loc1).reshape((-1, 1201)),
        axis=0)

    ttest.plot_corr(avg_corr_loc0, avg_euc_loc0, avg_mah_loc0,
              "Region IT average over all subjects")
    ttest.plot_corr(avg_corr_loc1, avg_euc_loc1, avg_mah_loc1,
              "Region Calcarine average over all subjects")

    ITtrange1, ITtrange2, ITtrange3 = ttest.hypothesis_ttest(total_corr_loc0,
                                                       total_euc_loc0,
                                                       total_mah_loc0)
    Calcarinetrange1, Calcarinetrange2, Calcarinetrange3 = ttest.hypothesis_ttest(
        total_corr_loc1, total_euc_loc1,
        total_mah_loc1)

    plot_comparison(np.subtract(avg_mah_loc0, avg_corr_loc0),
                    np.subtract(avg_mah_loc0, avg_euc_loc0),
                    np.subtract(avg_corr_loc0, avg_euc_loc0),
                    ITtrange1, ITtrange2, ITtrange3,
                    "Region IT hypothesis evaluation")
    plot_comparison(np.subtract(avg_mah_loc1, avg_corr_loc1),
                    np.subtract(avg_mah_loc1, avg_euc_loc1),
                    np.subtract(avg_corr_loc1, avg_euc_loc1),
                    Calcarinetrange1, Calcarinetrange2, Calcarinetrange3,
                    "Region Calcarine hypothesis evaluation")


def fusion_twinsets():
    total_corr_loc0_twin1 = []
    total_corr_loc1_twin1 = []
    total_euc_loc0_twin1 = []
    total_euc_loc1_twin1 = []
    total_mah_loc0_twin1 = []
    total_mah_loc1_twin1 = []
    total_corr_loc0_twin2 = []
    total_corr_loc1_twin2 = []
    total_euc_loc0_twin2 = []
    total_euc_loc1_twin2 = []
    total_mah_loc0_twin2 = []
    total_mah_loc1_twin2 = []

    for i in range(1, 16):
        subject = "Subject" + str(i)
        print(subject)

        # Load fMRI and MEG and create twin sets
        RDM_corr_fMRI_loc0 = np.load(
            './fMRI_RDMs/rdm_cor_no_nn/corr_loc0_avg.npy')
        RDM_euclidean_fMRI_loc0 = np.load(
            './fMRI_RDMs/rdm_euclidean_no_nn/euc_loc0_avg.npy')
        RDM_mahalanobis_fMRI_loc0 = np.load(
            './fMRI_RDMs/rdm_mahalanobis_nn/mah_loc0_avg.npy')

        RDM_corr_fMRI_loc0_twin1, RDM_corr_fMRI_loc0_twin2, \
        RDM_euclidean_fMRI_loc0_twin1, \
        RDM_euclidean_fMRI_loc0_twin2, \
        RDM_mahalanobis_fMRI_loc0_twin1, RDM_mahalanobis_fMRI_loc0_twin2 = \
            create_twin_RDMs(
                RDM_corr_fMRI_loc0, RDM_euclidean_fMRI_loc0,
                RDM_mahalanobis_fMRI_loc0, fMRI=True)

        RDM_corr_fMRI_loc1 = np.load(
            './fMRI_RDMs/rdm_cor_no_nn/corr_loc1_avg.npy')
        RDM_euclidean_fMRI_loc1 = np.load(
            './fMRI_RDMs/rdm_euclidean_no_nn/euc_loc1_avg.npy')
        RDM_mahalanobis_fMRI_loc1 = np.load(
            './fMRI_RDMs/rdm_mahalanobis_nn/mah_loc1_avg.npy')

        RDM_corr_fMRI_loc1_twin1, RDM_corr_fMRI_loc1_twin2, \
        RDM_euclidean_fMRI_loc1_twin1, \
        RDM_euclidean_fMRI_loc1_twin2, \
        RDM_mahalanobis_fMRI_loc1_twin1, RDM_mahalanobis_fMRI_loc1_twin2 = \
            create_twin_RDMs(
                RDM_corr_fMRI_loc1, RDM_euclidean_fMRI_loc1,
                RDM_mahalanobis_fMRI_loc1, fMRI=True)

        try:
            RDM_corr_MEG = np.load(
                './Subjects/' + subject +
                '/Normalized_RDM_Correlation_Final.npy')
            RDM_euclidean_MEG = np.load(
                './Subjects/' + subject +
                '/Normalized_RDM_Euclidean_Final.npy')
            RDM_mahalanobis_MEG = np.load(
                './Subjects/' + subject +
                '/RDM_Mahalanobis_Final.npy')
        except:
            print("Error in loading MEG RDMs")

        RDM_corr_MEG_twin1, RDM_corr_MEG_twin2, RDM_euclidean_MEG_twin1, \
        RDM_euclidean_MEG_twin2, \
        RDM_mahalanobis_MEG_twin1, RDM_mahalanobis_MEG_twin2 = \
            create_twin_RDMs(
                RDM_corr_MEG, RDM_euclidean_MEG, RDM_mahalanobis_MEG)

        print("RDMs are loaded")

        corr_loc0_twin1 = []
        corr_loc1_twin1 = []
        euc_loc0_twin1 = []
        euc_loc1_twin1 = []
        mah_loc0_twin1 = []
        mah_loc1_twin1 = []
        corr_loc0_twin2 = []
        corr_loc1_twin2 = []
        euc_loc0_twin2 = []
        euc_loc1_twin2 = []
        mah_loc0_twin2 = []
        mah_loc1_twin2 = []

        for t in range(1201):
            corr_loc0_twin1.append(spearmanr(squareform(RDM_corr_MEG_twin1[t]),
                                             squareform(
                                                 RDM_corr_fMRI_loc0_twin1))[0])
            corr_loc0_twin2.append(spearmanr(squareform(RDM_corr_MEG_twin2[t]),
                                             squareform(
                                                 RDM_corr_fMRI_loc0_twin2))[0])

            corr_loc1_twin1.append(spearmanr(squareform(RDM_corr_MEG_twin1[t]),
                                             squareform(
                                                 RDM_corr_fMRI_loc1_twin1))[0])
            corr_loc1_twin2.append(spearmanr(squareform(RDM_corr_MEG_twin2[t]),
                                             squareform(
                                                 RDM_corr_fMRI_loc1_twin2))[0])

            euc_loc0_twin1.append(
                spearmanr(squareform(RDM_euclidean_MEG_twin1[t]),
                          squareform(RDM_euclidean_fMRI_loc0_twin1))[0])
            euc_loc0_twin2.append(
                spearmanr(squareform(RDM_euclidean_MEG_twin2[t]),
                          squareform(RDM_euclidean_fMRI_loc0_twin2))[0])

            euc_loc1_twin1.append(
                spearmanr(squareform(RDM_euclidean_MEG_twin1[t]),
                          squareform(RDM_euclidean_fMRI_loc1_twin1))[0])
            euc_loc1_twin2.append(
                spearmanr(squareform(RDM_euclidean_MEG_twin2[t]),
                          squareform(RDM_euclidean_fMRI_loc1_twin2))[0])

            mah_loc0_twin1.append(
                spearmanr(squareform(RDM_mahalanobis_MEG_twin1[t]),
                          squareform(RDM_mahalanobis_fMRI_loc0_twin1))[0])
            mah_loc0_twin2.append(
                spearmanr(squareform(RDM_mahalanobis_MEG_twin2[t]),
                          squareform(RDM_mahalanobis_fMRI_loc0_twin2))[0])

            mah_loc1_twin1.append(
                spearmanr(squareform(RDM_mahalanobis_MEG_twin1[t]),
                          squareform(RDM_mahalanobis_fMRI_loc1_twin1))[0])
            mah_loc1_twin2.append(
                spearmanr(squareform(RDM_mahalanobis_MEG_twin2[t]),
                          squareform(RDM_mahalanobis_fMRI_loc1_twin2))[0])

        total_corr_loc0_twin1.append(corr_loc0_twin1)
        total_corr_loc1_twin1.append(corr_loc1_twin1)
        total_euc_loc0_twin1.append(euc_loc0_twin1)
        total_euc_loc1_twin1.append(euc_loc1_twin1)
        total_mah_loc0_twin1.append(mah_loc0_twin1)
        total_mah_loc1_twin1.append(mah_loc1_twin1)
        total_corr_loc0_twin2.append(corr_loc0_twin2)
        total_corr_loc1_twin2.append(corr_loc1_twin2)
        total_euc_loc0_twin2.append(euc_loc0_twin2)
        total_euc_loc1_twin2.append(euc_loc1_twin2)
        total_mah_loc0_twin2.append(mah_loc0_twin2)
        total_mah_loc1_twin2.append(mah_loc1_twin2)

    directory = 'C:/Users/nbayat5/Desktop/Fusion/'
    sio.savemat(directory + 'total_corr_loc0_twin1.mat', {'vect': total_corr_loc0_twin1})
    sio.savemat(directory + 'total_corr_loc0_twin2.mat', {'vect': total_corr_loc0_twin2})
    sio.savemat(directory + 'total_euc_loc0_twin1.mat', {'vect': total_euc_loc0_twin1})
    sio.savemat(directory + 'total_euc_loc0_twin2.mat', {'vect': total_euc_loc0_twin2})
    sio.savemat(directory + 'total_mah_loc0_twin1.mat', {'vect': total_mah_loc0_twin1})
    sio.savemat(directory + 'total_mah_loc0_twin2.mat', {'vect': total_mah_loc0_twin2})

    # save plot average over subjects per fMRI location

    avg_corr_loc0_twin1 = np.mean(
        np.asarray(total_corr_loc0_twin1).reshape((-1, 1201)), axis=0)
    avg_corr_loc0_twin2 = np.mean(
        np.asarray(total_corr_loc0_twin2).reshape((-1, 1201)), axis=0)

    avg_corr_loc1_twin1 = np.mean(
        np.asarray(total_corr_loc1_twin1).reshape((-1, 1201)), axis=0)
    avg_corr_loc1_twin2 = np.mean(
        np.asarray(total_corr_loc1_twin2).reshape((-1, 1201)), axis=0)

    avg_euc_loc0_twin1 = np.mean(
        np.asarray(total_euc_loc0_twin1).reshape((-1, 1201)), axis=0)
    avg_euc_loc0_twin2 = np.mean(
        np.asarray(total_euc_loc0_twin2).reshape((-1, 1201)), axis=0)

    avg_euc_loc1_twin1 = np.mean(
        np.asarray(total_euc_loc1_twin1).reshape((-1, 1201)), axis=0)
    avg_euc_loc1_twin2 = np.mean(
        np.asarray(total_euc_loc1_twin2).reshape((-1, 1201)), axis=0)

    avg_mah_loc0_twin1 = np.mean(
        np.asarray(total_mah_loc0_twin1).reshape((-1, 1201)), axis=0)
    avg_mah_loc0_twin2 = np.mean(
        np.asarray(total_mah_loc0_twin2).reshape((-1, 1201)), axis=0)

    avg_mah_loc1_twin1 = np.mean(
        np.asarray(total_mah_loc1_twin1).reshape((-1, 1201)), axis=0)
    avg_mah_loc1_twin2 = np.mean(
        np.asarray(total_mah_loc1_twin2).reshape((-1, 1201)), axis=0)

    corr_loc0_time = twin_ttest(total_corr_loc0_twin1, total_corr_loc0_twin2)
    corr_loc1_time = twin_ttest(total_corr_loc1_twin1, total_corr_loc1_twin2)
    euc_loc0_time = twin_ttest(total_euc_loc0_twin1, total_euc_loc0_twin2)
    euc_loc1_time = twin_ttest(total_euc_loc1_twin1, total_euc_loc1_twin2)
    mah_loc0_time = twin_ttest(total_mah_loc0_twin1, total_mah_loc0_twin2)
    mah_loc1_time = twin_ttest(total_mah_loc1_twin1, total_mah_loc1_twin2)

    plot_twin_corr(avg_corr_loc0_twin1, avg_corr_loc0_twin2,corr_loc0_time, "avg_corr_loc0")
    plot_twin_corr(avg_corr_loc1_twin1, avg_corr_loc1_twin2, corr_loc1_time, "avg_corr_loc1")
    plot_twin_corr(avg_euc_loc0_twin1, avg_euc_loc0_twin2, euc_loc0_time, "avg_euc_loc0")
    plot_twin_corr(avg_euc_loc1_twin1, avg_euc_loc1_twin2, euc_loc1_time, "avg_euc_loc1")
    plot_twin_corr(avg_mah_loc0_twin1, avg_mah_loc0_twin2, mah_loc0_time, "avg_mah_loc0")
    plot_twin_corr(avg_mah_loc1_twin1, avg_mah_loc1_twin2, mah_loc1_time, "avg_mah_loc1")


if __name__ == '__main__':
    fusion_twinsets()
    # fusion()
