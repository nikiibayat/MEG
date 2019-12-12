from HypothesisTtest import *
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    total_corr_loc0 = []
    total_corr_loc1 = []
    total_euc_loc0 = []
    total_euc_loc1 = []
    total_mah_loc0 = []
    total_mah_loc1 = []

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
        # Load MEG and fMRI RDMs
        try:
            RDM_corr_MEG = np.load(
                './Subjects/' + subject + '/RDM_Correlation_Final.npy')
            RDM_euclidean_MEG = np.load(
                './Subjects/' + subject +
                '/Magnet_Normalized_RDM_Euclidean_Final.npy')
            RDM_mahalanobis_MEG = np.load(
                './Subjects/' + subject + '/RDM_Mahalanobis_Final.npy')

            # RDM_corr_fMRI_loc0 = np.load(
            #     './fMRI_RDMs/rdm_cor_no_nn/' + str(i) + '_0.npy')
            # RDM_euclidean_fMRI_loc0 = np.load(
            #     './fMRI_RDMs/rdm_euclidean_no_nn/' + str(i) + '_0.npy')
            # RDM_mahalanobis_fMRI_loc0 = np.load(
            #     './fMRI_RDMs/rdm_mahalanobis_nn/' + str(i) + '_0.npy')
            #
            # RDM_corr_fMRI_loc1 = np.load(
            #     './fMRI_RDMs/rdm_cor_no_nn/' + str(i) + '_1.npy')
            # RDM_euclidean_fMRI_loc1 = np.load(
            #     './fMRI_RDMs/rdm_euclidean_no_nn/' + str(i) + '_1.npy')
            # RDM_mahalanobis_fMRI_loc1 = np.load(
            #     './fMRI_RDMs/rdm_mahalanobis_nn/' + str(i) + '_1.npy')

            print("RDMs are loaded")
        except:
            print("Error in Loading RDMs")

        # Create twin sets for each RDM
        # try:
        #     RDM_corr_MEG_twin1, RDM_corr_MEG_twin2, RDM_euclidean_MEG_twin1, \
        #     RDM_euclidean_MEG_twin2, \
        #     RDM_mahalanobis_MEG_twin1, RDM_mahalanobis_MEG_twin2 = \
        #         create_twin_RDMs(
        #             RDM_corr_MEG,
        #             RDM_euclidean_MEG,
        #             RDM_mahalanobis_MEG)
        #
        #     RDM_corr_fMRI_loc0_twin1, RDM_corr_fMRI_loc0_twin2, \
        #     RDM_euclidean_fMRI_loc0_twin1, \
        #     RDM_euclidean_fMRI_loc0_twin2, \
        #     RDM_mahalanobis_fMRI_loc0_twin1,
        #     RDM_mahalanobis_fMRI_loc0_twin2 = \
        #         create_twin_RDMs(
        #             RDM_corr_fMRI_loc0,
        #             RDM_euclidean_fMRI_loc0,
        #             RDM_mahalanobis_fMRI_loc0)
        #
        #     RDM_corr_fMRI_loc1_twin1, RDM_corr_fMRI_loc1_twin2, \
        #     RDM_euclidean_fMRI_loc1_twin1, \
        #     RDM_euclidean_fMRI_loc1_twin2, \
        #     RDM_mahalanobis_fMRI_loc1_twin1,
        #     RDM_mahalanobis_fMRI_loc1_twin2 = \
        #         create_twin_RDMs(
        #             RDM_corr_fMRI_loc1,
        #             RDM_euclidean_fMRI_loc1,
        #             RDM_mahalanobis_fMRI_loc1)
        #     print("twin sets are created!")
        # except:
        #     print("Error in creating twin set RDMs")

        corr_loc0 = []
        corr_loc1 = []
        euc_loc0 = []
        euc_loc1 = []
        mah_loc0 = []
        mah_loc1 = []

        for t in range(1201):
            # print(spearmanr(RDM_corr_MEG[t], RDM_corr_fMRI_loc0)[1].shape)
            corr_loc0.append(spearmanr(squareform(RDM_corr_MEG[t]),
                                       squareform(
                                           RDM_corr_fMRI_loc0))[0])
            corr_loc1.append(spearmanr(squareform(RDM_corr_MEG[t]),
                                       squareform(
                                           RDM_corr_fMRI_loc1))[0])

            euc_loc0.append(
                spearmanr(squareform(RDM_euclidean_MEG[t]),
                          squareform(RDM_euclidean_fMRI_loc0))[0])
            euc_loc1.append(
                spearmanr(squareform(RDM_euclidean_MEG[t]),
                          squareform(RDM_euclidean_fMRI_loc1))[0])

            mah_loc0.append(
                spearmanr(squareform(RDM_mahalanobis_MEG[t]),
                          squareform(RDM_mahalanobis_fMRI_loc0))[0])
            mah_loc1.append(
                spearmanr(squareform(RDM_mahalanobis_MEG[t]),
                          squareform(RDM_mahalanobis_fMRI_loc1))[0])

        total_corr_loc0.append(corr_loc0)
        total_corr_loc1.append(corr_loc1)
        total_euc_loc0.append(euc_loc0)
        total_euc_loc1.append(euc_loc1)
        total_mah_loc0.append(mah_loc0)
        total_mah_loc1.append(mah_loc1)

    # save plot average over subjects per fMRI location

    avg_corr_loc0 = np.mean(
        np.asarray(total_corr_loc0).reshape((-1, 1201)),
        axis=0)
    avg_corr_loc1 = np.mean(
        np.asarray(total_corr_loc1).reshape((-1, 1201)),
        axis=0)
    avg_euc_loc0 = np.mean(
        np.asarray(total_euc_loc0).reshape((-1, 1201)),
        axis=0)
    avg_euc_loc1 = np.mean(
        np.asarray(total_euc_loc1).reshape((-1, 1201)),
        axis=0)
    avg_mah_loc0 = np.mean(
        np.asarray(total_mah_loc0).reshape((-1, 1201)),
        axis=0)
    avg_mah_loc1 = np.mean(
        np.asarray(total_mah_loc1).reshape((-1, 1201)),
        axis=0)

    plot_corr(avg_corr_loc0, avg_euc_loc0, avg_mah_loc0,
              "Region IT average over all subjects")
    plot_corr(avg_corr_loc1, avg_euc_loc1, avg_mah_loc1,
              "Region Calcarine average over all subjects")

    ITtrange1, ITtrange2, ITtrange3 = hypothesis_ttest(total_corr_loc0,
                                                       total_euc_loc0,
                                                       total_mah_loc0)
    Calcarinetrange1, Calcarinetrange2, Calcarinetrange3 = hypothesis_ttest(
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
