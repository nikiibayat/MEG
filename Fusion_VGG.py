import numpy as np
import hdf5storage as h5
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt


def load_brain_RDMs(subj):
    MEG = np.load('./Subjects/Subject' + str(subj) + '/RDM_Correlation_Final.npy')
    fMRI0 = np.load('./fMRI_RDMs/rdm_cor_no_nn/' + str(subj) + '_0.npy')
    fMRI1 = np.load('./fMRI_RDMs/rdm_cor_no_nn/' + str(subj) + '_1.npy')
    return MEG, fMRI0, fMRI1

def transform_RDM(vgg_RDM):
    RDM = np.zeros((156, 156), dtype=np.float64)
    RDM[0: 28, 0: 28] = vgg_RDM[0: 28, 0: 28] #Objects
    RDM[28: 64, 28: 64] = vgg_RDM[60: 96, 60: 96] #Objects
    RDM[64: 100, 64: 100] = vgg_RDM[96: 132, 96: 132]  #Scenes
    RDM[100: 124, 100: 124] = vgg_RDM[132: 156, 132: 156]  #People
    RDM[124: 156, 124: 156] = vgg_RDM[28: 60, 28: 60]  #Faces
    return RDM

def load_vgg_RDMs():
    vgg_RDMs = {}
    vgg_RDMs['D_L1'] = transform_RDM(h5.loadmat('./Vgg_RDMs/D_L1.mat'))
    vgg_RDMs['D_L2'] = transform_RDM(h5.loadmat('./Vgg_RDMs/D_L2.mat'))
    vgg_RDMs['D_L3'] = transform_RDM(h5.loadmat('./Vgg_RDMs/D_L3.mat'))
    vgg_RDMs['D_L4'] = transform_RDM(h5.loadmat('./Vgg_RDMs/D_L4.mat'))
    vgg_RDMs['D_L5'] = transform_RDM(h5.loadmat('./Vgg_RDMs/D_L5.mat'))
    vgg_RDMs['D_L6'] = transform_RDM(h5.loadmat('./Vgg_RDMs/D_L6.mat'))

    vgg_RDMs['E_L1'] = transform_RDM(h5.loadmat('./Vgg_RDMs/E_L1.mat'))
    vgg_RDMs['E_L2'] = transform_RDM(h5.loadmat('./Vgg_RDMs/E_L2.mat'))
    vgg_RDMs['E_L3'] = transform_RDM(h5.loadmat('./Vgg_RDMs/E_L3.mat'))
    vgg_RDMs['E_L4'] = transform_RDM(h5.loadmat('./Vgg_RDMs/E_L4.mat'))
    vgg_RDMs['E_L5'] = transform_RDM(h5.loadmat('./Vgg_RDMs/E_L5.mat'))
    vgg_RDMs['E_L6'] = transform_RDM(h5.loadmat('./Vgg_RDMs/E_L6.mat'))

    return vgg_RDMs


def plot_MEG_corr(vgg_fusion):
    time = np.arange(-200, 1001)
    plt.title("MEG VGG correlation Encoder")
    plt.plot(time, vgg_fusion['E_L1_MEG'], color='#2166ac', label="E_L1")
    plt.plot(time, vgg_fusion['E_L2_MEG'], color='#1a9850', label="E_L2")
    plt.plot(time, vgg_fusion['E_L3_MEG'], color='#d73027', label="E_L3")
    plt.plot(time, vgg_fusion['E_L4_MEG'], color='#ef8a62', label="E_L4")
    plt.plot(time, vgg_fusion['E_L5_MEG'], color='#f0027f', label="E_L5")
    plt.plot(time, vgg_fusion['E_L6_MEG'], color='#ffff99', label="E_L6")
    plt.legend()
    plt.savefig('./fusion_plots/MEG_VGG_correlation_Encoder')
    plt.close()
    plt.title("MEG VGG correlation Decoder")
    plt.plot(time, vgg_fusion['D_L1_MEG'], color='#2166ac', label="D_L1")
    plt.plot(time, vgg_fusion['D_L2_MEG'], color='#1a9850', label="D_L2")
    plt.plot(time, vgg_fusion['D_L3_MEG'], color='#d73027', label="D_L3")
    plt.plot(time, vgg_fusion['D_L4_MEG'], color='#ef8a62', label="D_L4")
    plt.plot(time, vgg_fusion['D_L5_MEG'], color='#f0027f', label="D_L5")
    plt.plot(time, vgg_fusion['D_L6_MEG'], color='#ffff99', label="D_L6")
    plt.legend()
    plt.savefig('./fusion_plots/MEG_VGG_correlation_Decoder')
    plt.close()

if __name__ == '__main__':
    vgg_fusion = {}
    vgg_RDMs = load_vgg_RDMs()
    for i in range(1, 16):
        print("subject ",str(i))
        MEG, fMRI0, fMRI1 = load_brain_RDMs(i)
        for key in vgg_RDMs.keys():
            corr = []
            if (key + '_fMRI0') not in vgg_fusion.keys():
                vgg_fusion[key + '_fMRI0'] = [
                    spearmanr(squareform(fMRI0), squareform(vgg_RDMs[key]['RDM']))]
                vgg_fusion[key + '_fMRI1'] = [
                    spearmanr(squareform(fMRI1), squareform(vgg_RDMs[key]['RDM']))]
                vgg_fusion[key + '_MEG'] = []
            else:
                vgg_fusion[key + '_fMRI0'].append(
                    spearmanr(squareform(fMRI0), squareform(vgg_RDMs[key]['RDM'])))
                vgg_fusion[key + '_fMRI1'].append(
                    spearmanr(squareform(fMRI1), squareform(vgg_RDMs[key]['RDM'])))
            for t in range(1201):
                corr.append(spearmanr(squareform(MEG[t]),squareform(vgg_RDMs[key]['RDM'])))
            vgg_fusion[key + '_MEG'].append(corr)

    for key in vgg_fusion.keys():
        vgg_fusion[key] = np.mean(vgg_fusion[key])
        if 'MEG' not in key:
            print(key,"- Avg Correlation: ",vgg_fusion[key])

    # avg_meg = np.load('./Subjects/MEG_Corr_avg_oversubjects.npy')
    # avg_fMRI0 = np.load('./fMRI_RDMs/rdm_cor_no_nn/corr_loc0_avg.npy')
    # avg_fMRI0 = np.load('./fMRI_RDMs/rdm_cor_no_nn/corr_loc1_avg.npy')

    plot_MEG_corr(vgg_fusion)
