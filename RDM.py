import hdf5storage
import numpy as np
from numba import prange, njit


def cov1para(x, shrink):
    t, n = x.shape
    meanx = np.mean(x, axis=0).reshape((1, 306))
    x = np.subtract(x, meanx)
    sample = np.multiply((1 / t), np.matmul(x.T, x))
    meanvar = np.divide(np.trace(sample), n)
    prior = np.multiply(meanvar, np.identity(n))
    if shrink == -1:
        y = np.power(x, 2)
        phiMat = np.subtract(np.divide(np.matmul(y.T, y), t),
                             np.power(sample, 2))
        phi = np.sum(np.sum(phiMat))
        gamma = sample - prior
        gamma = np.sum(np.power(gamma, 2))
        kappa = phi / gamma
        shrinkage = max(0, min(1, kappa / t))
        # print("shrinkage: ", shrinkage)
    else:
        shrinkage = shrink
    sigma = np.multiply(shrinkage, prior) + np.multiply((1 - shrinkage), sample)
    # print("sigma (covariance) shape: ", sigma.shape)
    return sigma, shrinkage


def compute_cov(images):
    ntimes = 0
    cov = np.zeros((306, 306))
    for c in range(images.shape[0]):
        dat = images[c][0]
        for t in range(1, 1201, 12):
            ntimes += 1
            x = np.zeros((len(dat), 306))
            for trial in range(len(dat)):
                x[trial] = dat[trial][:, t]
            sigma, shrinkage = cov1para(np.squeeze(x),
                                        -1)  # x should be of size ntrials * 306
            cov += sigma
    cov = cov / (images.shape[0] * ntimes)
    W = np.linalg.inv(cov)
    return W


def correlation(img1, img2):
    return 1 - np.corrcoef(img1, img2)[0, 1]


def euclidean(img1, img2):
    # only using the magnetometer sensors
    img1 = img1[1: len(img1): 3]
    img2 = img2[1: len(img2): 3]
    distance = np.linalg.norm(img1 - img2)
    return distance


def mahalanobis_dist(img1, img2, data_cov):
    c = np.stack((img1, img2), axis=1)
    delta = img1 - img2
    return np.sqrt(np.dot(np.dot(delta, data_cov), delta))


def normalize_magnet(images, subject):
    for img in range(len(images)):
        condition = images[img].reshape(-1, 1)
        # print("condition: %d # of trials: %d" % (img, condition.shape[0]))
        for tri in range(condition.shape[0]):
            if subject == 1 and img == 23 and tri == 1:
                continue
            elif subject == 1 and img == 73 and tri == 14:
                continue
            elif subject == 1 and img == 93 and tri == 2:
                continue
            elif subject == 1 and img == 123 and tri == 19:
                continue
            elif subject == 1 and img == 147 and tri == 10:
                continue
            elif subject == 1 and img == 149 and tri == 18:
                continue
            elif subject == 6 and img == 125 and tri == 12:
                continue
            elif subject == 8 and img == 85 and tri == 11:
                continue
            elif subject == 9 and img == 89 and tri == 9:
                continue
            elif subject == 10 and img == 149 and tri == 24:
                continue

            trial = condition[tri][0]
            for sens in range(1, len(trial), 3):
                sensor = trial[sens]
                baseline = []
                for t in range(200):
                    baseline.append(sensor[t])
                baseline_std = np.std(np.asarray(baseline))
                images[img].reshape(-1, 1)[tri][0][sens] = np.divide(sensor,
                                                                     baseline_std)
    return images


if __name__ == '__main__':
    for subj in range(8, 9):
        print("Subject " + str(subj))
        data = hdf5storage.loadmat('./Raw_MEG/Subject_' + str(subj) + '.mat')
        images = data['Data'][0]
        # data_cov = compute_cov(images)
        images = normalize_magnet(images, subj)

        for trial in range(len(images)):
            images[trial] = np.mean(images[trial])


        def finalRDM(images):
            RDM_corr = np.zeros([1201, 156, 156], dtype=np.float64)
            RDM_euclidean = np.zeros([1201, 156, 156], dtype=np.float64)
            RDM_mahalanobis = np.zeros([1201, 156, 156], dtype=np.float64)
            for t in prange(1201):
                # if t % 100 == 0:
                #     print("Time point: ", t)
                for i in range(images.shape[0]):
                    for j in range(i + 1, images.shape[0]):
                        # RDM_corr[t, i, j] = correlation(images[i][:, t],
                        # images[j][:, t])
                        # RDM_corr[t, j, i] = RDM_corr[t, i, j]
                        RDM_euclidean[t, i, j] = euclidean(images[i][:, t],
                                                           images[j][:, t])
                        RDM_euclidean[t, j, i] = RDM_euclidean[t, i, j]
                        # RDM_mahalanobis[t, i, j] = mahalanobis_dist(images[
                        # i][:, t], images[j][:, t],data_cov)
                        # RDM_mahalanobis[t, j, i] = RDM_mahalanobis[t, i, j]

            return RDM_euclidean, RDM_corr, RDM_mahalanobis

        RDM_euclidean, RDM_corr, RDM_mahalanobis = finalRDM(images)
        np.save('./Subjects/Subject' + str(
            subj) + '/Magnet_Normalized_RDM_Euclidean_Final', RDM_euclidean)
        # np.save('./Subjects/Subject15/RDM_Correlation_Final', RDM_corr)
        # np.save('./Subjects/Subject15/RDM_Mahalanobis_Final', RDM_mahalanobis)
