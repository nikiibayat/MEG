import hdf5storage
import pickle
import numpy as np
from numba import prange, njit
from matplotlib.pyplot import imshow, show, colorbar, title, savefig
from scipy import stats
from multiprocessing import Pool



def save_obj(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


# data = hdf5storage.loadmat('Subject_5.mat')
# save_obj(data,filename)

# def covCor(x):
#     t, n = x.shape
#     mean_x = np.mean(x, axis=0)
#     x = np.subtract(x, mean_x)
#     # sample covariance matrix
#     sample = np.multiply((1/t), (np.dot(x.T, x)))
#     # compute prior
#     var = np.diag(sample).reshape(n, 1)
#     sqrtvar = np.sqrt(var)
#     rbar = float(np.divide(np.sum(np.subtract(np.sum(np.divide(sample, np.dot(sqrtvar, sqrtvar.T)), axis=0).reshape(n, 1), n), axis=0), (n*(n-1))))
#     prior = rbar*np.dot(sqrtvar, sqrtvar.T)
#     np.fill_diagonal(prior, var)
#     y = np.square(x)
#     phi_mat = np.add(np.subtract(np.divide(np.dot(y.T, y), t), np.divide(np.multiply(2*(np.dot(x.T, x)), sample), t)), np.square(sample))
#     phi = float(np.sum(np.sum(phi_mat, axis=0).reshape(-1, 1), axis=0))
#     term1 = np.divide(np.dot(np.power(x, 3).T, x), t)
#     help = np.divide(np.dot(x.T, x), t)
#     helpDiag = np.diag(help).reshape(-1, 1)
#     term2 = np.multiply(helpDiag, sample)
#     term3 = np.multiply(help, var)
#     term4 = np.multiply(var, sample)
#     theta_mat = term1 - term2 - term3 + term4
#     np.fill_diagonal(theta_mat, np.zeros((n, 1)))
#     rho = np.sum(np.diag(phi_mat)) + rbar*np.sum(np.sum(np.multiply(np.dot(np.divide(1, sqrtvar), sqrtvar.T), theta_mat), axis=0), axis=0)
#     gamma = np.power(np.linalg.norm(np.subtract(sample, prior)), 2)
#     kappa = (phi-rho)/gamma
#     shrinkage = max(0, min(1, np.divide(kappa, t)))
#     sigma = shrinkage*prior + (1-shrinkage)*sample
#     return sigma


def cov1para(x, shrink):
    t, n = x.shape
    meanx = np.mean(x, axis=0).reshape((1, 306))
    x = np.subtract(x, meanx)
    sample = np.multiply((1/t), np.matmul(x.T, x))
    meanvar = np.divide(np.trace(sample), n)
    prior=np.multiply(meanvar, np.identity(n))
    if shrink == -1:
      y = np.power(x, 2)
      phiMat = np.subtract(np.divide(np.matmul(y.T, y),t),np.power(sample, 2))
      phi = np.sum(np.sum(phiMat))
      gamma = sample-prior
      gamma = np.sum(np.power(gamma, 2))
      kappa = phi/gamma
      shrinkage = max(0, min(1, kappa/t))
      # print("shrinkage: ", shrinkage)
    else:
        shrinkage = shrink
    sigma = np.multiply(shrinkage, prior) + np.multiply((1-shrinkage), sample)
    # print("sigma (covariance) shape: ", sigma.shape)
    return sigma, shrinkage

def compute_cov(images):
    ntimes = 0
    cov = np.zeros((306, 306))
    for c in range(images.shape[0]):
        dat = images[c][0]
        for t in range(1, 1201, 12):
            ntimes += 1
            x = np.zeros((len(dat),306))
            for trial in range(len(dat)):
                x[trial] = dat[trial][:, t]
            sigma, shrinkage = cov1para(np.squeeze(x),-1) # x should be of size ntrials * 306
            cov += sigma
    cov = cov/(images.shape[0]*ntimes)
    W = np.linalg.inv(cov)
    return W


def correlation(img1,img2):
    return 1-np.corrcoef(img1, img2)[0, 1]


def euclidean(img1,img2):
    distance = np.linalg.norm(img1-img2)
    return distance


def mahalanobis_dist(img1,img2, data_cov):
    c = np.stack((img1, img2), axis=1)
    delta = img1 - img2
    return np.sqrt(np.dot(np.dot(delta, data_cov), delta))


if __name__ == '__main__':

    data = load_obj('saved_data.pkl')
    images = data['Data'][0]
    data_cov = compute_cov(images)

    for trial in range(len(images)):
        images[trial] = np.mean(images[trial])

    # @njit(parallel=True)
    def finalRDM(images):
        RDM_corr = np.zeros([1201, 156, 156], dtype=np.float64)
        RDM_euclidean = np.zeros([1201, 156, 156], dtype=np.float64)
        RDM_mahalanobis = np.zeros([1201, 156, 156], dtype=np.float64)
        for t in prange(1201):
            print("Time point: ",t)
            for i in range(images.shape[0]):
                for j in range(i + 1, images.shape[0]):
                    RDM_corr[t, i, j] = correlation(images[i][:, t], images[j][:, t])
                    RDM_corr[t, j, i] = RDM_corr[t, i, j]
                    RDM_euclidean[t, i, j] = euclidean(images[i][:, t], images[j][:, t])
                    RDM_euclidean[t, j, i] = RDM_euclidean[t, i, j]
                    RDM_mahalanobis[t, i, j] = mahalanobis_dist(images[i][:, t], images[j][:, t],data_cov)
                    RDM_mahalanobis[t, j, i] = RDM_mahalanobis[t, i, j]
        return RDM_euclidean, RDM_corr, RDM_mahalanobis


    RDM_euclidean, RDM_corr, RDM_mahalanobis = finalRDM(images)
    np.save('RDM_Euclidean_Final', RDM_euclidean)
    np.save('RDM_Correlation_Final', RDM_corr)
    np.save('RDM_Mahalanobis_Final', RDM_mahalanobis)


print("---------------------------------------------------------------")

RDM_corr = np.load('RDM_Correlation_Final.npy')
RDM_euclidean = np.load('RDM_Euclidean_Final.npy')
RDM_mahalanobis = np.load('RDM_Mahalanobis_Final.npy')
imshow(RDM_corr[1200,:,:])
title("RDM based on 1 - Correlation")
colorbar()
show()
imshow(RDM_euclidean[1200,:,:])
title("RDM based on Euclidean distance")
colorbar()
show()

imshow(RDM_mahalanobis[1200, :, :])
title("RDM based on Mahalanobis distance")
colorbar()
show()
savefig("Mahalanobis_RDM")
#
