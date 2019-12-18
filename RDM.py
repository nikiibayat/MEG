import hdf5storage
import numpy as np
from sklearn.svm import SVC

import multiprocessing as mp
from multiprocessing import Process
from os import walk


def cov1para(x, shrink):
    magnet = True
    sensor = 102 if magnet else 306
    t, n = x.shape
    meanx = np.mean(x, axis=0).reshape((1, sensor))
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
    else:
        shrinkage = shrink
    sigma = np.multiply(shrinkage, prior) + np.multiply((1 - shrinkage), sample)
    # print("sigma (covariance) shape: ", sigma.shape)
    return sigma, shrinkage


def compute_cov(images):
    magnet = True
    sensor = 102 if magnet else 306
    ntimes = 0
    cov = np.zeros((sensor, sensor))
    for c in range(images.shape[0]):
        dat = images[c][0]
        for t in range(1, 1201, 12):
            ntimes += 1
            x = np.zeros((len(dat), sensor))
            for trial in range(len(dat)):
                if magnet:
                    temp = dat[trial][:, t]
                    x[trial] = temp[1: 306: 3]
                else:
                    x[trial] = dat[trial][:, t]
            sigma, shrinkage = cov1para(np.squeeze(x), -1)
            cov += sigma
    cov = cov / (images.shape[0] * ntimes)
    W = np.linalg.inv(cov)
    return W


def correlation(img1, img2):
    img1 = img1[1: len(img1): 3]
    img2 = img2[1: len(img2): 3]
    return 1 - np.corrcoef(img1, img2)[0, 1]


def euclidean(img1, img2):
    # only using the magnetometer sensors
    img1 = img1[1: len(img1): 3]
    img2 = img2[1: len(img2): 3]
    distance = np.linalg.norm(img1 - img2)
    return distance


def mahalanobis_dist(img1, img2, data_cov):
    img1 = img1[1: len(img1): 3]
    img2 = img2[1: len(img2): 3]
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


def create_MEG_RDMs():
    for subj in range(1, 16):
        print("Subject " + str(subj))
        data = hdf5storage.loadmat('./Raw_MEG/Subject_' + str(subj) + '.mat')
        images = data['Data'][0]
        data_cov = compute_cov(images)
        images = normalize_magnet(images, subj)

        for trial in range(len(images)):
            images[trial] = np.mean(images[trial])

        def finalRDM(images):
            RDM_corr = np.zeros([1201, 156, 156], dtype=np.float64)
            RDM_euclidean = np.zeros([1201, 156, 156], dtype=np.float64)
            RDM_mahalanobis = np.zeros([1201, 156, 156], dtype=np.float64)
            for t in range(1201):
                if t % 100 == 0:
                    print("Time point: ", t)
                for i in range(images.shape[0]):
                    for j in range(i + 1, images.shape[0]):
                        RDM_corr[t, i, j] = correlation(images[i][:, t],
                                                        images[j][:, t])
                        RDM_corr[t, j, i] = RDM_corr[t, i, j]
                        # RDM_euclidean[t, i, j] = euclidean(images[i][:, t],
                        #                                    images[j][:, t])
                        # RDM_euclidean[t, j, i] = RDM_euclidean[t, i, j]
                        RDM_mahalanobis[t, i, j] = mahalanobis_dist(
                            images[i][:, t], images[j][:, t], data_cov)
                        RDM_mahalanobis[t, j, i] = RDM_mahalanobis[t, i, j]

            return RDM_euclidean, RDM_corr, RDM_mahalanobis

        RDM_euclidean, RDM_corr, RDM_mahalanobis = finalRDM(images)
        # np.save('./Subjects/Subject' + str(
        #     subj) + '/Magnet_Normalized_RDM_Euclidean_Final', RDM_euclidean)
        np.save('./Subjects/Subject15/Magnet_Normalized_RDM_Correlation_Final', RDM_corr)
        np.save('./Subjects/Subject15/Magnet_Normalized_RDM_Mahalanobis_Final', RDM_mahalanobis)


def create_SVM_RDMs():
    ####### works with MEG data on sharcnet##############
    print("Number of cpu : ", mp.cpu_count())
    for subj in range(1, 16):
        print("Subject " + str(subj))
        sub = "0" + str(subj) if subj < 10 else str(subj)
        subject_folder = '/home/nbayat5/projects/def-yalda/Memorability_Data' \
                         '/MEG/PercepFluFus_' + sub

        num_cores = 32
        # RDM_svm = np.zeros([1201, 156, 156], dtype=np.float64)
        jobs = []
        manager = mp.Manager()
        RDM_svm = manager.list()
        for t in range(1201):
            if num_cores > 1:
                p = Process(target=svm_RDM_per_time,
                            args=(subject_folder, t))
                p.start()
                jobs.append(p)
                if (len(jobs) == num_cores):
                    for job in jobs:
                        job.join()
                    jobs = []
                    print("16 time points for subject " + str(
                        subj) + " are done!")
            else:
                RDM_svm.append(svm_RDM_per_time(subject_folder, t))

        if len(jobs) != 0:
            for job in jobs:
                job.join()

        np.array(RDM_svm).reshape((1201, 156, 156))
        np.save('./RDM_SVM_' + str(subj), RDM_svm)


def svm_RDM_per_time(subject_folder, t):
    result = np.zeros([156, 156], dtype=np.float64)
    x_train_total = []
    x_test_total = []
    y_train_total = []
    y_test_total = []
    for i in range(156):
        x_train, y_train, x_test, y_test = get_svm_data(t, i, subject_folder)
        x_train_total.append(x_train)
        y_train_total.append(y_train)
        x_test_total.append(x_test)
        y_test_total.append(y_test)

    x_train_total = np.array(x_train_total).reshape(-1, 306)
    y_train_total = np.array(y_train_total).ravel()
    x_test_total = np.array(x_test_total).reshape(-1, 306)
    y_test_total = np.array(y_test_total).ravel()
    # one against one - for 156 condition
    clf = SVC(decision_function_shape='ovo', gamma='auto')
    model = clf.fit(x_train_total, y_train_total)
    for img1 in range(156):
        for img2 in range(img1 + 1, 156):
            result[img1, img2] = model.score(x_test_total, y_test_total)
    return result


def get_svm_data(time, i, subject_folder):
    img1_dir = subject_folder + "/" + str(i + 1) + "/"
    y_train = []
    y_test = []
    trials1 = []
    ############################Load trials#############################
    for (dirpath, dirnames, filenames) in walk(img1_dir):
        for filename in filenames:
            if "low" in filename:
                trials1.append(hdf5storage.loadmat(img1_dir + filename))

    ###################Randomly permute trials####################
    image1_trials = np.random.permutation(
        np.array(trials1).reshape(-1, 1))
    bins1 = np.array_split(image1_trials, 6)

    ###################Average each bin####################
    for idx in range(6):
        bins1[idx] = np.sum(
            [bins1[idx][k][0]['F'] for k in range(bins1[idx].shape[0])],
            axis=0) / \
                     bins1[idx].shape[0]
    #########Train svm using 5 bins leave one for test#########
    x_train = []
    for i in range(5):
        baseline_std = np.std(np.asarray(bins1[i][:306, :200]))
        x_train.append(bins1[i][:306, time] / baseline_std)
        y_train.append(i + 1)

    x_test = bins1[5][:306, time]
    y_test.append(i + 1)

    return x_train, y_train, x_test, y_test


# def avg_svm_dist(time, img1_dir, img2_dir, num_cores):
#     jobs = []
#     manager = mp.Manager()
#     final_svm = manager.list()
#     for iter in range(50):
#         if num_cores > 1:
#             p = Process(target=svm_dist,
#                         args=(time, img1_dir, img2_dir, final_svm))
#             p.start()
#             jobs.append(p)
#             if (len(jobs) == num_cores):
#                 for job in jobs:
#                     job.join()
#                 jobs = []
#                 print("16 permutations of svm for time "+str(time)+" are
#                 done!")
#         else:
#             svm_dist(time, img1_dir, img2_dir, final_svm)
#
#     if len(jobs) != 0:
#         for job in jobs:
#             job.join()
#     return np.mean(final_svm)


if __name__ == '__main__':
    create_MEG_RDMs()
    # create_SVM_RDMs()
