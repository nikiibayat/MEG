import hdf5storage
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt


def save_obj(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


# # data = hdf5storage.loadmat('Subject_5.mat')
# # save_obj(data,filename)

filename = 'saved_data.pkl'
data = load_obj(filename)

scenes = data['Data'][:, 64:100][0]
faces = data['Data'][:, 124:156][0]

Accuracy = []
for t in range(1201):
    X = []
    y = []
    for i in range(len(scenes)):
        for trial in range(scenes[i].shape[1]):
            raw_data = scenes[i][:, trial][0][:, t]
            print("raw data for each scene trial at time t looks like: ",raw_data[0])
            X.append(raw_data / np.std(raw_data))
            y.append(0)
    for j in range(len(faces)):
        for trial in range(faces[j].shape[1]):
            raw_data = faces[j][:, trial][0][:, t]
            X.append(raw_data / np.std(raw_data))
            y.append(1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = svm.SVC(gamma='scale')
    model = clf.fit(X_train, y_train)
    print(t, ": ", model.score(X_test, y_test))
    Accuracy.append(model.score(X_test, y_test))

print("---------------------------------------------------------------")
print("Accuracy at time point {} is highest!".format(np.argmax(Accuracy)))
x = np.arange(1201)
plt.plot(x,Accuracy)
plt.title('Accuracy of SVM in different time points')
plt.ylabel('Accuracy')
plt.xlabel('Time')
plt.show()
plt.savefig("output.png")
# T = 340 is highest
