import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import preprocessing
import pickle

data = pd.read_csv("Motor.csv")

le =                    preprocessing.LabelEncoder()
INSURED_AGE =           le.fit_transform(list(data["INSURED_AGE"]))
np.save('insured_age.npy', le.classes_)
INSURED_NATIONALITY =   le.fit_transform(list(data["INSURED_NATIONALITY"]))
np.save('insured_nationality.npy', le.classes_)
SOURCE =                le.fit_transform(list(data["SOURCE"]))
np.save('source.npy', le.classes_)
POL_TYPE =              le.fit_transform(list(data["POL_TYPE"]))
np.save('pol_type.npy', le.classes_)
BUSINESS =              le.fit_transform(list(data["BUSINESS"]))
np.save('business.npy', le.classes_)
TYPE_OF_BODY =          le.fit_transform(list(data["TYPE_OF_BODY"]))
np.save('type_of_body.npy', le.classes_)
VEHICLE_MAKE =          le.fit_transform(list(data["VEHICLE_MAKE"]))
np.save('vehicle_make.npy', le.classes_)
VEHICLE_MODEL =         le.fit_transform(list(data["VEHICLE_MODEL"]))
np.save('vehicle_model.npy', le.classes_)
VEHICLE_AGE =              le.fit_transform(list(data["VEHICLE_AGE"]))
np.save('vehicle_age.npy', le.classes_)
CLASS_OF_USE =          le.fit_transform(list(data["CLASS_OF_USE"]))
np.save('class_of_use.npy', le.classes_)
LOSS_RATIO =            le.fit_transform(list(data["NET_LOSS_RATIO"]))
np.save('loss_ratio.npy', le.classes_)

predict = "NET_LOSS_RATIO"

X = list(zip(INSURED_AGE, INSURED_NATIONALITY, SOURCE, POL_TYPE, BUSINESS, TYPE_OF_BODY, VEHICLE_MAKE, VEHICLE_MODEL, VEHICLE_AGE, CLASS_OF_USE))
y = list(LOSS_RATIO)

#s = preprocessing.StandardScaler()
#X = s.fit_transform(X)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

knn_model = KNeighborsClassifier(n_neighbors=15)
knn_model.fit(x_train, y_train)
acc = knn_model.score(x_test, y_test)
print(acc)

with open("motor_knn_model.pickle", "wb") as f:
    pickle.dump(knn_model, f)

pickle_in = open("motor_knn_model.pickle", "rb")
knn = pickle.load(pickle_in)

predicted = knn.predict(x_test)
predicted = le.inverse_transform(predicted)
y_test = le.inverse_transform((y_test))

xx = []
ya = []
yy = []
pfile = open("predictions.csv", "w+")
for x in range(len(predicted)):
    pfile.write("%s %s\n" % ((y_test[x]), (predicted[x])))
    xx.append(x_test[x])
    ya.append(y_test[x])
    yy.append(predicted[x])
pfile.close()
