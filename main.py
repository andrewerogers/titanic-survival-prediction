from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

CLASS_TRAIN = 1
GENDER_TRAIN = 4
AGE_TRAIN = 5

TRAIN_CELLS_LABELLED = open('titanic-dataset/titanic_train.csv', 'r')
TEST_CELLS_LABELLED = open('titanic-dataset/titanic_test.csv', 'r')


# process_file splits the file into features and reviews
def process_file(file, idx_list):
    features = []
    labels = []
    for line in file:
        s = line.split(',')
        feature = []
        for idx in idx_list:
            if s[idx] == 'male':
                s[idx] = 0
            elif s[idx] == 'female':
                s[idx] = 1
            elif s[idx] == '':
                s[idx] = 0
            feature.append(s[idx])

        features.append(feature)
        labels.append(s[len(s)-1])
    return features, labels


# train_model trains logistic regression model and displays
# cross validation metrics to use for model tuning.
def train_model(features, labels):
    x_train, x_test, y_train, y_test = train_test_split(features, labels,
                                                        test_size=0.1)
    clf = LogisticRegression(random_state=0).fit(x_train, y_train)
    acc = clf.score(x_test, y_test)
    print(f"mean-accuracy: {acc * 100}%")
    return clf


features_train, labels_train = process_file(TRAIN_CELLS_LABELLED, [CLASS_TRAIN, GENDER_TRAIN, AGE_TRAIN])
model = train_model(features_train, labels_train)
# {class, sex, age} for predict
print(model.predict([[3, 0, 28.0]]))
pickle.dump(model, open("./out/pickled_model.p", "wb"))
