from tools import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import lightgbm as lgb

seed = 42
target = 'label'

X, y = preprocessing.load_valid_data('dataset/train.csv', target=target)

X = preprocessing.feature_engineering_method(X)
model = lgb.LGBMClassifier()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

scores = 100 * cross_val_score(model, X, y, cv=KFold(10, shuffle=True, random_state=seed), scoring='f1_macro',
                               verbose=1, n_jobs=-1)
print("%0.4f score with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
