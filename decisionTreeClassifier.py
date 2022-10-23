from pandas import read_csv,DataFrame
from numpy import array
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

train_df = read_csv('training_demo.csv')
test_df = read_csv('Testing.csv')
test_df[test_df.duplicated()]
# print("A :".test_df[test_df.duplicated()])
# train_df.head()

# train_df.prognosis.value_counts().head()

X_train = train_df.iloc[:, :-1].values 
y_train = train_df.iloc[:, 132].values

X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, 132].values

# y_train
dt_clf = DecisionTreeClassifier(splitter='best', criterion='entropy', min_samples_leaf=2)

dt_clf.fit(X_train, y_train)

# X_test.shape

pred = dt_clf.predict(X_test)
results=DataFrame({
    'Actual': y_test, 
    'Predicted': pred
}).head(10)
# print(results)


print("Accuracy Score: ", accuracy_score(y_test, pred))
def predict_diasease(inp):
    return dt_clf.predict(array([inp]))

if __name__=="__main__":
    # inp.shape
    print(predict_diasease([1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))