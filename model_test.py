from sklearn.datasets import load_iris
import pandas as pd
from lib.models.svc import SVCModel as TestModel # CHANGE THIS
from lib.report import Report
from sklearn.model_selection import train_test_split

def get_bench_dataset():
    # Test algos on a well-known dataset
    # https://janakiev.com/notebooks/keras-iris/
    from sklearn.preprocessing import StandardScaler
    iris = load_iris()
    df = pd.DataFrame(data=iris['data'], columns=['sepal_length','sepal_width','petal_length','petal_width'])
    y = iris['target']
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df.values)
    input = pd.DataFrame(scaled, columns=df.columns, index=df.index)
    input['target'] = y
    df = input.sample(frac=1) # Shuffle the dataset
    return df.loc[:, df.columns != 'target'].values, df['target'].values

m = TestModel()
x,y = get_bench_dataset()
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.66, shuffle=False)
m.fit(x_train, y=y_train)
pred = m.predict(x_test)
r = Report(prediction=pred, labels=y_test, classifier=m, parameters=m.params)
print('{} accuracy: {}'.format(str(r), str(r.accuracy())))