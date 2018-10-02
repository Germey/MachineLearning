import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df = pd.read_csv('data.csv')

df['outlook'], _ = pd.factorize(df['outlook'])
df['temperature'], _ = pd.factorize(df['temperature'])
df['humidity'], _ = pd.factorize(df['humidity'])
df['windy'], _ = pd.factorize(df['windy'])

ohe = OneHotEncoder(sparse=False)

x_data = df[['outlook', 'temperature', 'humidity', 'windy']].values[:-1]
ohe.fit(x_data)
x_data = ohe.transform(x_data)

y_data = df['play'].values[:-1]
le = LabelEncoder()
le.fit(y_data)
y_data = le.transform(y_data)

gnb = GaussianNB()
gnb.fit(x_data, y_data)

x_test = df[['outlook', 'temperature', 'humidity', 'windy']].values[-1:]
x_test = ohe.transform(x_test)
y_test = gnb.predict(x_test)
print('是否打球：', y_test[0])
