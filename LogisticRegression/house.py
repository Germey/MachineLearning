from sklearn.linear_model import LogisticRegression

x_data = [
    [6000, 58],
    [9000, 77],
    [11000, 89],
    [15000, 54]
]
y_data = [
    0, 0, 1, 1
]

lr = LogisticRegression()
lr.fit(x_data, y_data)
x_test = [[12000, 60]]
print('Intercept', lr.intercept_)
print('Coef', lr.coef_)
print('款项是否可以立即到账', lr.predict(x_test)[0])
