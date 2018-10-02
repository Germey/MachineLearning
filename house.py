from sklearn.linear_model import LinearRegression

x_data = [
    [6000, 58],
    [9000, 77],
    [11000, 89],
    [15000, 54]
]
y_data = [
    30000, 55010, 73542, 63201
]

lr = LinearRegression()
lr.fit(x_data, y_data)
print('方程为 y={w1}x1+{w2}x2+{b}'.format(w1=round(lr.coef_[0], 2),
                                       w2=round(lr.coef_[1], 2),
                                       b=lr.intercept_))
x_test = [[12000, 60]]
print('住房面积为', lr.predict(x_test)[0])
