from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

def linear_regression(X_train, y_train, X_test, y_test):
  num_cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
  scaler = StandardScaler()
  scaler.fit(X_train[num_cols])
  X_train[num_cols] = scaler.transform(X_train[num_cols])
  reg=LinearRegression()
  print(X_train.shape, y_train.shape)

  reg.fit(X_train, y_train)

  mean_absolute_error(y_train,reg.predict(X_train))
  X_test[num_cols] = scaler.transform(X_test[num_cols])
  y_pred = reg.predict(X_test)
  print(mean_absolute_error(y_test,y_pred), mean_squared_error(y_test,y_pred)**0.5)