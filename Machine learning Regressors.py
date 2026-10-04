import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['HousePrice'] = housing.target
print(df.head(1000).to_string())
df.info()
df.isnull().sum()
print("----outlier detection----")

print("1) interquartile range")
Q1,Q3 = np.percentile(df['Population'], [25, 75])
IQR = Q3 - Q1
upper = Q3 + 1.5 * IQR
lower = Q1 - 1.5 * IQR
df = df[(df['Population'] >= lower) & (df['Population'] <= upper)]
outliers_housing = ((df<Q1 - 1.5*IQR) | (df>Q3 + 1.5*IQR))
print("-----Outlier Removed Data-----")
print(df.head(1000))
print(outliers_housing.sum())

print("2) Graphical")
plt.figure(figsize=(10,10))
sns.boxplot(data = df)
plt.xticks(rotation=45)
plt.title('Box Plot')
plt.show()

'''In the data preprocessing step, I used the statistical method of interquartile range 
and boxplot for graphical detection of outliers.There were around 1000+ outliers in the population column
and is removed by this line of code - df = df[(df['Population'] >= lower) & (df['Population'] <= upper)]'''

#preprocessing
#1)feature selection
X = df.drop(columns = ['HousePrice'])
y = df['HousePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#2)scaling
scaler  = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
'''Transforming features so they have a mean of 0 and a standard deviation of 1 using tools, example: StandardScaler.
Algorithms which relies on distance calculations(Support vector machines,Linear regression) are sensitive towards feature
magnitude. This causes features with large scale to dominate over the features with small scale.Feature scaling ensure equal
contribution of all feature.'''

print('----Linear regressor----')
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)
#prediction
y_pred = model.predict(X_test_scaled)
#real prediction using the model
factors ={'MedInc' : 4.5,
          'HouseAge' : 25.0,
          'AveRooms':6.8,
          'AveBedrms':2.0,
          'Population':1234.0,
          'AveOccup':2.54,
          'Latitude':37.88,
          'Longitude':-122.26}
input_df = pd.DataFrame([factors])
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)
print("The predicted value is : ",prediction)
print('----Performance of linear regressor----')
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print("mae",mae)
print("mse",mse)
print("r2",r2)
'''Assumes a linear relationship between the input features and the target variable.
 It calculates a straight line (or hyperplane in higher dimensions) that minimizes 
 the sum of squared differences between actual and predicted values.Acts as a fast and interpretable
baseline model to check if basic linear correlations exist between housing features and median prices.'''

#Decision tree
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train_scaled, y_train)
y_pred = dt.predict(X_test_scaled)
print('---performance of decision regressor---')
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print("mae",mae)
print("mse",mse)
print("r2",r2)
'''Recursively splits the dataset into branches based on feature threshold values, 
creating a tree structure where each leaf node predicts a continuous target value.'''

print('----Random forest regressor----')
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100,random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)
print('---performance of random forest regressor---')
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print("mae",mae)
print("mse",mse)
print("r2",r2)
'''An ensemble learning method that builds numerous decision trees on random 
subsets of the data and averages their predictions.'''
# Gradient booster regressor
print('----gradient booster regressor----')
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(X_train_scaled, y_train)
y_pred = gbr.predict(X_test_scaled)
print('---performance of gradient boosting regressor---')
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print("mae",mae)
print("mse",mse)
print("r2",r2)
'''An ensemble technique that builds trees sequentially. 
Each new tree is trained specifically to predict and minimize the errors (residuals) left behind by the previous trees.'''

print('----SVM regressor----')
from sklearn.svm import SVR
svm = SVR(kernel='linear', C=50)
svm.fit(X_train_scaled, y_train)
y_pred = svm.predict(X_test_scaled)
print('---performance of SVM regressor---')
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print("mae",mae)
print("mse",mse)
print("r2",r2)
'''Finds an optimal hyperplane that fits the data points within a defined margin of tolerance, 
using kernel functions to handle linear or non-linear boundaries'''

'''CONCLUSION:-Ensemble methods (Random Forest and Gradient Boosting) significantly outperform individual base learners and linear models. 
Because real-world real estate data features complex spatial and economic dependencies, 
tree-ensemble aggregation provides the most reliable predictive accuracy.'''


















