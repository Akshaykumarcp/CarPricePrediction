import pandas as pd
import seaborn as sb

dataset = pd.read_csv("dataset/archive/car data.csv")

dataset.shape

dataset.columns

# check for unique values in categorical features
dataset['Seller_Type'].unique()
dataset['Transmission'].unique()
dataset['Owner'].unique()

# check for null values in the dataset
dataset.isnull().sum()

dataset.describe()

# there is year column, there will problem if the year is too old.
# so substract by current year and create a new column

dataset['Current_Year'] = 2020

dataset['no_year'] = dataset['Current_Year'] - dataset['Year']

# drop Current_Year and Year columns
dataset.drop(columns=['Current_Year','Year'],inplace=True)

# based on car name, we cannot predict price, so drop it
dataset.drop('Car_Name',axis=1,inplace=True)

# convert categorical features values into numerical values
dataset = pd.get_dummies(dataset,drop_first=True)

# find correlation
dataset.corr()

sb.pairplot(dataset) ##no much info

cormat = dataset.corr()
top_cor = cormat.index
# plot heatmap
g = sb.heatmap(dataset[top_cor].corr(),annot=True,cmap='RdYlGn')

# divide dataset into independent and dependent features
X = dataset.iloc[:,1:]
y = dataset.iloc[:,0]

# FEATURE IMPORTANCE
from sklearn.ensemble import ExtraTreesRegressor
imp_features_model = ExtraTreesRegressor()
imp_features_model.fit(X,y)

imp_features_model.feature_importances_

# lets plot it
#plot graph of feature importances for better visualization
feat_importances = pd.Series(imp_features_model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

from sklearn.ensemble import RandomForestRegressor

RFRModel = RandomForestRegressor()

#hyperparameter tuning
import  numpy as np
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

from sklearn.model_selection import RandomizedSearchCV

#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)

# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations
RFM = RandomizedSearchCV(estimator = RFRModel, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)

RFM.fit(X_train,y_train)

prediction = RFM.predict(X_test)

sb.distplot(y_test-prediction) # Normal distribution, so we can say, model is giving good results

sb.scatterplot(y_test,prediction) # prediction is linearly available so good prediction

import pickle
# open file and write
file = open('random_forest_reg_model.pkl','wb')

pickle.dump(RFM,file)