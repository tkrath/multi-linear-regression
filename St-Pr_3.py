import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pickle

'''----------------- Reading the data ---------------------'''

df = pd.read_csv("50_Startups.csv")

####---------------------Renaming the column------------------####
df.columns = "RD_Exp", "Admin", "Mkt_exp", "St", "Profit"

####---------------------For checing the outlier visually---------------####
df.boxplot()

####--------------------For checking the relationshiop between variables visually--------#########
sns.pairplot(df)
sns.heatmap(df.corr(),linewidth = 0.2, vmax=1.0, square=True, linecolor='red',annot=True)

model1 = smf.ols('Profit~RD_Exp+Admin+Mkt_exp+St',data =df).fit()
model1.summary()

'''
-------------------Model Stats-----------------------------------

R-squared:                       0.951
Adj. R-squared:                  0.945
F-statistic:                     169.9
Prob (F-statistic):           1.34e-27
Log-Likelihood:                -525.38
AIC:                             1063.
BIC:                             1074.                                    
'''

sm.graphics.influence_plot(model1)
df_new=df.drop(df.index[[46,49]], axis = 0)

final_model = smf.ols('Profit~RD_Exp+Admin+Mkt_exp', data=df_new).fit()
final_model.summary()

profit_pred=final_model.predict(df_new[['RD_Exp','Admin','Mkt_exp']])
profit_pred

sm.graphics.plot_partregress_grid(final_model)

er=df_new.Profit-profit_pred
er
plt.boxplot(er)

rmsee=np.sqrt(np.mean(er*er))
rmsee

resid=pd.DataFrame(pd.Series(df_new['Profit']-profit_pred))
resid
resid.mean()
resid.describe()
resid.rename(columns={"Profit":"Residuals"},inplace=True)

'''Fitted values vs residuals'''
plt.scatter(x=profit_pred, y=resid, color="red")

'''Actual values vs fitted values'''
plt.scatter(x=profit_pred, y=df_new["Profit"], color="green")
sns.regplot(x=profit_pred, y=df_new["Profit"], color="green")

''' Splitting the dataset'''
df_train,df_test  = train_test_split(df_new,test_size = 0.2, random_state =0)

'''Building the model with train data set'''
model_train = smf.ols('Profit~RD_Exp+Admin+Mkt_exp',data=df_train).fit()
model_train.summary()

# train_data prediction
train_pred = model_train.predict(df_train)

# train residual values 
train_resid  = train_pred - df_train.Profit

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid)) ##7917.8104
train_rmse


# prediction on test data set 
test_pred = model_train.predict(df_test)

# test residual values 
test_resid  = test_pred - df_test.Profit

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid)) ##4734.365
test_rmse

#Saving the file the dump
pickle.dump(model_train,('final_model.pkl','wb'))
