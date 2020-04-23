'''Problem Statement - Consider only the below columns and prepare a prediction model for predicting Price.

Corolla<-Corolla[c("Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight")] '''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy import stats

'''Reading the dataset'''

df = pd.read_csv("ToyotaCorolla.csv",encoding= 'unicode_escape')

#Dropping the first column as it is insignificant
df.drop(['Id'],axis = 1, inplace = True)

'''-----------------------Exploratory Data Analysis(EDA)------------------------------------------------'''
##Considering the only given columns as per the problem Statement

param_1 = df['Price']
param_2 = df['Age_08_04']
param_3 = df['HP']
param_4 = df['cc']
param_5 = df['Doors']
param_6 = df['Gears']
param_7 = df.iloc[:,15:17]
param_8 = df['KM']

df1 = pd.concat([param_1,param_2,param_3,param_4,param_5,param_6,param_7,param_8],axis=1)

df1.boxplot()

##Checking for missing values
def missing_values_table(df):
        mis_val = df_new.isnull().sum()
        mis_val_percent = 100 * df_new.isnull().sum() / len(df_new)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)
        print ("Your selected dataframe has " + str(df_new.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        return mis_val_table_ren_columns

missing_values_table(df1)

'''##Outlier Detection
Q1 = df1.quantile(0.25)
Q3 = df1.quantile(0.75)
IQR = Q3 - Q1
IQR
print(df1 < (Q1 - 1.5 * IQR))'''

sns.heatmap(df1.corr(),xticklabels='auto', cmap=None, center=None, robust=False, annot=True, yticklabels='auto')
sns.pairplot(df1)
df1.skew() ## cc column has a skewness of 27.89 hence needs to be normalized. This is done using log() and passing the
## same through a lambda function
df1['cc'] = df1['cc'].map(lambda i : np.log(i) if i >0 else 0)
df1.hist(bins=20)

## Using statsmodel.formula.api package we identified the significant features of the given dataset. Hence after
## plotting influencedindex plot we found that index no - 80,221,601,960 index elements have high influence on the Price
## Hence we dropped the same and prepared the final dataset which is to be passed to the train_test split package for deployement
df1.drop(df.index[[80,221,601,960]], inplace=True)

model_1 = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=df1).fit()
model_1.summary()
sm.graphics.plot_partregress_grid(model_1)
sm.graphics.influence_plot(model_1)


price_pred = model_1.predict(df1[['Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight']])
price_pred

er= df1.Price - price_pred

rmse=np.sqrt(np.mean(er*er))
rmse ##1201.17

resid = pd.DataFrame(pd.Series(df1['Price']-price_pred))
resid
resid.mean()
resid.describe()
resid.rename(columns={"Price_Pred":"Residuals"},inplace=True)

sns.regplot(x=price_pred, y=df1['Price'], color="green")


'''-------------------------------------Feeding Data into Model---------------------'''

''' Using sklearn.linear_model package
x = df1.iloc[:, 1:]
y = df1.iloc[:,0]

### Splitting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state = False)

reg = LinearRegression()
reg.fit(X_train,y_train)

# regression coefficients 
print('Coefficients: \n', reg.coef_) 
  
# variance score: 1 means perfect prediction 
print('Variance score: {}'.format(reg.score(X_test, y_test)))

# plot for residual error 
  
## setting plot style 
plt.style.use('fivethirtyeight') 
  
## plotting residual errors in training data 
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train, color = "green", s = 10, label = 'Train data') 
  
## plotting residual errors in test data 
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test, color = "blue", s = 10, label = 'Test data') 

## plotting line for zero residual error 
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2) 

## plotting legend 
plt.legend(loc = 'upper right') 
  
## plot title 
plt.title("Residual errors") 
  
## function to show plot 
plt.show()'''

'''Using statsmodel.formula.api
Price_pred=model_1.predict(df_final[["Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]])
Price_pred

er=df_final.Price-Price_pred
er
plt.boxplot(er)

rmsee=np.sqrt(np.mean(er*er))
rmsee

resid=pd.DataFrame(pd.Series(df_final['Price']-Price_pred))
resid
resid.mean()
resid.describe()
resid.rename(columns={"Price":"Residuals"},inplace=True)

'''''''Fitted values vs residuals
plt.scatter(x=Price_pred, y=resid, color="red")

Actual values vs fitted values
plt.scatter(x=Price_pred, y=df_final["Price"], color="green")'''
