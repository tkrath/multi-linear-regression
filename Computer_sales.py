import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import preprocessing
from scipy import stats
import math
import pickle

#Reading the data
df = pd.read_csv("Computer_Data.csv")

#Changing the name of the column and dropping the first index column as the same doesn't play any significance

df.columns = "Index", "Sales_Price", "Speed", "HDD", "RAM", "Screen", "cd", "Multi", "premium","ads","trend"

df.drop(["Index"], axis = 1, inplace=True)

####---------------------For checing the outlier visually---------------####
df.boxplot()

####--------------------For checking the relationshiop between variables visually--------#########
sns.pairplot(df)
sns.heatmap(df.corr(),linewidth = 0.2, vmax=1.0, square=True, linecolor='red',annot=True)

###df.columns

##Converting catgorical values to numeric
cd_drive = pd.get_dummies(df['cd'], drop_first= True)
multi = pd.get_dummies(df['Multi'], drop_first= True)
Prem = pd.get_dummies(df['premium'], drop_first= True)

##Joining the dummy variable columns to the main dataframe
df=pd.concat([df,cd_drive,multi,Prem], axis=1)
df=df.drop(['cd','Multi','premium'], axis=1)

#Renaming the columns for understanding
df.columns = "Sales_Price", "Speed", "HDD", "RAM", "Screen","ads","trend","cd","multi","Prem"

'''df['Screen'].min() #17
df['Screen'].max() #14
df['HDD'].max() #2100
df['HDD'].min() #80
df['RAM'].max() #32
df['RAM'].min() #02
df['Speed'].max() #100
df['Speed'].min() #025
df['ads'].max() #339
df['ads'].min() #39'''

df.skew()
##Checking for missing values

'''def missing_values_table(df):
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        return mis_val_table_ren_columns

missing_values_table(df)'''

'''#Outlier Detection
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
IQR
print(df < (Q1 - 1.5 * IQR)) ## Wherever the value supersides with 'True' it will be considered as an outlier'''

'''This method deletes the complete data by taking IQR into consideration and 
found that it deletes almost 50% of the data which is not recommended
#df_out = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
#print(df_out.shape)
#df_out.describe() (3273,10)'''

'''Outlier treatment - Here we replace the extreme values with the median values. Also known as Median Imputation'''''''
##For Sales_Price
print(df['Sales_Price'].quantile(0.50)) 
print(df['Sales_Price'].quantile(0.95)) 
df['Sales_Price'] = np.where(df['Sales_Price'] > 3209, 2144, df['Sales_Price'])

##For Speed
print(df['Speed'].quantile(0.50)) 
print(df['Speed'].quantile(0.95)) 
df['Speed'] = np.where(df['Speed'] > 100, 50, df['Speed'])

##For HDD
print(df['HDD'].quantile(0.50)) #340
print(df['HDD'].quantile(0.95)) #1000
df['HDD'] = np.where(df['HDD'] > 1000, 340, df['HDD'])

##For RAM
print(df['RAM'].quantile(0.50)) #8.0 
print(df['RAM'].quantile(0.95)) #16.79
df['RAM'] = np.where(df['RAM'] > 16.79, 8.0, df['RAM'])

##For Speed
print(df['Screen'].quantile(0.50)) #14.0
print(df['Screen'].quantile(0.95)) #17.0
df['Screen'] = np.where(df['Screen'] > 17, 14, df['Screen'])

##For ads
print(df['ads'].quantile(0.50)) #246 
print(df['ads'].quantile(0.95)) #339
df['ads'] = np.where(df['ads'] > 339, 246, df['ads'])

##For Speed
print(df['trend'].quantile(0.50)) #16
print(df['trend'].quantile(0.95)) #29
df['trend'] = np.where(df['trend'] > 29, 16, df['trend'])'''

'''df["Prem"] = df["Prem"].map(lambda i: np.sqrt(i) if i > 0 else 0)
df['Prem'].skew()
df["Screen"] = df["Screen"].map(lambda i: np.log(i) if i > 0 else 0)
df['Screen'].skew()
df["HDD"] = df["HDD"].map(lambda i: np.log(i) if i > 0 else 0)
df['HDD'].skew()
df["RAM"] = df["RAM"].map(lambda i: np.log(i) if i > 0 else 0)
df['RAM'].skew()
df["multi"] = df["multi"].map(lambda i: np.exp(i) if i > 0 else 0)
df['multi'].skew()
df["ads"] = df["ads"].map(lambda i: np.log(i) if i > 0 else 0)
df['ads'].skew()
df.skew()'''

model=smf.ols('Sales_Price~Speed+HDD+RAM+Screen+ads+trend+cd+multi+Prem', data=df).fit()
model.summary()
sm.graphics.influence_plot(model)
sm.graphics.plot_partregress_grid(model)

'''Checking for VIF
# calculating VIF's values of independent variables
#Unlike R we don't have any inbuilt fuction for VIF so we write our own formula
#VIF=1/(1-R^2)

rsq_sp = smf.ols('Speed~HDD+RAM+Screen+ads+trend+cd+multi+Prem',data=df).fit().rsquared
rsq_sp #0.209
#We are doing Linear Regression of HP and other variables and capturing rsquared
vif_sp = 1/(1-rsq_sp) 

rsq_hd = smf.ols('HDD~Speed+RAM+Screen+ads+trend+cd+multi+Prem',data=df).fit().rsquared  
rsq_hd #0.76
vif_hd = 1/(1-rsq_hd)

rsq_ram = smf.ols('RAM~Speed+HDD+Screen+ads+trend+cd+multi+Prem',data=df).fit().rsquared  
rsq_ram #0.66
vif_ram = 1/(1-rsq_ram) 

rsq_scrn = smf.ols('Screen~Speed+HDD+RAM+ads+trend+cd+multi+Prem',data=df).fit().rsquared  
rsq_scrn #0.075
vif_scrn = 1/(1-rsq_scrn)

rsq_trend=smf.ols('trend~Speed+RAM+Screen+ads+cd+multi+Prem',data=df).fit().rsquared
rsq_trend #0.36
vif_trend = 1/(1-rsq_trend)

rsq_cd=smf.ols('cd~Speed+RAM+Screen+ads+trend+multi+Prem',data=df).fit().rsquared
rsq_cd #0.45
vif_cd= 1/(1-rsq_cd)

rsq_multi=smf.ols('multi~Speed+RAM+Screen+ads+trend+cd+Prem',data=df).fit().rsquared
rsq_multi #0.21
vif_multi= 1/(1-rsq_multi)

rsq_prem=smf.ols('Prem~Speed+RAM+Screen+ads+trend+multi+cd',data=df).fit().rsquared
rsq_prem #0.095
vif_prm= 1/(1-rsq_prem)

rsq_ads = smf.ols('ads~Speed+HDD+RAM+Screen+trend+cd+multi+Prem', data=df).fit().rsquared
rsq_ads


# Storing vif values in a data frame
d1 = {'Variables':['Speed','HDD','RAM','Screen','trend','cd','multi','Prem'],'VIF':[vif_sp,vif_hd,vif_ram,vif_scrn,vif_trend,vif_cd,vif_multi, vif_prm]}
#Creating a Dictionary
#One Column we have Variables
#Other Column we have VIF values
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As none of the value is above 10 we are not going to drop any column'''

## Checking the mean of error from the predicted values
sales_pred=model.predict(df[['Speed','HDD','RAM','Screen','ads','trend','cd','multi','Prem']])
sales_pred

er=df.Sales_Price-sales_pred
er
plt.boxplot(er)

rmse = np.sqrt(np.mean(er*er))
rmse #275
resid=pd.DataFrame(pd.Series(df['Sales_Price']-sales_pred))
resid
resid.mean()
resid.describe()
resid.rename(columns={"Sales_Price":"Residuals"},inplace=True)

plt.scatter(x=sales_pred, y=resid, color="red")
plt.hlines(y = 0, xmin = 0, xmax = 4500, linewidth = 2)
plt.scatter(x=sales_pred, y=df['Sales_Price'], color="red")
sns.regplot(x=sales_pred, y=df['Sales_Price'], color="green")
plt.show()

#splitting the data set

df_train,df_test  = train_test_split(df,test_size = 0.2, random_state =0)
  
''' This is the plotting of the error values, predicted values and residuals
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
plt.show() '''

'''##Testing the o/p
X1 = [[60,340,8,15,94,2,1,1,1]]
out1 = reg.predict(X1)
print('Sales_price for entered specifications computer is : ', out1)'''
