#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Loading all our libraries to set up enviornment

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl 
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Importing our raw dataset for autocars
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
df = pd.read_csv(r'C:\Users\super\Documents\Excel\autocars.csv', names = headers)
df.head()


# In[3]:


#We will now proceed with data cleaning starting with checking for duplicates
print(df.shape)
df.duplicated().sum()


# In[4]:


#we see that there are no duplicates present in this dataset.
#Now let us move on to indentify and handle missing data.
#The first step would be to convert all '?' to 'NaN'
df.replace('?',np.nan, inplace = True)
df.head()


# In[5]:


#We will now check for missing values
missing_df = df.isnull()

for column in missing_df.columns.values.tolist():
    print(column)
    print(missing_df[column].value_counts())
    print("")

#Any results that show as 'True' would mean that there are missing values in the column


# We notice that there are 7 columns with missing values, we will replace them with the following methods.
# 
# Replace missing values with mean:
# normalized-losses: 41 missing values
# bore: 4 missing values
# stroke: 4 missing values
# horsepower: 2 missing values
# peak-rpm: 2 missing values
#     
# Replace missing values with mode:
# num-of-doors: 2 missing values
#     
# Drop missing values:
# price: 4 missing values
# We need to drop this row because this is what we want to predict using the other variables.

# In[7]:


#Replacing Missing values with Mean

#normalized-losses
avg_norm_loss = df['normalized-losses'].astype('float').mean(axis = 0)
df['normalized-losses'].replace(np.nan, avg_norm_loss, inplace = True)

#bore
avg_bore = df['bore'].astype('float').mean(axis = 0)
df['bore'].replace(np.nan, avg_bore, inplace = True)

#stroke
avg_stroke = df['stroke'].astype('float').mean(axis = 0)
df['stroke'].replace(np.nan, avg_stroke, inplace = True)

#horsepower
avg_horsepower = df['horsepower'].astype('float').mean(axis = 0)
df['horsepower'].replace(np.nan, avg_horsepower, inplace = True)

#peak-rpm
avg_peak_rpm = df['peak-rpm'].astype('float').mean(axis = 0)
df['peak-rpm'].replace(np.nan, avg_peak_rpm, inplace = True)


# In[8]:


#Replacing Missing Values with Mode

mode_num_doors = df['num-of-doors'].value_counts().idxmax()
df['num-of-doors'].replace(np.nan, mode_num_doors, inplace = True)


# In[9]:


#Dropping missing values

df.dropna(subset=['price'],axis = 0, inplace = True)
df.reset_index(drop=True, inplace=True)
df.head()


# In[10]:


#Now let us check if all the data types are correctly corresponding to the data present.

df.dtypes


# In[12]:


#Let us correct some of the data types and check our work.

#Let us change normalized-losses to "int"
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
#Let us change bore and stroke to "float"
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
#Let us change price to "float"
df[["price"]] = df[["price"]].astype("float")
#Let us change horsepower and peak-rpm to "float"
df[["horsepower","peak-rpm"]] = df[["horsepower","peak-rpm"]].astype("float")
df.dtypes


# In[13]:


#Let us now export our dataset since its cleaned
df.to_csv('clean_autocars.csv')


# Let us now proceed with the Data Analysis.
# 
# First we are going to do some descriptive Statistical Analysis.

# In[14]:


#numeric variables
df.describe()


# In[15]:


#categorical variables
df.describe(include=['object'])


# Now that we have found all the values and separated them by their categories to get a better understanding, we will now proceed with the next step which would be finding the correlation to see which relationships have the strongest relationship with the price.

# In[16]:


df.corr().loc[:,'price'].to_frame().sort_values(by='price')


# All those in the positive have a higher correlation with price. The list above is set from the least correlated to the most.
# 
# Since the top 5 would be:
# engine-size
# curb-weight
# horsepower
# width
# length
# 
# We will now check the p-values to test if these 5 are statistically significant or not.

# In[17]:


from scipy import stats


# In[27]:


#find all of the pearson's correlation coefficients, and their corresponding p-values. 
engine_corr, engine_pvalue = stats.pearsonr(df['engine-size'], df['price'])
curb_corr, curb_pvalue = stats.pearsonr(df['curb-weight'], df['price'])
horsepower_corr, horsepower_pvalue = stats.pearsonr(df['horsepower'], df['price'])
width_corr, width_pvalue = stats.pearsonr(df['width'], df['price'])
highway_corr, highway_pvalue = stats.pearsonr(df['highway-mpg'], df['price'])


print('engine-size: ','\nCorrelation: ',engine_corr,'\nP-value:',engine_pvalue,'\n')
print('curb-weight: ','\nCorrelation: ',curb_corr,'\nP-value:',curb_pvalue,'\n')
print('horsepower: ','\nCorrelation: ',horsepower_corr,'\nP-value:',horsepower_pvalue,'\n')
print('width: ','\nCorrelation: ',width_corr,'\nP-value:',width_pvalue,'\n')
print('highway-mpg: ','\nCorrelation: ',highway_corr,'\nP-value:',highway_pvalue,'\n')


# Now that we have shown that the p-values are statistically significant. Let us view the relationships with the other categorical fields to see if they would have a significant impact.

# In[28]:


#Let us check the relationship between make and price using boxplot
sns.boxplot(x="make", y="price", data=df)


# In[29]:


sns.boxplot(x="fuel-type", y="price", data=df)


# In[30]:


sns.boxplot(x="aspiration", y="price", data=df)


# In[31]:


sns.boxplot(x="num-of-doors", y="price", data=df)


# In[32]:


sns.boxplot(x="body-style", y="price", data=df)


# In[33]:


sns.boxplot(x="drive-wheels", y="price", data=df)


# In[34]:


sns.boxplot(x="engine-location", y="price", data=df)


# In[35]:


sns.boxplot(x="engine-type", y="price", data=df)


# In[36]:


sns.boxplot(x="num-of-cylinders", y="price", data=df)


# In[37]:


sns.boxplot(x="fuel-system", y="price", data=df)


# Since there seems to be an overlap between the categorical values. Let us use the analysis of variance to get a better view.

# In[38]:


drive_test = df[['drive-wheels','price']].groupby(['drive-wheels'])
engine_test = df[['engine-location','price']].groupby(['engine-location'])


# Now that we set up drive_test and engine_test. Let us check the unique values before checking the variance.

# In[39]:


drive_test['drive-wheels'].unique()


# In[40]:


engine_test['engine-location'].unique()


# In[41]:


# Now let us check the fvalue and pvalue for all the unique values 
drive_fvalue, drive_pvalue = stats.f_oneway(drive_test.get_group('4wd')['price'], drive_test.get_group('fwd')['price'], drive_test.get_group('rwd')['price'])
engine_fvalue, engine_pvalue = stats.f_oneway(engine_test.get_group('front')['price'], engine_test.get_group('rear')['price'])

print('Drive-wheels ANOVA results: ','\nF-value: ',drive_fvalue,'\nP-value: ',drive_pvalue,'\n')
print('Engine-location ANOVA results: ','\nF-value: ',engine_fvalue,'\nP-value: ',engine_pvalue)


# In[42]:


#Let us now take a deeper look by diving into the drive-wheels categories.

#4wd and fwd
fvalue_1, pvalue_1 = stats.f_oneway(drive_test.get_group('4wd')['price'], drive_test.get_group('fwd')['price'])

#4wd and rwd
fvalue_2, pvalue_2 = stats.f_oneway(drive_test.get_group('4wd')['price'], drive_test.get_group('rwd')['price'])

#fwd and rwd
fvalue_3, pvalue_3 = stats.f_oneway(drive_test.get_group('fwd')['price'], drive_test.get_group('rwd')['price'])

print('4wd and fwd: ', '\nF-value: ', fvalue_1, '\nP-value: ', pvalue_1,'\n')
print('4wd and rwd: ', '\nF-value: ', fvalue_2, '\nP-value: ', pvalue_2,'\n')
print('fwd and rwd: ', '\nF-value: ', fvalue_3, '\nP-value: ', pvalue_3,'\n')


# We notice that fwd and rwd have significantly different values compared to the rest. So let us now create some dummy variables for the analysis using this set.

# In[43]:


#drive-wheels
drive_dummy_var = pd.get_dummies(df['drive-wheels'])
drive_dummy_var.drop('4wd', axis = 1, inplace = True)
drive_dummy_var.rename(columns = {'fwd':'drive-wheels-fwd','rwd':'drive-wheels-rwd'}, inplace=True)


#engine-location
engine_dummy_var = pd.get_dummies(df['engine-location'])
engine_dummy_var.rename(columns = {'front':'engine-location-front','rear':'engine-location-rear'},inplace=True)


# In[44]:


drive_dummy_var.head(), engine_dummy_var.head()


# In[45]:


#Let's add these dummy variables to our original dataframe.

df = pd.concat([df, drive_dummy_var, engine_dummy_var], axis = 1)
df.head()


# Now that we have found our variables let us start the model development and evaluation.
# 
# Numerical Variables:
# 1. engine-size
# 2. curb-weight
# 3. horsepower
# 4. width
# 5. highway-mpg
# 
# Categorical Variables:
# 1. drive-wheels (fwd and rwd categories)
# 2. engine-location (front and rear categories)

# In[46]:


#Let us check the multi-collinearity between variables that we found.
df_model = df[['engine-size','curb-weight','horsepower','width','highway-mpg','drive-wheels-fwd', 
               'drive-wheels-rwd', 'engine-location-front','engine-location-rear','price']]
corr = df_model.corr()
corr.style.background_gradient(cmap='coolwarm')


# Since we are checking the compatiblity with price now, we can see that drive-wheels and engine location are not very compatible with price so let us remove them and check again.

# In[47]:


df_model = df[['engine-size','curb-weight','horsepower','width','highway-mpg','price']]
corr = df_model.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[48]:


#Let's also compute the Variance Inflation Factor(VIF) to get more information on the multi-collinearity.
from statsmodels.stats.outliers_influence import variance_inflation_factor
df_model_noprice = df[['engine-size','curb-weight','horsepower','width','highway-mpg']]

vif = pd.DataFrame()
vif["features"] = df_model_noprice.columns
vif["vif_Factor"] = [variance_inflation_factor(df_model_noprice.values, i) for i in range(df_model_noprice.shape[1])]
vif


# Let's try and remove some variables to reduce the collinearity.
# 
# We need to keep engine-size and curb-weight as they are most highly correlated with price. So we should observe how removing horsepower, width, and highway-mpg would impacts our VIF.

# In[49]:


#Let us first remove horsepower and check.
df_model_noprice = df[['engine-size','curb-weight','highway-mpg','width']]
                       
vif = pd.DataFrame()
vif["features"] = df_model_noprice.columns
vif["vif_Factor"] = [variance_inflation_factor(df_model_noprice.values, i) for i in range(df_model_noprice.shape[1])]
vif


# In[50]:


#Let us remove highway-mpg now
df_model_noprice = df[['engine-size','curb-weight','horsepower','width']]
                       
vif = pd.DataFrame()
vif["features"] = df_model_noprice.columns
vif["vif_Factor"] = [variance_inflation_factor(df_model_noprice.values, i) for i in range(df_model_noprice.shape[1])]
vif


# In[51]:


#Let us now remove width
df_model_noprice = df[['engine-size','curb-weight','horsepower','highway-mpg']]
                       
vif = pd.DataFrame()
vif["features"] = df_model_noprice.columns
vif["vif_Factor"] = [variance_inflation_factor(df_model_noprice.values, i) for i in range(df_model_noprice.shape[1])]
vif


# Removing width has the greatest impact on the VIF, but our model still has significant multi-collinearity. The most important variables for our model are:
# 
# 1. engine-size
# 2. curb-weight
# 3. horsepower
# 4. highway-mpg
# 
# Since we are developing a model with multiple variables that deal with multi-collinearity, a ridge regression model will be best to help us predict price accurately.

# In[52]:


#Let us use ridge regression to predict the price

df_model = df[['engine-size','curb-weight','horsepower','highway-mpg','price']]

#select our independent variables and our dependent variable
y = df_model['price']
x = df_model.drop('price',axis=1)

#split our data into train and test datasets

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1)


# In[55]:


#Now we can use Grid Search to find the best alpha for our Ridge Regression model.

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

#select potential parameters
parameters = [{'alpha':np.linspace(0.001,50000)}]

#create ridge regression object
RR = Ridge()

#create ridge grid search object
Grid = GridSearchCV(RR, parameters, cv=4)

#train the model
Grid.fit(x_train, y_train)

#find the Ridge Regression Model with the best value for alpha
Best_RR = Grid.best_estimator_
Best_RR


# In[54]:


# Let's now test how good our model is using our test data by finding the R-squared value.
Best_RR.score(x_test,y_test)


# This Ridge regression model predicts 78.3% of the variation in the price.
# 
# Let's plot the distribution of the predicted values vs. the actual values to see how well our Ridge Regression model predicts price.

# In[56]:


#First, we will predict new values using our Ridge Regression Model.
y_hat = Best_RR.predict(x_test)


# In[57]:


#Now, we can plot the predicted values vs. the actual values.
def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 6
    height = 4
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')
    plt.legend()

    plt.show()
    plt.close()


# In[58]:


Title = 'Distribution of Predicted Test Output vs. Actual Test Data'
DistributionPlot(y_test, y_hat, "Actual Test Data",'Predicted Test Values', Title)


# Now that we have this let's try and compare this to a ridge regression model with polynomial features of 2 degrees so we have a better view

# In[59]:


from sklearn.pipeline import Pipeline
#set parameters for the Grid Search
parameters2 = [{'alpha':np.linspace(23,24)}]

#set the steps in the pipeline 
Input = [('scale',StandardScaler()),('polynomial',PolynomialFeatures(degree=2)),
         ('mode',Ridge(23.5))]

#Create the Pipeline object with the steps specified above. 
pipe = Pipeline(Input)

#Train the model
pipe.fit(x_train,y_train)


# In[60]:


#get the R-squared score
pipe.score(x_test,y_test)


# This Polynomial Ridge regression model predicts about 80.9% of the variation in the price. This model is a better predictor of price than our other model.
# 
# So now let us plot the predicted values vs actual values.

# In[61]:


#predicted values
yhat = pipe.predict(x_test)

#plot the graph
Title = 'Distribution of Predicted Test Output vs. Actual Test Data'
DistributionPlot(y_test, y_hat, "Actual Test Data",'Predicted Test Values', Title)


# In conclusion based on the different analysis method the one that gave us the best prediction is Ridge Regression model with Polynomial Features of Degree 2.
