# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Part 4: Churn Prediction
#
# Every company puts its efforts into knowing who their best customer are and then it also work hard on retaining them. That’s what makes **Retention Rate** is one of the most critical metrics.
#
# Retention Rate is an indication of how good is your product market fit (PMF). If your PMF is not satisfactory, you should see your customers churning very soon. One of the powerful tools to improve Retention Rate (hence the PMF) is Churn Prediction. By using this technique, you can easily find out who is likely to churn in the given period. 
#
# In this notebook, we will use a Telco dataset and go over following steps to develop churn prediction:
# * Exploratory data analysis
# * Feature engineering
# * Investigating how the features affect Retention by using Logistic Regression
# * Building a classification model with XGBoost

# %%
# import libraries
import pandas as pd
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import plotly.graph_objects as go
import plotly.figure_factory as ff

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

import growth_kit as gk

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ## Exploratory Data Analysis
# We start with checking out how our data looks like and visualize how it interacts with our label (churned or not?). 

# %%
df_data = pd.read_csv('data/churn-data.csv')
df_data.head()

# %%
df_data.info()

# %% [markdown]
# Our data fall under two categories:
# * Categorical features: gender, streaming tv, payment method &, etc.
# * Numerical features: tenure, monthly charges, total charges
#
# Now starting from the categorical ones, we shed light on all features and see how helpful they are to identify if a customer is going to churn.

# %%
# convert churn label to integer 0-No, 1-Yes
df_data['Churn'] = df_data['Churn'].replace({"Yes": 1, "No": 0})


# %% [markdown]
# #### Gender
# Let's start with how Churn rate looks with respect to Gender:

# %%
def plot_churn(data, category):
    # Chrun percentage and total count
    df_plot = df_data.groupby(category)['Churn'].agg(['mean', 'count']).reset_index()
    print(df_plot[[category, 'mean']])
    
    # Set up two plot figure
    fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(8,8))
    
    # Churn Rate
    sns.barplot(x=category, y='mean', palette=sns.color_palette("OrRd", 3), data=df_plot, ax=axs[0])
    axs[0].set(xlabel=None, ylabel='Churn Rate', title='{} based Churn Rate'.format(category))
    
    # Count of customers in each group
    sns.barplot(x=category, y='count', palette=sns.color_palette("Blues"), data=df_plot, ax=axs[1])
    axs[1].set(xlabel=category, ylabel='Total Customers')
    plt.show()



# %%
plot_churn(df_data, 'gender')

# %% [markdown]
# Female customers are more likely to churn vs. male customers, but the difference is minimal (~0.8%).

# %% [markdown]
# Let’s replicate this for all categorical columns.

# %% [markdown]
# #### InternetService

# %%
plot_churn(df_data, 'InternetService')

# %% [markdown]
# This chart reveals customers who have Fiber optic as Internet Service are more likely to churn. I normally expect Fiber optic customers to churn less due to they use a more premium service. But this can happen due to high prices, competition, customer service, and many other reasons.

# %% [markdown]
# #### Contract

# %%
plot_churn(df_data, 'Contract')

# %% [markdown]
# As expected, the shorter contract means higher churn rate.
#
# #### Tech Support

# %%
plot_churn(df_data, 'TechSupport')

# %% [markdown]
# Customers don’t use Tech Support are more like to churn (~25% difference).
#
# #### Payment Method

# %%
plot_churn(df_data, 'PaymentMethod')

# %% [markdown]
# Automating the payment makes the customer more likely to retain in your platform (~30% difference).
#
# #### Others

# %%
cat_columns = ['Partner', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
              'DeviceProtection', 'StreamingTV', 'StreamingMovies' ,'PaperlessBilling', ]

for col in cat_columns:
    plot_churn(df_data, col)

# %% [markdown]
# Other indicative columns are: Partner, Online Security, Online Backup, Paperless Billing.
#
# We are done with the categorical features. Let’s see how numerical features look like.
#
# #### Tenure
#
# To see the trend between Tenure and average Churn Rate, let’s build a scatter plot:

# %%
df_plot = df_data.groupby('tenure').Churn.mean().reset_index()
plot_data = [
    go.Scatter(
        x=df_plot['tenure'],
        y=df_plot['Churn'],
        mode='markers',
        name='Low',
        marker= dict(size= 7,
            line= dict(width=1),
            color= 'blue',
            opacity= 0.8
           ),
    )
]
plot_layout = go.Layout(
        yaxis= {'title': "Churn Rate"},
        xaxis= {'title': "Tenure"},
        title='Tenure based Churn rate',
        plot_bgcolor  = "rgb(243,243,243)",
        paper_bgcolor  = "rgb(243,243,243)",
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()

# %%
# tenure based churn rate
df_plot = df_data.groupby('tenure').Churn.mean().reset_index()
ax = sns.scatterplot(x='tenure', y='Churn', data=df_plot)
ax.set(title='Tenure based Churn Rate', ylabel='Churn Rate')

# %% [markdown]
# Super apparent that the higher tenure means lower Churn Rate. We are going to apply the same for Monthly and Total Charges.
#
# #### Monthly Charges

# %%
df_data.MonthlyCharges.describe()

# %%
# monthly charge
df_plot = (df_data.assign(MonthlyChargesBin = lambda x: x['MonthlyCharges'].astype(int))
                  .groupby('MonthlyChargesBin')['Churn'].mean()
                  .reset_index())
ax = sns.scatterplot(x='MonthlyChargesBin', y='Churn', data=df_plot)
ax.set(title='Monthly Charges vs. Churn Rate', ylabel='Churn Rate')

# %% [markdown]
# #### Total Charges

# %%
pd.to_numeric(df_data['TotalCharges'], errors='coerce').dropna().astype(int).head()

# %%
# total charge
df_plot = (df_data.assign(TotalChargesBin = lambda x: pd.to_numeric(x['TotalCharges'], errors='coerce'))
                    .dropna()
                    .astype({'TotalChargesBin': 'int32'})
                    .groupby('TotalChargesBin')['Churn'].mean()
                    .reset_index())
ax = sns.scatterplot(x='TotalChargesBin', y='Churn', data=df_plot)
ax.set(title='Total Charges vs. Churn Rate', ylabel='Churn Rate')

# %% [markdown]
# Unfortunately, there is no trend between Churn Rate and Monthly & Total Charges.
#
# ## Feature Engineering
# In this section, we are going to transform our raw features to extract more information from them. Our strategy is as follows:
# 1. Group the numerical columns by using clustering techniques
# 1. Apply Label Encoder to categorical features which are binary
# 1. Apply get_dummies() to categorical features which have multiple value
#
# ### Numerical Columns
# As we know from the EDA section, We have three numerical columns:
# * Tenure
# * Monthly Charges
# * Total Charges
#
# We are going to apply the following steps to create groups:
# * Using Elbow Method to identify the appropriate number of clusters
# * Applying K-means logic to the selected column and change the naming
# * Observe the profile of clusters
#
# #### Tenure Cluster

# %%
# check appopriate number of clusters
gk.elbow_method(df_data[['tenure']])

# %% [markdown]
# According to elbow method, for tenure we optimal choice is 3 clusters. We could go with other number if business requires so.

# %%
# calculate tenure cluster for each customer
kmeans = KMeans(n_clusters=3)
kmeans.fit(df_data[['tenure']])
df_data['TenureCluster'] = kmeans.predict(df_data[['tenure']])

# order cluster numbers in ascending order based on mean of group tenure 
df_data = gk.order_cluster('TenureCluster', 'tenure', df_data, True)

# assign labels to tenure cluster
df_data['TenureCluster'] = df_data["TenureCluster"].replace({0:'Low',1:'Mid',2:'High'})

# %%
df_data.groupby('TenureCluster').tenure.describe()

# %%
ax = sns.barplot(x='TenureCluster', y='Churn', data=df_data)
ax.set(title='Tenure Cluster vs. Churn Rate');

# %% [markdown]
# #### Monthly Charges
# This is how it looks after applying the same for Monthly & Total Charges:

# %%
# check appopriate number of clusters
gk.elbow_method(df_data[['MonthlyCharges']])

# %% [markdown]
# According to elbow method, for MonthlyCharges we optimal choice is 3 clusters. We could go with other number if business requires so.

# %%
# calculate tenure cluster for each customer
kmeans = KMeans(n_clusters=3)
kmeans.fit(df_data[['MonthlyCharges']])
df_data['MonthlyChargesCluster'] = kmeans.predict(df_data[['MonthlyCharges']])

# order cluster numbers in ascending order based on mean of group tenure 
df_data = gk.order_cluster('MonthlyChargesCluster', 'tenure', df_data, True)

# assign labels to tenure cluster
df_data['MonthlyChargesCluster'] = df_data["MonthlyChargesCluster"].replace({0:'Low',1:'Mid',2:'High'})

# %%
df_data.groupby('MonthlyChargesCluster').tenure.describe()

# %%
ax = sns.barplot(x='MonthlyChargesCluster', y='Churn', data=df_data)
ax.set(title='Monthly Charges Cluster vs. Churn Rate');

# %% [markdown]
# #### Total Charges
# Total charges after converting to numeric hace few NA values. Those are customers that just signed up and didn't receice their first invoice yet or only received single invoice.

# %%
# coerce to numeric
df_data['TotalCharges'] = pd.to_numeric(df_data['TotalCharges'], errors='coerce')

# %%
# check missing data
df_missing = df_data.loc[df_data['TotalCharges'].isnull(), ['tenure', 'MonthlyCharges','TotalCharges']]
df_missing

# %%
# fill TotalCharges with MonthlyCharges
df_data['TotalCharges'] = df_data['TotalCharges'].fillna(value=df_data['MonthlyCharges'])

# %%
# check 
df_data.loc[df_missing.index, ['tenure', 'MonthlyCharges','TotalCharges']]

# %%
# check appopriate number of clusters
gk.elbow_method(df_data[['TotalCharges']])

# %% [markdown]
# According to elbow method, for MonthlyCharges we optimal choice is 3 clusters. We could go with other number if business requires so.

# %%
# calculate tenure cluster for each customer
kmeans = KMeans(n_clusters=3)
kmeans.fit(df_data[['TotalCharges']])
df_data['TotalChargesCluster'] = kmeans.predict(df_data[['TotalCharges']])

# order cluster numbers in ascending order based on mean of group tenure 
df_data = gk.order_cluster('TotalChargesCluster', 'tenure', df_data, True)

# assign labels to tenure cluster
df_data['TotalChargesCluster'] = df_data["TotalChargesCluster"].replace({0:'Low',1:'Mid',2:'High'})

# %%
df_data.groupby('TotalChargesCluster').tenure.describe()

# %%
ax = sns.barplot(x='TotalChargesCluster', y='Churn', data=df_data)
ax.set(title='Total Charges Charges Cluster vs. Churn Rate');

# %% [markdown]
# ### Categorical Columns
# Before using categorical columns we need to convert them from lables to numbers. Two approaches are availiable:
# * Label Encoder converts categorical columns to numerical by simply assigning integers to distinct values. For instance, the column gender has two values: Female & Male. Label encoder will convert it to 1 and 0.
# * get_dummies() method creates new columns out of categorical ones by assigning 0 & 1s
#
# Let's use both to handle remaining columns.

# %%
df_data.info()

# %%
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ohe = gk.OneHotEncoder(handle_unknown="ignore")  # + drop='first'
dummy_columns = [] # array for multiple value columns

for column in df_data.select_dtypes(include='object').columns:
    if column != 'customerID':
        if df_data[column].nunique() == 2:
            #apply Label Encoder for binary ones
            df_data[column] = le.fit_transform(df_data[column]) 
        else:
            dummy_columns.append(column)

#apply get dummies for selected columns
df_data = pd.get_dummies(data = df_data, columns = dummy_columns)

#apply ohe for selected columns (better for machine learning algorithms because of failure handling on unseen data)
#df_data = pd.concat([df_data.drop(columns=dummy_columns), 
#                     ohe.fit_transform(df_data[dummy_columns])]
#          )

# %% [markdown]
# Check out how the data looks like for the selected columns:

# %%
df_data.info()

# %%
df_data[['gender','Partner','TenureCluster_High','TenureCluster_Low','TenureCluster_Mid']].head()

# %% [markdown]
# As we can see easily, gender & Partner columns became numerical ones, and we have three new columns for TenureCluster.
#
# It is time to fit a logistic regression model and extract insights to make better business decisions.
#
# ## Logistic Regression
# Predicting churn is a binary classification problem. Customers either churn or retain in a given period. Along with being a robust model, Logistic Regression provides interpretable outcomes too. As we did before, let’s sort out our steps to follow for building a Logistic Regression model:
# 1. Prepare the data (inputs for the model)
# 1. Fit the model and see the model summary
#

# %%
# remove unwanted characters from column names
all_columns = []
for column in df_data.columns:
    column = column.replace(" ", "_").replace("(", "_").replace(")", "_").replace("-", "_")
    all_columns.append(column)

df_data.columns = all_columns

# %%
glm_columns = 'gender'

for column in df_data.columns:
    if column not in ['Churn','customerID','gender']:
        glm_columns = glm_columns + ' + ' + column

# %% [markdown]
# And the summary looks like below:

# %%
import statsmodels.api as sm
import statsmodels.formula.api as smf
 

glm_model = smf.glm(formula='Churn ~ {}'.format(glm_columns), data=df_data, family=sm.families.Binomial())
res = glm_model.fit()
print(res.summary())

# %% [markdown]
# We have two important outcomes from this report. When you prepare a Churn Prediction model, you will be faced with the questions below:
# 1. Which characteristics make customers churn or retain?
# 1. What are the most critical ones? What should we focus on?
#
# For the first question, you should look at the 4th column (P>|z|). If the absolute p-value is smaller than 0.05, it means, that feature affects Churn in a statistically significant way. Examples are:
# * SeniorCitizen
# * InternetService_DSL
# * OnlineSecurity_NO
#
# Then the second question. We want to reduce the Churn Rate, where we should start? The scientific version of this question is;
#
# > _Which feature will bring the best ROI if I increase/decrease it by one unit?_
#
# That question can be answered by looking at the coef column. Exponential coef gives us the expected change in Churn Rate if we change it by one unit.  If we apply the code below, we will see the transformed version of all coefficients:

# %%
np.exp(res.params)

# %% [markdown]
# As an example, one unit change in Monthly Charge (coef. 0.965881) means ~3.4% improvement in the odds for churning if we keep everything else constant. From the table above, we can quickly identify which features are more important.
# Now, everything is ready for building our classification model.

# %% [markdown]
# ## Binary Classification Model with XGBoost
# To fit XGBoost to our data, we should prepare features (X) and label(y) sets and do the train & test split.

# %%
# create feature set and labels
X = df_data.drop(['Churn','customerID'],axis=1)
y = df_data.Churn

# %%
# train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=56)

# %%
#building the model
xgb_model = xgb.XGBClassifier(max_depth=5, learning_rate=0.08, objective= 'binary:logistic',n_jobs=-1).fit(X_train, y_train)

print('Accuracy of XGB classifier on training set: {:.2f}'
       .format(xgb_model.score(X_train, y_train)))
print('Accuracy of XGB classifier on test set: {:.2f}'
       .format(xgb_model.score(X_test[X_train.columns], y_test)))

# %% [markdown]
# By using this simple model, we have achieved 83% accuracy,
#
# Our actual Churn Rate in the dataset was 26.5% (reflects as 73.5% for baseline model performance). This shows our model is a useful one. Better to check our classification model to see where exactly our model fails.

# %%
y_pred = xgb_model.predict(X_test)

# %%
print(classification_report(y_test, y_pred))

# %% [markdown]
# We can interpret the report above as if our model tells us, 100 customers will churn, 70 of it will churn (0.70 precision). And actually, there are around 170 customers who will churn (0.58 recall). Especially recall is the main problem here, and we can improve our model’s overall performance by:
# * Adding more data (we have around 2000 rows for this example)
# * Adding more features
# * More feature engineering
# * Trying other models
# * Hyper-parameter tuning
#
# Moving forward, let’s see how our model works in detail. First off, we want to know which features our model exactly used from the dataset. Also, which were the most important ones?
# For addressing this question, we can use the code below:

# %%
from xgboost import plot_importance
fig, ax = plt.subplots(figsize=(10,8))
plot_importance(xgb_model, ax=ax)

# %% [markdown]
# We can see that our model assigned more importance to **TotalCharges** and **MonthlyCharges** compared to others.
#
# Finally, the best way to use this model is assigning Churn Probability for each customer, create segments, and build strategies on top of that. Below we get the churn probability from our model:

# %%
df_data['proba'] = xgb_model.predict_proba(df_data[X_train.columns])[:,1]

# %%
df_data[['customerID', 'proba']].head()

# %% [markdown]
# Now we know if there are likely to churn customers in our best segments and we can build actions based on it!
