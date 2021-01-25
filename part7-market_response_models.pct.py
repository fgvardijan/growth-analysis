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
# # Part 7: Market Response Models
# Predicting incremental gains of promotional campaigns
#
# By using the models we have built in the previous articles, we can easily segment the customers and predict their lifetime value (LTV) for targeting purposes. As a side note, we also know what will be our sales numbers. But how we can increase our sales? If we do a discount today, how many incremental transactions should we expect?
#
# Segmenting customers and doing A/B tests enable us to try lots of different ideas for generating incremental sales. This is one of the building blocks of Growth Hacking. You need to ideate and experiment continuously to find growth opportunities.
# Splitting the customers who we are going to send the offer into test and control groups helps us to calculate incremental gains.
#
# Let’s see the example below:
#
# <img src="images/7_mrm1.png" title="Market response experiment"/>
#
# In this setup, the target group was divided into three groups to find an answer to the questions below:
#  1. Does giving an offer increase conversion?
#  1. If yes, what kind of offer performs best? Discount or Buy One Get One?
#
# Assuming the results are statistically significant, Discount (Group A) looks the best as it’s increased the conversion by 3% compared to the Control group and brought 1% more conversion against Buy One Get One.
#
# Of course in the real world, it is much more complicated. Some offers perform better on specific segments. So you need to create a portfolio of offers for selected segments. Moreover, you can’t count on conversion as the only criterion of success. There is always a cost trade-off. Generally, while conversion rates go up, cost increases too. That’s why sometimes you need to select an offer that is cost-friendly but brings less conversion.
#
# Now we know which offer performed well compared to others thanks to the experiment. But what about predicting it? If we predict the effect of giving an offer, we can easily maximize our transactions and have a forecast of the cost. Market Response Models help us building this framework. But there is more than one way of doing it. We can group them into two:
#
# 1. If you *don’t have a control group* (imagine you did an open promotion to everyone and announced it on social media), then you cannot calculate the incrementality. For this kind of situation, better to build a regression model that predicts overall sales. The prior assumption will be that the model will provide higher sales numbers for the promo days.
#
# To build this kind of model, your dataset should include promo & non-promo days sales numbers so that the machine learning model can calculate the incrementality.
#
# 2. If you have a control group, you can build the response model based on segment or individual level. For both of them, the assumption is the same. Giving an offer should increase the probability of conversion. 
#
# > The uptick in the individuals’ conversion probability will bring us the incremental conversion.
#

# %%
from datetime import datetime, timedelta, date

# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# %matplotlib inline

#do not show warnings
import warnings
warnings.filterwarnings("ignore")

#import plotly for visualization
import plotly.graph_objects as go
import plotly.figure_factory as ff

#import machine learning related libraries
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import growth_kit as gk

# %%
#import the data
df_data = pd.read_csv('data/response_data.csv')

#print first 10 rows
df_data.head(10)

# %%
df_data.info()

# %%
df_data.conversion.mean()


# %% [markdown]
# Our first 8 columns are providing individual-level data and conversion column is our label to predict:
# * __recency__: months since last purchase
# * __history__: $value of the historical purchases
# * __used_discount/used_bogo__: indicates if the customer used a discount or buy one get one before
# * __zip_code__: class of the zip code as Suburban/Urban/Rural
# * __is_referral__: indicates if the customer was acquired from referral channel
# * __channel__: channels that the customer using, Phone/Web/Multichannel
# * __offer__: the offers sent to the customers, Discount/But One Get One/No Offer

# %% [markdown]
# We will be building a binary classification model for scoring the conversion probability of all customers. For doing that, we are going to follow the steps below:
# * Building the uplift formula
# * Exploratory Data Analysis (EDA) & Feature Engineering
# * Scoring the conversion probabilities
# * Observing the results on the test set
#
# ## Uplift Formula
# First off, we need to build a function that calculates our uplift. To keep it simple, we will assume every conversion means 1 order and the average order value is 25$.
#
# We are going to calculate three types of uplift:
# > __Conversion Uplift__ = Conversion rate of test group - conversion rate of control group
#
# > __Order Uplift__ = Conversion uplift * # customers in test group
#
# > __Revenue Uplift__ = Order Uplift * Average order $ value
#
# Let’s build our calc_uplift function:

# %%
def calc_uplift(df):
    #assigning 25$ to the average order value
    avg_order_value = 25
    
    #calculate conversions for each offer type
    base_conv = df[df.offer == 'No Offer']['conversion'].mean()
    disc_conv = df[df.offer == 'Discount']['conversion'].mean()
    bogo_conv = df[df.offer == 'Buy One Get One']['conversion'].mean()
    
    #calculate conversion uplift for discount and bogo
    disc_conv_uplift = disc_conv - base_conv
    bogo_conv_uplift = bogo_conv - base_conv
    
    #calculate order uplift
    disc_order_uplift = disc_conv_uplift * len(df.loc[(df.offer == 'Discount'), 'conversion'])
    bogo_order_uplift = bogo_conv_uplift * len(df.loc[(df.offer == 'Buy One Get One'), 'conversion'])
    
    #calculate revenue uplift
    disc_rev_uplift = disc_order_uplift * avg_order_value
    bogo_rev_uplift = bogo_order_uplift * avg_order_value
        
    print('Discount Conversion Uplift: {0}%'.format(np.round(disc_conv_uplift*100,2)))
    print('Discount Order Uplift: {0}'.format(np.round(disc_order_uplift,2)))
    print('Discount Revenue Uplift: ${0}\n'.format(np.round(disc_rev_uplift,2)))
          
    print('-------------- \n')

    print('BOGO Conversion Uplift: {0}%'.format(np.round(bogo_conv_uplift*100,2)))
    print('BOGO Order Uplift: {0}'.format(np.round(bogo_order_uplift,2)))
    print('BOGO Revenue Uplift: ${0}'.format(np.round(bogo_rev_uplift,2)))  


# %%
calc_uplift(df_data)

# %% [markdown]
# Discount looks like a better option if we want to get more conversion. It brings 7.6% uptick compared to the customers who didn’t receive any offer. BOGO (Buy One Get One) has 4.5% uptick as well.
#
# Let’s start exploring which factors are the drivers of this incremental change.

# %% [markdown]
# ## EDA & Feature Engineering
# We check every feature one by one to find out their impact on conversion

# %% [markdown]
# ### Recency
# Ideally, the conversion should go down while recency goes up since inactive customers are less likely to buy again:

# %%
df_plot = df_data.groupby('recency').conversion.mean().reset_index()
plot_data = [
    go.Bar(
        x=df_plot['recency'],
        y=df_plot['conversion'],
    )
]
plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Recency vs Conversion',
        plot_bgcolor  = 'rgb(243,243,243)',
        paper_bgcolor  = 'rgb(243,243,243)',
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()

# %% [markdown]
# It goes as expected until 11 months of recency. Then it increases. It can be due to many reasons like having less number of customers in those buckets or the effect of the given offers.

# %%
df_data.recency.value_counts().sort_index().plot.bar()

# %% [markdown]
# ### History
# We will create a history cluster and observe its impact. Let’s apply k-means clustering to define the significant groups in history:

# %%
kmeans = KMeans(n_clusters=5)
kmeans.fit(df_data[['history']])
df_data['history_cluster'] = kmeans.predict(df_data[['history']])

# %%
#order the cluster numbers 
df_data = gk.order_cluster('history_cluster', 'history', df_data, ascending=True)

# %% [markdown]
# Overview of the clusters and the plot vs. conversion:

# %%
#print how the clusters look like
df_data.groupby('history_cluster').agg({'history':['mean','min','max'], 'conversion':['count', 'mean']})

# %%
df_plot = df_data.groupby('history_cluster').conversion.mean().reset_index()
plot_data = [
    go.Bar(
        x=df_plot['history_cluster'],
        y=df_plot['conversion'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='History vs Conversion',
        plot_bgcolor  = 'rgb(243,243,243)',
        paper_bgcolor  = 'rgb(243,243,243)',
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()

# %% [markdown]
# Customers with higher $ value of history are more likely to convert.

# %% [markdown]
# ### Used Discount & BOGO
# We will check these two features together with the following code line:

# %%
df_data.groupby(['used_discount', 'used_bogo', 'offer']).agg({'conversion': np.mean})

# %% [markdown]
# Customers, who used both of the offers before, have the highest conversion rate.

# %% [markdown]
# ### Zip Code
# Rural shows better conversion compared to others:

# %%
df_plot = df_data.groupby('zip_code').conversion.mean().reset_index()
plot_data = [
    go.Bar(
        x=df_plot['zip_code'],
        y=df_plot['conversion'],
        marker=dict(
        color=['green', 'blue', 'orange'])
    )
]
plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Zip Code vs Conversion',
        plot_bgcolor  = 'rgb(243,243,243)',
        paper_bgcolor  = 'rgb(243,243,243)',
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()

# %% [markdown]
# ### Referral
# As we see below, customers from referral channel have less conversion rate:

# %%
df_plot = df_data.groupby('is_referral').conversion.mean().reset_index()

# %%
df_plot = df_data.groupby('is_referral').conversion.mean().reset_index()
plot_data = [
    go.Bar(
        x=df_plot['is_referral'],
        y=df_plot['conversion'],
        marker=dict(
        color=['green', 'blue'])
    )
]
plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Referral vs Conversion',
        plot_bgcolor  = 'rgb(243,243,243)',
        paper_bgcolor  = 'rgb(243,243,243)',
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()

# %% [markdown]
# They show almost 5% less conversion.

# %% [markdown]
# ### Channel

# %%
df_plot = df_data.groupby('channel').conversion.mean().reset_index()
plot_data = [
    go.Bar(
        x=df_plot['channel'],
        y=df_plot['conversion'],
        marker=dict(
        color=['green', 'blue', 'orange'])
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Channel vs Conversion',
        plot_bgcolor  = 'rgb(243,243,243)',
        paper_bgcolor  = 'rgb(243,243,243)',
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()

# %% [markdown]
# Multichannel shows higher conversion rate as we expected. Using more than one channel is an indicator of high engagement.

# %% [markdown]
# ### Offer Type

# %%
df_plot = df_data.groupby('offer').conversion.mean().reset_index()
plot_data = [
    go.Bar(
        x=df_plot['offer'],
        y=df_plot['conversion'],
        marker=dict(
        color=['green', 'blue', 'orange'])
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Offer vs Conversion',
        plot_bgcolor  = 'rgb(243,243,243)',
        paper_bgcolor  = 'rgb(243,243,243)',
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()

# %% [markdown]
# Customers who get discount offers show ~18% conversion whereas it is ~15% for BOGO. If customers don’t get an offer, their conversion rate drops by ~4%.

# %% [markdown]
# Feature Engineering of this data will be pretty simple. We will apply .get_dummies() to convert categorical columns to numerical ones:

# %%
df_model = df_data.copy()
df_model = pd.get_dummies(df_model)

# %%
df_model.head()

# %%
df_model.conversion.mean()

# %% [markdown]
# It is time to build our machine learning model to score conversion probabilities.
#
# ## Scoring conversion probabilities
# To build our model, we need to follow the steps we mentioned earlier in the articles.
# Let’s start with splitting features and the label:

# %%
#create feature set and labels
X = df_model.drop(['conversion'],axis=1)
y = df_model.conversion

# %%
X.columns

# %%
# Creating training and test sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56)

# %% [markdown]
# We will fit the model and get the conversion probabilities. `predit_proba()` function of our model assigns probability for each row:

# %%
xgb_model = xgb.XGBClassifier().fit(X_train, y_train)
X_test['proba'] = xgb_model.predict_proba(X_test)[:,1]

# %% [markdown]
# Let’s see how our probability column looks like:

# %%
X_test.head()

# %% [markdown]
# From the above, we can see that our model assigned the probability of conversion (from 0 to 1) for each customer.

# %%
X_test.proba.mean()

# %%
y_test.mean()

# %%
X_test['conversion'] = y_test

# %% [markdown]
# Let's compare actual vs. predicted conversion means per group:

# %%
X_test.loc[X_test['offer_Buy One Get One'] == 1, ['conversion', 'proba']].mean()

# %%
X_test.loc[X_test['offer_Discount'] == 1, ['conversion', 'proba']].mean()

# %%
X_test.loc[X_test['offer_No Offer'] == 1, ['conversion', 'proba']].mean()


# %% [markdown]
# Finally, we need to understand if our model works well or not.
#
# ## Results on test set
# Now we assume, the difference in the probability of discount, bogo and control group should be similar to conversion differences between them.
#
# We need to use our test set to find it out.
#
# Let’s calculate predicted and real order upticks for discount and bogo:

# %%
def calc_uptick(df, target='conversion'):
    #assigning 25$ to the average order value
    avg_order_value = 25
    
    #calculate conversions for each offer type
    base_conv = df[df['offer_No Offer'] == 1][target].mean()
    disc_conv = df[df['offer_Discount'] == 1][target].mean()
    bogo_conv = df[df['offer_Buy One Get One'] == 1][target].mean()
    
    
    #calculate conversion uplift for discount and bogo
    disc_conv_uplift = disc_conv - base_conv
    bogo_conv_uplift = bogo_conv - base_conv
    
    #calculate order uplift
    disc_order_uplift = disc_conv_uplift * len(df.loc[df['offer_Discount'] == 1, target])
    bogo_order_uplift = bogo_conv_uplift * len(df.loc[df['offer_Buy One Get One'] == 1, target])
    
    #calculate revenue uplift
    disc_rev_uplift = disc_order_uplift * avg_order_value
    bogo_rev_uplift = bogo_order_uplift * avg_order_value
        
    print('Discount Conversion Uplift: {0}%'.format(np.round(disc_conv_uplift*100,2)))
    print('Discount Order Uplift: {0}'.format(np.round(disc_order_uplift,2)))
    print('Discount Revenue Uplift: ${0}\n'.format(np.round(disc_rev_uplift,2)))
          
    print('-------------- \n')

    print('BOGO Conversion Uplift: {0}%'.format(np.round(bogo_conv_uplift*100,2)))
    print('BOGO Order Uplift: {0}'.format(np.round(bogo_order_uplift,2)))
    print('BOGO Revenue Uplift: ${0}'.format(np.round(bogo_rev_uplift,2)))  

# %%
print("Real uplift:")
calc_uptick(X_test, 'conversion')

# %%
print("Predicted uplift:")
calc_uptick(X_test, 'proba')

# %% [markdown]
# Below is original calculation that differs in multiplication with len(X_test) instead of with len(X_test[group == [Discount|Bogo]])

# %%
real_disc_uptick = int(len(X_test)*(X_test[X_test['offer_Discount'] == 1].conversion.mean() - X_test[X_test['offer_No Offer'] == 1].conversion.mean()))

pred_disc_uptick = int(len(X_test)*(X_test[X_test['offer_Discount'] == 1].proba.mean() - X_test[X_test['offer_No Offer'] == 1].proba.mean()))


# %% [markdown]
# For real uptick calculation, we used the conversion column. For the predicted one, we replaced it with proba.

# %%
print('Real Discount Uptick - Order: {}, Revenue: {}'.format(real_disc_uptick, real_disc_uptick*25))
print('Predicted Discount Uptick - Order: {}, Revenue: {}'.format(pred_disc_uptick, pred_disc_uptick*25))

# %% [markdown]
# The results are pretty good. The real order uptick was 1106 and the model predicted it as 888 (x.x% error).
#
# Revenue uptick prediction comparison: 27650 vs 22200.
#
# We need to check if the results are good for BOGO as well:

# %%
real_bogo_uptick = len(X_test)*(X_test[X_test['offer_Buy One Get One'] == 1].conversion.mean() - X_test[X_test['offer_No Offer'] == 1].conversion.mean())

pred_bogo_uptick = len(X_test)*(X_test[X_test['offer_Buy One Get One'] == 1].proba.mean() - X_test[X_test['offer_No Offer'] == 1].proba.mean())

# %%
print('Real BOGO Uptick - Order: {}, Revenue: {}'.format(real_bogo_uptick, real_bogo_uptick*25))
print('Predicted BOGO Uptick - Order: {}, Revenue: {}'.format(pred_bogo_uptick, pred_bogo_uptick*25))

# %% [markdown]
# Promising results for BOGO:
# Order uptick - real vs predicted: 604 vs 564
# Revenue uptick — real vs predicted: 15103 vs 14119
#
# The error rate is around 5.6%. The model can benefit from improving the prediction scores on BOGO offer type.
#
# Calculating conversion probabilities help us a lot in different areas as well. We have predicted the return of the different types of offers but it can help us to find out who to target for maximizing the uplift as well. In the next article, we will build our own uplift model.
