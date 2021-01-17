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
# # Part 1: Know Your Metrics
#
# In terms of growth we could want more customers, more orders, more revenue, more signups, more efficiency... 
#
# Before going into coding, we need to understand what exactly is our metric.
#
# > _The North Star Metric is the single metric that best captures the core value that your product delivers to customers._
#
# This metric depends on company’s product, position, targets & more. Airbnb’s North Star Metric is nights booked whereas for Facebook, it is daily active users.
#
# We will use a [sample dataset of an online retail](https://www.kaggle.com/vijayuv/onlineretail). For an online retail, we can choose our North Star Metric as __Monthly Revenue__. In addition to Monthly Revenue we will also calculate following metrics:
# * __Monthly Active Customer__
# * __Monthly Order Count__
# * __Average Revenue per Order__
# * __New Customer Ratio__
# * __Monthly Retention Rate__
# * __Cohort Based Retention Rate__
#
#
# ## Load the dataset
# Let's first load the dataset. This is how our data looks like.

# %%
# import libraries
from datetime import datetime, timedelta
import pandas as pd
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import plotly.graph_objects as go
import plotly.figure_factory as ff

# %%
dateparse = lambda x: datetime.strptime(x, '%m/%d/%Y %H:%M')
tx_data = pd.read_csv('data/OnlineRetail.csv', parse_dates=['InvoiceDate'], date_parser=dateparse, encoding = 'unicode_escape')

# %%
print(tx_data.info())
tx_data.head(10)

# %%
tx_data.shape

# %% [markdown]
# We have all the crucial information we need:
# * Customer ID
# * Unit Price
# * Quantity
# * Invoice Date
#
# With all these features, we can build our North Star Metric equation:
# > _Revenue = Active Customer Count * Order Count * Average Revenue per Order_

# %% [markdown]
# ## Revenue
# We want to see monthly revenue. So let's calculate revenue of each order first.

# %%
# create YearMonth field for the ease of reporting and visualization
tx_data['InvoiceYearMonth'] = tx_data['InvoiceDate'].map(lambda date: 100*date.year + date.month)

# calculate Revenue for each row
tx_data['Revenue'] = tx_data['UnitPrice'] * tx_data['Quantity']

tx_data.head()

# %% [markdown]
# Summing up for every month and we have montly revenue.

# %%
# and create a new dataframe with YearMonth - Revenue columns
tx_revenue = tx_data.groupby(['InvoiceYearMonth'])['Revenue'].sum().reset_index()
tx_revenue

# %% [markdown]
# We can also visualize **Monthly Revenue**.

# %%
#X and Y axis inputs for Plotly graph. We use Scatter for line graphs
plot_data = [
    go.Scatter(
        x=tx_revenue['InvoiceYearMonth'],
        y=tx_revenue['Revenue'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Montly Revenue'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()

# %% [markdown]
# This clearly shows our revenue is growing especially Aug ‘11 onwards (and our data in December is incomplete). Absolute numbers are fine, let’s figure out what is our __Monthly Revenue Growth Rate__:

# %%
# using pct_change() function to see monthly percentage change
tx_revenue['MonthlyGrowth'] = tx_revenue['Revenue'].pct_change()
tx_revenue.head()

# %%
#visualization - line graph
plot_data = [
    go.Scatter(
        x=tx_revenue.query("InvoiceYearMonth < 201112")['InvoiceYearMonth'],
        y=tx_revenue.query("InvoiceYearMonth < 201112")['MonthlyGrowth'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Montly Growth Rate'
    )

fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()

# %% [markdown]
# Everything looks good, we saw 36.5% growth previous month (December is excluded in the code since it hasn’t been completed yet). But we need to identify what exactly happened on April. Was it due to less active customers or our customers did less orders? Maybe they just started to buy cheaper products? We can’t say anything without doing a deep-dive analysis.

# %% [markdown]
# ## Monthly Active Customers
# To see the details Monthly Active Customers, we will follow the steps we exactly did for Monthly Revenue. Starting from this part, we will be focusing on UK data only (which has the most records). We can get the monthly active customers by counting unique `CustomerID`s.

# %%
tx_data.groupby('Country')['Revenue'].sum().sort_values(ascending=False).astype(int).head()

# %%
tx_uk = tx_data.query("Country=='United Kingdom'").reset_index(drop=True)
tx_uk.head()

# %% [markdown]
# Number of active customers per month and its bar plot:

# %%
# create monthly active customers dataframe by counting unique Customer IDs
tx_monthly_active = tx_uk.groupby('InvoiceYearMonth')['CustomerID'].nunique().reset_index()
tx_monthly_active['pct_change'] = tx_monthly_active['CustomerID'].pct_change()
tx_monthly_active

# %%
#plotting the output
plot_data = [
    go.Bar(
        x=tx_monthly_active['InvoiceYearMonth'],
        y=tx_monthly_active['CustomerID'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Monthly Active Customers'
    )

fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()

# %% [markdown]
# In April, Monthly Active Customer number dropped to 817 from 923 (-11.5%).

# %%
#visualization - line graph
plot_data = [
    go.Scatter(
        x=tx_monthly_active.query("InvoiceYearMonth < 201112")['InvoiceYearMonth'],
        y=tx_monthly_active.query("InvoiceYearMonth < 201112")['pct_change'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Montly Active Customers (pct_change)'
    )

fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()

# %% [markdown]
# We will see the same trend for number of orders as well.

# %% [markdown]
# ## Monthly Order Count
# We will apply the same steps this time using Quantity field:

# %%
#create a new dataframe for no. of order by using quantity field
tx_monthly_sales = tx_uk.groupby('InvoiceYearMonth')['Quantity'].sum().reset_index()
tx_monthly_sales['pct_change'] = tx_monthly_sales['Quantity'].pct_change()

#print the dataframe
tx_monthly_sales

# %%
#plot
plot_data = [
    go.Bar(
        x=tx_monthly_sales['InvoiceYearMonth'],
        y=tx_monthly_sales['Quantity'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Monthly Total # of Order'
    )

fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()

# %%
#plot
plot_data = [
    go.Bar(
        x=tx_monthly_sales.query('InvoiceYearMonth < 201112')['InvoiceYearMonth'],
        y=tx_monthly_sales.query('InvoiceYearMonth < 201112')['pct_change'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Monthly Total # of Order (pct_change)'
    )

fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()

# %% [markdown]
# As we expected, Order Count is also declined in April (279k to 257k, -8%)
# We know that Active Customer Count directly affected Order Count decrease. At the end, we should definitely check our __Average Revenue per Order__ as well.

# %% [markdown]
# ## Average Revenue per Order
# To get this data, we need to calculate the average of revenue for each month:

# %%
# create a new dataframe for average revenue by taking the mean of it
tx_monthly_order_avg = tx_uk.groupby('InvoiceYearMonth')['Revenue'].mean().reset_index()
tx_monthly_order_avg

# %%
#plot the bar chart
plot_data = [
    go.Bar(
        x=tx_monthly_order_avg['InvoiceYearMonth'],
        y=tx_monthly_order_avg['Revenue'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Monthly Order Average'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()

# %% [markdown]
# Even the monthly order average dropped for April (16.7 to 15.8). We observed slow-down in every metric affecting our North Star.
#
# We have looked at our major metrics. Of course there are many more and it varies across industries. Let’s continue investigating some other important metrics:
# * __New Customer Ratio__: a good indicator of if we are losing our existing customers or unable to attract new ones
# * __Retention Rate__: King of the metrics. Indicates how many customers we retain over specific time window. We will be showing examples for monthly retention rate and cohort based retention rate.

# %% [markdown]
# ## New Customer Ratio
# First we should define what is a new customer. In our dataset, we can assume a new customer is whoever did his/her first purchase in the time window we defined. We will do it monthly for this example.
#
# We will be using `.min()` function to find our first purchase date for each customer and define new customers based on that. The code below will apply this function and show us the revenue breakdown for each group monthly.

# %%
#create a dataframe contaning CustomerID and first purchase date
tx_min_purchase = tx_uk.groupby('CustomerID')['InvoiceDate'].min().reset_index()
tx_min_purchase.columns = ['CustomerID','MinPurchaseDate']
tx_min_purchase['MinPurchaseYearMonth'] = tx_min_purchase['MinPurchaseDate'].map(lambda date: 100*date.year + date.month)
tx_min_purchase.head()

# %%
#merge first purchase date column to our main dataframe (tx_uk)
tx_uk = pd.merge(tx_uk, tx_min_purchase, on='CustomerID')

# %%
#create a column called User Type and assign Existing if User's First Purchase Year Month 
#before the selected Invoice Year Month
tx_uk['UserType'] = 'New'
tx_uk.loc[tx_uk['InvoiceYearMonth'] > tx_uk['MinPurchaseYearMonth'], 'UserType'] = 'Existing'

# %%
tx_uk.UserType.value_counts()

# %%
tx_uk.head()

# %% [markdown]
# Lets calculate the Revenue per month for each user type.

# %%
#calculate the Revenue per month for each user type
tx_user_type_revenue = tx_uk.groupby(['InvoiceYearMonth','UserType'])['Revenue'].sum().reset_index()
tx_user_type_revenue

# %%
# filter the dates and plot the result
tx_user_type_revenue = tx_user_type_revenue.query("InvoiceYearMonth != 201012 and InvoiceYearMonth != 201112")

plot_data = [
    go.Scatter(
        x=tx_user_type_revenue.query("UserType == 'Existing'")['InvoiceYearMonth'],
        y=tx_user_type_revenue.query("UserType == 'Existing'")['Revenue'],
        name = 'Existing'
    ),
    go.Scatter(
        x=tx_user_type_revenue.query("UserType == 'New'")['InvoiceYearMonth'],
        y=tx_user_type_revenue.query("UserType == 'New'")['Revenue'],
        name = 'New'
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='New vs Existing Users'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()


# %% [markdown]
# Existing customers are showing a positive trend and tell us that our customer base is growing but new customers have a slight negative trend.
#
# Let’s have a better view by looking at the New Customer Ratio:

# %% tags=["hide_input", "hide_output"]
tx_uk.query("UserType == 'New'").groupby(['InvoiceYearMonth'])['CustomerID'].nunique()

# %% tags=["hide_input", "hide_output"]
tx_uk.query("UserType == 'Existing'").groupby(['InvoiceYearMonth'])['CustomerID'].nunique()

# %%
#create a dataframe that shows new user ratio - we also need to drop NA values (first month new user ratio is 0)
tx_user_ratio = (tx_uk.query("UserType == 'New'").groupby(['InvoiceYearMonth'])['CustomerID'].nunique() 
                 / tx_uk.query("UserType == 'Existing'").groupby(['InvoiceYearMonth'])['CustomerID'].nunique() )
tx_user_ratio = tx_user_ratio.reset_index()
tx_user_ratio = tx_user_ratio.dropna()

tx_user_ratio

# %%
#plot the result
plot_data = [
    go.Bar(
        x=tx_user_ratio.query("InvoiceYearMonth>201101 and InvoiceYearMonth<201112")['InvoiceYearMonth'],
        y=tx_user_ratio.query("InvoiceYearMonth>201101 and InvoiceYearMonth<201112")['CustomerID'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='New Customer Ratio'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()

# %% [markdown]
# New Customer Ratio has declined as expected (we assumed on Feb, all customers were New) and running around 20%.

# %% [markdown]
# ## Generate Sign Up Data
# We want to inspect channel the customers came from when they made first purchase.
#
# Imagine we have data on the date cusotmer SignedUp and date when customer installed the mobile app. We don't so we'll simulate this data. This is what fucntion `generate_signup_data` does.
#
#

# %%
tx_min_purchase.head()

# %%
unq_month_year =  tx_min_purchase.MinPurchaseYearMonth.unique()
unq_month_year


# %%
def generate_signup_date(year_month):
    signup_date = [el for el in unq_month_year if year_month >= el]
    return np.random.choice(signup_date)


# %%
tx_min_purchase['SignupYearMonth'] = tx_min_purchase.apply(lambda row: generate_signup_date(row['MinPurchaseYearMonth']), axis=1)
tx_min_purchase['InstallYearMonth'] = tx_min_purchase.apply(lambda row: generate_signup_date(row['SignupYearMonth']), axis=1)

# %%
tx_min_purchase.head()

# %% [markdown]
# Simulate the channel from which the customer came.

# %%
channels = ['organic','inorganic','referral']
tx_min_purchase['AcqChannel'] = tx_min_purchase.apply(lambda x: np.random.choice(channels),axis=1)

# %% [markdown]
# ## Activation Rate
# Activation rate should be monitored very closely because it indicates how effective is your campaign. 
#
# Activation rate indicates the percentage of people who upon signing-up your campaign actually engage with by making a purchase.
#
# For making Monthly Activation Rate visualized, we need to calculate how many customers were made purchase in the month they signed up.
#
# > _**Monthly Activation Rate** = Signed Up and Engaged Customers / Total Signed Up Customers_

# %%
tx_activation = (tx_min_purchase[tx_min_purchase['MinPurchaseYearMonth'] == tx_min_purchase['SignupYearMonth']].groupby('SignupYearMonth').CustomerID.count() 
                 / tx_min_purchase.groupby('SignupYearMonth').CustomerID.count())
tx_activation = tx_activation.reset_index()
tx_activation

# %%
plot_data = [
    go.Bar(
        x=tx_activation.query("SignupYearMonth>201101 and SignupYearMonth<201109")['SignupYearMonth'],
        y=tx_activation.query("SignupYearMonth>201101 and SignupYearMonth<201109")['CustomerID'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Monthly Activation Rate'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()

# %% [markdown]
# Let's also check activation rates by acquisition channel. We do it by counting all customers that purchased and signed-up in the selected month vs. total signed-up customers and grouping by month and channel.

# %%
tx_activation_ch = (tx_min_purchase[tx_min_purchase['MinPurchaseYearMonth'] == tx_min_purchase['SignupYearMonth']].groupby(['SignupYearMonth','AcqChannel']).CustomerID.count()
                    / tx_min_purchase.groupby(['SignupYearMonth','AcqChannel']).CustomerID.count())
tx_activation_ch = tx_activation_ch.reset_index()
tx_activation_ch.head(6)

# %%
# filter the data
tx_activation_ch = tx_activation_ch.query("SignupYearMonth>201101 and SignupYearMonth<201108")

# %%
plot_data = [
    go.Scatter(
        x=tx_activation_ch.query("AcqChannel == 'organic'")['SignupYearMonth'],
        y=tx_activation_ch.query("AcqChannel == 'organic'")['CustomerID'],
        name="organic"
    ),
    go.Scatter(
        x=tx_activation_ch.query("AcqChannel == 'inorganic'")['SignupYearMonth'],
        y=tx_activation_ch.query("AcqChannel == 'inorganic'")['CustomerID'],
        name="inorganic"
    ),
    go.Scatter(
        x=tx_activation_ch.query("AcqChannel == 'referral'")['SignupYearMonth'],
        y=tx_activation_ch.query("AcqChannel == 'referral'")['CustomerID'],
        name="referral"
    )
    
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Monthly Activation Rate - Channel Based'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()

# %% [markdown]
# Since it is synteticaly generated data we won't take it into consideration for the analysis.

# %% [markdown]
# ## Monthly Retention Rate
# Retention rate should be monitored very closely because it indicates how sticky is your service and how well your product fits the market. For making Monthly Retention Rate visualized, we need to calculate how many customers were retained from previous month.
#
# > _**Monthly Retention Rate** = Retained Customers From Prev. Month / Active Customers Total_
#
# We will be using `crosstab()` function of pandas which to calculate Retention Rate.

# %%
#identify which users are active by looking at their revenue per month
tx_user_purchase = tx_uk.groupby(['CustomerID','InvoiceYearMonth'])['Revenue'].sum().reset_index()

#create retention matrix with crosstab
tx_retention = pd.crosstab(index=tx_user_purchase['CustomerID'], columns=tx_user_purchase['InvoiceYearMonth']).reset_index()
tx_retention.head()

# %% [markdown]
# Retention table shows us which customers are active on each month (1 stands for active).
#
# In the for loop, for each month we calculate Retained Customer Count from previous month and Total Customer Count.

# %%
#create an array of dictionaries which keeps Retained & Total User count for each month
months = tx_retention.columns[2:]
retention_array = []
for i in range(len(months)-1):
    retention_data = {}
    selected_month = months[i+1]
    prev_month = months[i]
    retention_data['InvoiceYearMonth'] = int(selected_month)
    retention_data['TotalUserCount'] = tx_retention[selected_month].sum()
    retention_data['RetainedUserCount'] = tx_retention[(tx_retention[selected_month]>0) & (tx_retention[prev_month]>0)][selected_month].sum()
    retention_array.append(retention_data)
    
#convert the array to dataframe and calculate Retention Rate
tx_retention = pd.DataFrame(retention_array)
tx_retention['RetentionRate'] = tx_retention['RetainedUserCount']/tx_retention['TotalUserCount']

# %%
tx_retention

# %%
#plot the retention rate graph
plot_data = [
    go.Scatter(
        x=tx_retention.query("InvoiceYearMonth<201112")['InvoiceYearMonth'],
        y=tx_retention.query("InvoiceYearMonth<201112")['RetentionRate'],
        name="organic"
    )
    
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Monthly Retention Rate'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()

# %% [markdown]
# Monthly Retention Rate significantly jumped from June to August and went back to previous levels afterwards.

# %% [markdown]
# ## Churn Rate
# Chrun rate is opposite of retention rate. It is the percentage of customer that were active on the previous month but did not buy anyting in the current month.

# %%
tx_retention['ChurnRate'] =  1 - tx_retention['RetentionRate']

# %%
plot_data = [
    go.Scatter(
        x=tx_retention.query("InvoiceYearMonth<201112")['InvoiceYearMonth'],
        y=tx_retention.query("InvoiceYearMonth<201112")['ChurnRate'],
        name="organic"
    )
    
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Monthly Churn Rate'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()

# %% [markdown]
# ## Cohort Based Retention Rate
# There is another way of measuring Retention Rate which allows you to see Retention Rate for each cohort. Cohorts are determined as first purchase year-month of the customers. We will be measuring what percentage of the customers retained after their first purchase in each month. 
#
# This view will help us to see how recent and old cohorts differ regarding retention rate and if recent changes in customer experience affected new customer’s retention or not.

# %%
#create our retention table again with crosstab() - we need to change the column names for using them in .query() function
tx_retention = pd.crosstab(tx_user_purchase['CustomerID'], tx_user_purchase['InvoiceYearMonth']).reset_index()
new_column_names = [ 'm_' + str(column) for column in tx_retention.columns]
tx_retention.columns = new_column_names

#create the array of Retained users for each cohort monthly
retention_array = []
for i in range(len(months)):
    retention_data = {}
    selected_month = months[i]
    prev_months = months[:i]
    next_months = months[i+1:]
    for prev_month in prev_months:
        retention_data[prev_month] = np.nan
        
    total_user_count =  retention_data['TotalUserCount'] = tx_retention['m_' + str(selected_month)].sum()
    retention_data[selected_month] = 1 
    
    query = "{} > 0".format('m_' + str(selected_month))
    

    for next_month in next_months:
        query = query + " and {} > 0".format(str('m_' + str(next_month)))
        retention_data[next_month] = np.round(tx_retention.query(query)['m_' + str(next_month)].sum()/total_user_count,2)
    retention_array.append(retention_data)
    
tx_retention = pd.DataFrame(retention_array)
tx_retention.index = months

# %% [markdown]
# Tx_retention has this amazing view of cohort based retention rate:

# %%
#showing new cohort based retention table
tx_retention

# %% [markdown]
# We can see that first month retention rate became better recently (don’t take Dec ’11 into account) and in almost 1 year, only 7% of our customers retain with us.

# %%
# first month retention rate
first_month_retention = tx_retention[:-1].apply(lambda x: x[x.name + 1], axis=1)
first_month_retention.name = 'FirstMonthRetention'
first_month_retention

# %% [markdown]
# Automate for first 4 months

# %%
# retention rate evolution - [1st, 2nd, 3rd] month retention rate over time
retention_data = {}
for i in np.arange(1,5):  
    retention_data[str(i) +'_month_retention'] = tx_retention[:-i].apply(lambda x: x[x.name + i], axis=1)

# %%
month_retention = pd.DataFrame(retention_data)
month_retention

# %%
plot_data = [
    go.Scatter(
        x=month_retention.index,
        y=month_retention['1_month_retention'],
        name="1st month"
    ),
    go.Scatter(
        x=month_retention.index,
        y=month_retention['2_month_retention'],
        name="2nd month"
    )
    
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Retention Rate after X Months'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()

# %% [markdown]
# # Summary 
# In this notebook defined important metrics for the company and calculated / analysed them with Python.
# * Monthly Revenue
# * Monthly Active Customer
# * Monthly Order Count
# * Average Revenue per Order
# * New Customer Ratio
# * Monthly Retention Rate
# * Cohort Based Retention Rate
#
# In next part we'll try to segment our base to see who are our best customers.
