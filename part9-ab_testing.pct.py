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
# # Part 9: A/B Testing Design & Execution
# How to conduct A/B tests programmatically
#
#

# %% [markdown]
# As a (Data-Driven) Growth Hacker, one of the main responsibilities is to experiment new ideas and sustain continuous learning. Experimentation is a great way to test your machine learning models, new actions & improve existing ones. Let’s give an example:
#
# You have a churn model that works with 95% accuracy. By calling the customers who are likely to churn and giving an attractive offer, you are assuming 10% of them will retain and bring monthly $20 per each.
# That’s a lot of assumptions. Breaking it down:
#
# * The model’s accuracy is 95%. Is it really? You have trained your model based on last month’s data. The next month, there will be new users, new product features, marketing & brand activities, seasonality and so on. Historical accuracy and real accuracy rarely match in this kind of cases. You can’t come up with a conclusion without a test.
# * By looking at the previous campaigns’ results, you are assuming a 10% conversion. It doesn’t guarantee that your new action will have 10% conversion due to the factors above. Moreover, since it is a new group, their reaction is partly unpredictable.
# * Finally, if those customers bring $20 monthly today, that doesn’t mean they will bring the same after your new action.
#
# To see what’s going to happen, we need to conduct an A/B test. In this notebook, we are going to focus on how we can execute our test programmatically and report the statistics behind it. Just before jumping into coding, there are two important points you need to think while designing and A/B test.
#
# **1. What is your hypothesis?**
#
# Going forward with the example above, our hypothesis is, test group will have more retention:
#
# > Group A → Offer → Higher Retention
#
# > Group B → No offer → Lower Retention
#
# This also helps us to test model accuracy as well. If group B’s retention rate is 50%, it clearly shows that our model is not working. The same applies to measure revenue coming from those users too.
#
# **2. What is your success metric?**
#
# In this case, we are going to check the retention rate of both groups.

# %% [markdown]
# # Programmatic A/B Testing
# For this coding example, we are going to create our own dataset by using numpy library and evaluate the result of an A/B test.
# Let’s start with importing the necessary libraries:

# %%
from datetime import datetime, timedelta, date

from scipy.stats import norm

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

# %% [markdown]
# Now we are going to create our own dataset. The dataset will contain the columns below:
# * customer_id: the unique identifier of the customer
# * segment: customer’s segment; high-value or low-value
# * group: indicates whether the customer is in the test or control group
# * purchase_count: # of purchases completed by the customer
#
# The first three will be quite easy:

# %%
np.array(['high-value' for _ in range(20000)])

# %%
df_hv = pd.DataFrame()
df_hv['customer_id'] = np.arange(20000)
df_hv['segment'] = 'high-value'
df_hv['group'] = 'control'
df_hv.loc[df_hv.index < 10000, 'group'] = 'test'

# %% [markdown]
# Ideally, purchase count should be a Poisson distribution. There will be customers with no purchase and we will have less customers with high purchase counts. Let’s use `numpy.random.poisson()` for doing that and assign different distributions to test and control group:

# %%
df_hv.loc[df_hv.group == 'test', 'purchase_count'] = np.random.poisson(0.6, 10000)
df_hv.loc[df_hv.group == 'control', 'purchase_count'] = np.random.poisson(0.5, 10000)

# %%
df_hv.group.value_counts()

# %%
df_hv.head(10)

# %%
df_hv.tail(10)

# %% [markdown]
# Awesome. We have everything to evaluate our A/B test. Assume we applied an offer to 50% of high-value users and observed their purchases in a given period. Best way to visualize it to check the densities:

# %% [markdown]
# ## Density distribution

# %%
test_results = df_hv[df_hv.group == 'test'].purchase_count
control_results = df_hv[df_hv.group == 'control'].purchase_count

hist_data = [test_results, control_results]
group_labels = ['test', 'control']

# Create distplot with curve_type set to 'normal'
fig = ff.create_distplot(hist_data, group_labels, bin_size=.5,
                         curve_type='normal', show_rug=False)

fig.layout = go.Layout(
    title='High Value Customers Test vs Control',
    plot_bgcolor  = 'rgb(243,243,243)',
    paper_bgcolor  = 'rgb(243,243,243)'
)

fig.show()

# %%
# Same with matplotlib
from scipy.stats import norm

colors = ['r','b']
plt.hist(hist_data, label=group_labels, color=colors, 
         density=True, alpha=0.5);

for data, color in zip(hist_data, colors):
    # Fit a normal distribution to the data:
    mu, std = norm.fit(data)
    
    # Plot the PDF
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, color, linewidth=2)
    
plt.legend()    
plt.show()

# %% [markdown]
# The results are looking really good. The density of the test group’s purchase is better starting from 1. But how we can certainly say this experiment is successful and the difference didn’t happen due to other factors?
#
# To answer this question, we need to check if the uptick in the test group is statistically significant. **scipy** library allows us to programmatically check this:

# %% [markdown]
# ## T-Test
#

# %%
from scipy import stats
ttest = stats.ttest_ind(test_results, control_results)
print(ttest)


# %% [markdown]
# `ttest_ind()` method returns two output:
# * t-statistic: represents the difference between averages of test and control groups in units of standard error. Higher t-statistic value means bigger difference and supports our hypothesis.
# * p-value: measures the probability of the null hypothesis to be true.
#
# What is a **null hypothesis**?
#
# If null hypothesis is true, it means there is no significant difference between your test and control group. So the lower p-value means the better. As the industry standard, we accept that p-value<5% makes the result statistically significant (but it depends on your business logic, there are cases that people use 10% or even 1%).
#
# To understand if our test is statistically significant or not, let’s build a function and apply to our dataset:

# %%
def eval_test(test_results, control_results):
    ttest = stats.ttest_ind(test_results, control_results)
    if ttest[1] < 0.05:
        print('results is significant')
    else:
        print('result is not significant')


# %%
eval_test(test_results, control_results)

# %% [markdown]
# Looks great but unfortunately, it is not that simple. If you select a biased test group, your results will be statistically significant by default. As an example, if we allocate more high-value customer to test group and more low-value customers to control group, then our experiment becomes a failure from the beginning. That’s why selecting the group is the key to a healthy A/B test.

# %% [markdown]
# ## Selecting Test & Control Groups
# The most common approach to select test & control groups is **random sampling**. Let’s see how we can do it programmatically. We are going to start with creating the dataset first. In this version, it will have 20k high-value and 80k low-value customers:

# %%
#create hv segment
df_hv = pd.DataFrame()
df_hv['customer_id'] = np.arange(20000)
df_hv['segment'] = 'high-value'
df_hv['prev_purchase_count'] = np.random.poisson(0.9, 20000)

df_lv = pd.DataFrame()
df_lv['customer_id'] = np.arange(20000, 100000)
df_lv['segment'] = 'low-value'
df_lv['prev_purchase_count'] = np.random.poisson(0.3, 80000)

df_customers = pd.concat([df_hv, df_lv], axis=0).reset_index(drop=True)

# %%
df_customers.head()

# %%
df_customers.tail()

# %%
len(df_customers)

# %% [markdown]
# By using pandas `sample()` function, we can select our test groups. Assuming we will have 90% test and 10% control group:

# %%
df_test = df_customers.sample(frac=0.9)
df_control = df_customers[~df_customers.customer_id.isin(df_test.customer_id)]

# %%
df_test.segment.value_counts()

# %%
df_control.segment.value_counts()

# %% [markdown]
# In this example, we extracted 90% of the whole group and labeled it as test. But there is a small problem that can ruin our experiment. If you have significantly different multiple groups in your dataset (in this case, high-value & low-value), better to do random sampling separately. Otherwise, we can’t guarantee that the ratio of high-value to low-value is the same for test and control group.
#
# To ensure creating test and control groups correctly, we need to apply the following code:

# %%
df_test_hv = df_customers[df_customers.segment == 'high-value'].sample(frac=0.9)
df_test_lv = df_customers[df_customers.segment == 'low-value'].sample(frac=0.9)

df_test = pd.concat([df_test_hv, df_test_lv], axis=0)
df_control = df_customers[~df_customers.customer_id.isin(df_test.customer_id)]

# %% [markdown]
# This makes the allocation correct for both:

# %%
df_test.segment.value_counts()

# %%
df_control.segment.value_counts()

# %% [markdown]
# We have explored how to do the t-test and selecting test and control groups. But what if we are doing A/B/C test or A/B test on multiple groups like above. It’s time to introduce ANOVA tests.

# %% [markdown]
# # One-way ANOVA
# Let’s assume we are testing 2+ variants on same groups (i.e 2 different offers and no-offer to low-value high-value customers). Then we need to apply one-way ANOVA for evaluating our experiment. Let’s start from creating our dataset:

# %%
#create hv segment
df_hv = pd.DataFrame()
df_hv['customer_id'] = np.arange(30000)
df_hv['segment'] = 'high-value'
df_hv['group'] = 'A'
df_hv.loc[df_hv.index>=10000,'group'] = 'B' 
df_hv.loc[df_hv.index>=20000,'group'] = 'C' 

# %%
df_hv.group.value_counts()

# %%
df_hv.loc[df_hv.group == 'A', 'purchase_count'] = np.random.poisson(0.4, 10000)
df_hv.loc[df_hv.group == 'B', 'purchase_count'] = np.random.poisson(0.6, 10000)
df_hv.loc[df_hv.group == 'C', 'purchase_count'] = np.random.poisson(0.2, 10000)

# %%
a_stats = df_hv[df_hv.group=='A'].purchase_count
b_stats = df_hv[df_hv.group=='B'].purchase_count
c_stats = df_hv[df_hv.group=='C'].purchase_count

hist_data = [a_stats, b_stats, c_stats]

group_labels = ['A', 'B','C']

# Create distplot with curve_type set to 'normal'
fig = ff.create_distplot(hist_data, group_labels, bin_size=.5,
                         curve_type='normal',show_rug=False)

fig.layout = go.Layout(
        title='Test vs Control Stats',
        plot_bgcolor  = 'rgb(243,243,243)',
        paper_bgcolor  = 'rgb(243,243,243)',
    )
fig.show()


# %% [markdown]
# To evaluate the result, we will apply the function below:

# %%
def one_anova_test(a_stats, b_stats, c_stats):
    test_result = stats.f_oneway(a_stats, b_stats, c_stats)
    if test_result[1] < 0.05:
        print('result is significant')
    else:
        print('result is not significant')


# %% [markdown]
# The logic is similar to t_test. If p-value is lower than 5%, our test become significant:

# %%
one_anova_test(a_stats,b_stats,c_stats)

# %% [markdown]
# Let’s check out how it will look like if there was no difference between the groups:

# %%
df_hv.loc[df_hv.group == 'A', 'purchase_count'] = np.random.poisson(0.5, 10000)
df_hv.loc[df_hv.group == 'B', 'purchase_count'] = np.random.poisson(0.5, 10000)
df_hv.loc[df_hv.group == 'C', 'purchase_count'] = np.random.poisson(0.5, 10000)

# %%
a_stats = df_hv[df_hv.group=='A'].purchase_count
b_stats = df_hv[df_hv.group=='B'].purchase_count
c_stats = df_hv[df_hv.group=='C'].purchase_count

hist_data = [a_stats, b_stats, c_stats]

group_labels = ['A', 'B','C']

# Create distplot with curve_type set to 'normal'
fig = ff.create_distplot(hist_data, group_labels, bin_size=.5,
                         curve_type='normal',show_rug=False)

fig.layout = go.Layout(
        title='Test vs Control Stats',
        plot_bgcolor  = 'rgb(243,243,243)',
        paper_bgcolor  = 'rgb(243,243,243)',
    )

fig.show()

# %%
one_anova_test(a_stats,b_stats,c_stats)

# %% [markdown]
# If we want to see if there is difference between A and B or C, we can apply the t_test that I explained above.

# %% [markdown]
# # Two-way ANOVA
# Let’s say we are doing the same test on both high-value and low-value customers. In this case, we need to apply two-way ANOVA. We are going to create our dataset again and build our evaluation method:

# %%
#create hv segment
df_hv = pd.DataFrame()
df_hv['customer_id'] = np.arange(20000)
df_hv['segment'] = 'high-value'
df_hv['group'] = 'control'
df_hv.loc[df_hv.index < 10000,'group'] = 'test' 
df_hv.loc[df_hv.group == 'control', 'purchase_count'] = np.random.poisson(0.6, 10000)
df_hv.loc[df_hv.group == 'test', 'purchase_count'] = np.random.poisson(0.8, 10000)


df_lv = pd.DataFrame()
df_lv['customer_id'] = np.arange(20000,100000)
df_lv['segment'] = 'low-value'
df_lv['group'] = 'control'
df_lv.loc[df_lv.index < 40000,'group'] = 'test' 
df_lv.loc[df_lv.group == 'control', 'purchase_count'] = np.random.poisson(0.2, 40000)
df_lv.loc[df_lv.group == 'test', 'purchase_count'] = np.random.poisson(0.3, 40000)

df_customers = pd.concat([df_hv,df_lv],axis=0)

# %%
df_customers.head()

# %%
df_customers.tail()

# %% [markdown]
# Two-way ANOVA requires building a model like below:

# %%
import statsmodels.formula.api as smf 
from statsmodels.stats.anova import anova_lm

model = smf.ols(formula='purchase_count ~ segment + group', data=df_customers).fit()
aov_table = anova_lm(model, typ=2)

# %% [markdown]
# By using **segment & group**, the model trying to reach **purchase_count**. **aov_table** above helps us to see if our experiment is successful:

# %%
print(np.round(aov_table,4))

# %% [markdown]
# The last column represents the result and showing us the difference is significant. 
#
# If it wasn’t, it would look like below:

# %%
#create hv segment
df_hv = pd.DataFrame()
df_hv['customer_id'] = np.array([count for count in range(20000)])
df_hv['segment'] = np.array(['high-value' for _ in range(20000)])
df_hv['group'] = 'control'
df_hv.loc[df_hv.index<10000,'group'] = 'test' 
df_hv.loc[df_hv.group == 'control', 'purchase_count'] = np.random.poisson(0.8, 10000)
df_hv.loc[df_hv.group == 'test', 'purchase_count'] = np.random.poisson(0.8, 10000)


df_lv = pd.DataFrame()
df_lv['customer_id'] = np.array([count for count in range(20000,100000)])
df_lv['segment'] = np.array(['low-value' for _ in range(80000)])
df_lv['group'] = 'control'
df_lv.loc[df_lv.index<40000,'group'] = 'test' 
df_lv.loc[df_lv.group == 'control', 'purchase_count'] = np.random.poisson(0.2, 40000)
df_lv.loc[df_lv.group == 'test', 'purchase_count'] = np.random.poisson(0.2, 40000)

df_customers = pd.concat([df_hv,df_lv],axis=0)


# %%
import statsmodels.formula.api as smf 
from statsmodels.stats.anova import anova_lm
model = smf.ols(formula='purchase_count ~ segment + group ', data=df_customers).fit()
aov_table = anova_lm(model, typ=2)

# %%
print(np.round(aov_table,4))

# %% [markdown]
# This shows, segment (being high-value or low-value) significantly affects the purchase count but group doesn’t since it is almost 70%, way higher than 5%.
#
# Now we know how to select our groups and evaluate the results. But there is one more missing part. To reach statistical significance, our sample size should be enough. Let’s see how we can calculate it.

# %% [markdown]
# # Sample Size Calculation
# To calculate the required sample size, first we need to understand two concepts:
# * __Effect size__: this represents the magnitude of difference between averages of test and control group. It is the variance in averages between test and control groups divided by the standard deviation of the control.
# * __Power__: this refers to the probability of finding a statistical significance in your test. To calculate the sample size, 0.8 is the common value that is being used.
#
# Let’s build our dataset and see the sample size calculation in an example:

# %%
from statsmodels.stats import power
ss_analysis = power.TTestIndPower()

#create hv segment
df_hv = pd.DataFrame()
df_hv['customer_id'] = np.arange(20000)
df_hv['segment'] = 'high-value'
df_hv['prev_purchase_count'] = np.random.poisson(0.7, 20000)

# %%
purchase_mean = df_hv.prev_purchase_count.mean()
purchase_std = df_hv.prev_purchase_count.std()

# %%
print(np.round(purchase_mean,4),np.round(purchase_std,4))

# %% [markdown]
# In this example, the average of purchases (purchase_mean) is 0.7 and the standard deviation (purchase_std) is 0.84.
#
# Let’s say we want to increase the purchase_mean to 0.75 in this experiment. We can calculate the effect size like below:

# %%
effect_size = (0.75 - purchase_mean)/purchase_std

# %%
alpha = 0.05
power = 0.8
ratio = 1

# %%
ss_result = ss_analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha, ratio=ratio , nobs1=None) 
print(ss_result)


# %% [markdown]
# Alpha is the threshold for statistical significance (5%) and our ratio of test and control sample sizes are 1 (equal). As a result, our required sample size is (output of ss_result) **5754**.

# %% [markdown]
# Let’s build a function to use this everywhere we want:

# %%
def calculate_sample_size(c_data, column_name, target, ratio):
    value_mean = c_data[column_name].mean()
    value_std = c_data[column_name].std()
    
    value_target = value_mean * target
    
    effect_size = (value_target - value_mean)/value_std
    
    power = 0.8
    alpha = 0.05
    ss_result = ss_analysis.solve_power(effect_size=effect_size, power=power,alpha=alpha, ratio=ratio , nobs1=None) 
    print(int(ss_result))


# %% [markdown]
# To this function, we need to provide our dataset, the column_name that represents the value (purchase_count in our case), our target mean (0.75 was our target in the previous example) and the ratio.
#
# In the dataset above, let’s assume we want to increase purchase count mean by 5% and we will keep the sizes of both groups the same:

# %%
calculate_sample_size(df_hv, 'prev_purchase_count', 1.05, 1)

# %% [markdown]
# Then the result becomes **9092**.
