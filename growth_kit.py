import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objects as go
import plotly.figure_factory as ff

from sklearn.cluster import KMeans

def get_retail_transactions():
    """
    Load the Online Retail Dataset.
    """
    dateparse = lambda x: pd.datetime.strptime(x, '%m/%d/%Y %H:%M')
    tx_data = pd.read_csv('data/OnlineRetail.csv', parse_dates=['InvoiceDate'],
                          date_parser=dateparse, encoding = 'unicode_escape')
    return tx_data


def get_rfm_clusters(tx_data):
    """
    Calculate RFM clusters and Overall Score for users in the data
    """
    #create tx_user for assigning clustering
    tx_user = pd.DataFrame(tx_data['CustomerID'].unique())
    tx_user.columns = ['CustomerID']
    
    #calculate recency score
    tx_max_purchase = tx_data.groupby('CustomerID').InvoiceDate.max().reset_index()
    tx_max_purchase.columns = ['CustomerID','MaxPurchaseDate']
    tx_max_purchase['Recency'] = (tx_max_purchase['MaxPurchaseDate'].max() - 
                                  tx_max_purchase['MaxPurchaseDate']).dt.days
    tx_user = pd.merge(tx_user, tx_max_purchase[['CustomerID','Recency']], on='CustomerID')
    
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(tx_user[['Recency']])
    tx_user['RecencyCluster'] = kmeans.predict(tx_user[['Recency']])

    tx_user = order_cluster('RecencyCluster', 'Recency', tx_user, False)
    
    #calcuate frequency score
    tx_frequency = tx_data.groupby('CustomerID').InvoiceDate.count().reset_index()
    tx_frequency.columns = ['CustomerID','Frequency']
    tx_user = pd.merge(tx_user, tx_frequency, on='CustomerID')

    kmeans = KMeans(n_clusters=4)
    kmeans.fit(tx_user[['Frequency']])
    tx_user['FrequencyCluster'] = kmeans.predict(tx_user[['Frequency']])

    tx_user = order_cluster('FrequencyCluster', 'Frequency',tx_user,True)
    
    #calcuate revenue score
    tx_data['Revenue'] = tx_data['UnitPrice'] * tx_data['Quantity']
    tx_revenue = tx_data.groupby('CustomerID')['Revenue'].sum().reset_index()
    tx_revenue.columns = ['CustomerID', 'Revenue']
    tx_user = pd.merge(tx_user, tx_revenue, on='CustomerID')
    
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(tx_user[['Revenue']])
    tx_user['RevenueCluster'] = kmeans.predict(tx_user[['Revenue']])
    
    # overall score
    tx_user['OverallScore'] = tx_user.eval("RecencyCluster + FrequencyCluster + RevenueCluster")
    tx_user['Segment'] = 'Low-Value'
    tx_user.loc[tx_user['OverallScore'] > 2, 'Segment'] = 'Mid-Value'
    tx_user.loc[tx_user['OverallScore'] > 4, 'Segment'] = 'High-Value'
    
    return tx_user
                    
def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final




def plot_clusters(x, y, tx_graph):
    plot_data = [
        go.Scatter(
            x=tx_graph.query("Segment == 'Low-Value'")[x],
            y=tx_graph.query("Segment == 'Low-Value'")[y],
            mode='markers',
            name='Low',
            marker= dict(size= 7,
                line= dict(width=1),
                color= 'blue',
                opacity= 0.8
               )
        ),
            go.Scatter(
            x=tx_graph.query("Segment == 'Mid-Value'")[x],
            y=tx_graph.query("Segment == 'Mid-Value'")[y],
            mode='markers',
            name='Mid',
            marker= dict(size= 9,
                line= dict(width=1),
                color= 'green',
                opacity= 0.5
               )
        ),
            go.Scatter(
            x=tx_graph.query("Segment == 'High-Value'")[x],
            y=tx_graph.query("Segment == 'High-Value'")[y],
            mode='markers',
            name='High',
            marker= dict(size= 11,
                line= dict(width=1),
                color= 'red',
                opacity= 0.9
               )
        )
    ]

    plot_layout = go.Layout(
            xaxis= {'title': x},
            yaxis= {'title': y},           
            title='Segments'
        )

    fig = go.Figure(data=plot_data, layout=plot_layout)
    fig.show()