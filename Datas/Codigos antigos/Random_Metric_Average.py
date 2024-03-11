import random
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from igraph import Graph
from scipy.stats import ttest_ind
#Gerar tabela para verios delays
def calculate_random_metric_averages(data_df):
    normalized_values = {}
    accumulated_sum_degree = 0
    accumulated_sum_clustering = 0
    accumulated_sum_weight = 0
    accumulated_sum_betweenness = 0
    accumulated_sum_closeness_w = 0
    accumulated_sum_eignv_w = 0
    dates = data_df['date'].unique()
    normalized_values['Date'] = dates.tolist()
    normalized_values['Degree'] = []
    normalized_values['Clustering'] = []
    normalized_values['Weight'] = []
    normalized_values['Betweenness'] = []
    normalized_values['Closeness_w'] = []
    normalized_values['Eignv_w'] = []
    geocodes = list(data_df['geocode'])
    soma_degree = data_df['degree'].sum()
    soma_weight = data_df['Weight'].sum()
    soma_clustering = data_df['clustering'].sum()
    soma_betweenness = data_df['betweenness'].sum()
    soma_closeness_w = data_df['closeness_w'].sum()
    soma_eignv_w = data_df['eignv_w'].sum()
    
    for date in dates:
        cities = data_df[data_df['date'] == date]['geocode'].unique()
        num_cities = len(cities)
        random_cities = []
        random_cities = random.sample(geocodes, num_cities)
        geocodes = list(set(geocodes) - set(random_cities))
        
        selected_data = data_df[(data_df['geocode'].isin(random_cities))]
        
        sum_degree = selected_data['degree'].sum()
        sum_clustering = selected_data['clustering'].sum()
        sum_weight = selected_data['Weight'].sum()
        sum_betweenness = selected_data['betweenness'].sum()
        sum_closeness_w = selected_data['closeness_w'].sum()
        sum_eignv_w = selected_data['eignv_w'].sum()
        
        accumulated_sum_degree += sum_degree
        accumulated_sum_clustering += sum_clustering
        accumulated_sum_weight += sum_weight
        accumulated_sum_betweenness += sum_betweenness
        accumulated_sum_closeness_w += sum_closeness_w
        accumulated_sum_eignv_w += sum_eignv_w
        
        normalized_values['Degree'].append(accumulated_sum_degree / soma_degree)
        normalized_values['Clustering'].append(accumulated_sum_clustering / soma_clustering)
        normalized_values['Weight'].append(accumulated_sum_weight / soma_weight)
        normalized_values['Betweenness'].append(accumulated_sum_betweenness / soma_betweenness)
        normalized_values['Closeness_w'].append(accumulated_sum_closeness_w / soma_closeness_w)
        normalized_values['Eignv_w'].append(accumulated_sum_eignv_w / soma_eignv_w)
    
    df_normalized = pd.DataFrame(normalized_values)
    return df_normalized


def filter_cases(csv_file, n):
    df = pd.read_csv(
        csv_file,
        encoding='utf-8',
        sep=',',
        usecols=['ibgeID', 'newCases', 'totalCases', 'date'],
        dtype={'ibgeID': int}
    )
    filtered_df = df[(df['totalCases'] >= n) & (df['newCases'] >= 1) & (df['ibgeID'] != 0) & (df['ibgeID'] > 1000)]
    filtered_df = filtered_df.drop_duplicates(subset='ibgeID')    
    return filtered_df


graph = Graph.Read_GraphML("Datas/networks/grafo_Peso_Geral.GraphML")
geocodes = list(map(int, graph.vs["geocode"]))
degrees = graph.degree()
clustering = graph.transitivity_local_undirected()
weight = graph.strength(weights="weight")
graph.es['w_inv'] = 1.0 / np.array(graph.es['weight'])
betweenness = graph.betweenness(vertices=None, directed=False, cutoff=None, weights='w_inv')
closeness_w = graph.closeness(vertices=None, mode='all', cutoff=None, weights='w_inv', normalized=True)
eignv_w = graph.evcent(directed=False, scale=True, weights='w_inv', return_eigenvalue=False)

metrics_df = pd.DataFrame({
    "geocode": geocodes,
    "degree": degrees,
    "clustering": clustering,
    "Weight": weight,
    "betweenness": betweenness,
    "closeness_w": closeness_w,
    "eignv_w": eignv_w
})
delay = 0
df = filter_cases("Datas/Pre-processed/cases-brazil-cities-time_2020.csv", delay)
df = df.merge(metrics_df, left_on='ibgeID', right_on='geocode')
df = df.sort_values("date")

df_sum = df.groupby('date').sum().reset_index()

df_sum['Degree Accumulated'] = df_sum['degree'].cumsum()
df_sum['Clustering Accumulated'] = df_sum['clustering'].cumsum()
df_sum['Weight Accumulated'] = df_sum['Weight'].cumsum()
df_sum['Betweenness Accumulated'] = df_sum['betweenness'].cumsum()
df_sum['Closeness_w Accumulated'] = df_sum['closeness_w'].cumsum()
df_sum['Eignv_w Accumulated'] = df_sum['eignv_w'].cumsum()

metrics_table = df_sum[['date', 'Degree Accumulated', 'Clustering Accumulated', 'Weight Accumulated',
                        'Betweenness Accumulated', 'Closeness_w Accumulated', 'Eignv_w Accumulated']]

metrics_table = metrics_table.reset_index(drop=True)
dates = df_sum['date']

result_list = [calculate_random_metric_averages(df) for _ in range(10)]

combined_table = pd.concat(result_list)
mean_table = combined_table.groupby('Date').mean()
std_table = combined_table.groupby('Date').std()

upper_bound = mean_table + 2 * std_table
lower_bound = mean_table - 2 * std_table

point_indices = np.linspace(0, len(dates) - 1, num=100)
date_indices = np.arange(len(dates))

soma_degree = df['degree'].sum()
soma_weight = df['Weight'].sum()
soma_clustering = df['clustering'].sum()
soma_betweenness = df['betweenness'].sum()
soma_closeness = df['closeness_w'].sum()
soma_eignv_w = df['eignv_w'].sum()
degree_interp = np.interp(point_indices, date_indices, metrics_table['Degree Accumulated']) / soma_degree
clustering_interp = np.interp(point_indices, date_indices, metrics_table['Clustering Accumulated']) / soma_clustering
weight_interp = np.interp(point_indices, date_indices, metrics_table['Weight Accumulated']) / soma_weight
betweenness_interp = np.interp(point_indices, date_indices, metrics_table['Betweenness Accumulated'])/soma_betweenness
closeness_w_interp = np.interp(point_indices, date_indices, metrics_table['Closeness_w Accumulated'])/ soma_closeness
eignv_w_interp = np.interp(point_indices, date_indices, metrics_table['Eignv_w Accumulated'])/soma_eignv_w

plt.figure(figsize=(10, 6))
plt.plot(point_indices, degree_interp, 'ro-', label='Sum degree')
plt.plot(point_indices, clustering_interp, 'go-', label='Sum clustering')
plt.plot(point_indices, weight_interp, 'yo-', label='Sum Weight')
plt.plot(point_indices, betweenness_interp, 'bo-', label='Betweenness')
plt.plot(point_indices, closeness_w_interp, 'mo-', label='Closeness_w')
plt.plot(point_indices, eignv_w_interp, 'co-', label='Eignv_w')
plt.fill_between(mean_table.index, lower_bound['Degree'], upper_bound['Degree'], color='gray', alpha=0.3)
plt.fill_between(mean_table.index, lower_bound['Clustering'], upper_bound['Clustering'], color='gray', alpha=0.3)
plt.fill_between(mean_table.index, lower_bound['Weight'], upper_bound['Weight'], color='gray', alpha=0.3)
plt.fill_between([], [], [], color='gray', alpha=0.3, label='Development with Random Cases')
plt.legend()

visible_indices = np.arange(0, len(metrics_table), step=5)
visible_dates = metrics_table['date'].iloc[visible_indices]
plt.xticks(visible_indices, visible_dates, rotation=45)

plt.title(f'Development of Metrics Over Time: delay = {delay}')
plt.xlabel('Date')
plt.ylabel('Normalized Values')

plt.savefig(f'Datas/results/Desenvolvimento_metricas_{delay}cases.png')
plt.show()
