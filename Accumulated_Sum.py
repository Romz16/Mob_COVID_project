import random
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from igraph import Graph
from scipy.optimize import curve_fit
from matplotlib import rc
#Gerar tabela para verios minimum_casess
def calculate_random_metric_averages(data_df):
    normalized_values = {}
    accumulated_sum_degree = 0
    accumulated_sum_clustering = 0
    accumulated_sum_strength = 0
    accumulated_sum_betweenness = 0
    accumulated_sum_closeness_w = 0
    accumulated_sum_eignv_w = 0
    dates = data_df['date'].unique()
    normalized_values['Date'] = dates.tolist()
    normalized_values['Degree'] = []
    normalized_values['Clustering'] = []
    normalized_values['Weighted Strength'] = []
    normalized_values['Weighted Betweenness'] = []
    normalized_values['Weighted Closeness'] = []
    normalized_values['Weighted Eignv'] = []
    geocodes = list(data_df['geocode'])
    soma_degree = data_df['degree'].sum()
    soma_strength = data_df['Weighted_strength'].sum()
    soma_clustering = data_df['clustering'].sum()
    soma_betweenness = data_df['Weighted_betweenness'].sum()
    soma_closeness_w = data_df['Weighted_closeness'].sum()
    soma_eignv_w = data_df['Weighted_eignv'].sum()
    
    for date in dates:
        cities = data_df[data_df['date'] == date]['geocode'].unique()
        num_cities = len(cities)
        random_cities = []
        random_cities = random.sample(geocodes, num_cities)
        geocodes = list(set(geocodes) - set(random_cities))
        
        selected_data = data_df[(data_df['geocode'].isin(random_cities))]
        
        sum_degree = selected_data['degree'].sum()
        sum_clustering = selected_data['clustering'].sum()
        sum_strength = selected_data['Weighted_strength'].sum()
        sum_betweenness = selected_data['Weighted_betweenness'].sum()
        sum_closeness = selected_data['Weighted_closeness'].sum()
        sum_eignv = selected_data['Weighted_eignv'].sum()
        
        accumulated_sum_degree += sum_degree
        accumulated_sum_clustering += sum_clustering
        accumulated_sum_strength += sum_strength
        accumulated_sum_betweenness += sum_betweenness
        accumulated_sum_closeness_w += sum_closeness
        accumulated_sum_eignv_w += sum_eignv
        
        normalized_values['Degree'].append(accumulated_sum_degree / soma_degree)
        normalized_values['Clustering'].append(accumulated_sum_clustering / soma_clustering)
        normalized_values['Weighted Strength'].append(accumulated_sum_strength / soma_strength)
        normalized_values['Weighted Betweenness'].append(accumulated_sum_betweenness / soma_betweenness)
        normalized_values['Weighted Closeness'].append(accumulated_sum_closeness_w / soma_closeness_w)
        normalized_values['Weighted Eignv'].append(accumulated_sum_eignv_w / soma_eignv_w)
    
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

def logistic_growth(x, a, b, k):
    return a / (1 + b * np.exp(-k * x))
#x = eixo x de datas 
#a =  b=  k =
def plotGraphWithLogisticCurve(leng, minimum_cases):
    upper_bound = mean_table + 2 * std_table
    lower_bound = mean_table - 2 * std_table

    point_indices = np.linspace(0, len(dates) - 1, num=20)
    date_indices = np.arange(len(dates))
    
    soma_degree = df['degree'].sum()
    soma_strength = df['Weighted_strength'].sum()
    soma_clustering = df['clustering'].sum()
    soma_betweenness = df['Weighted_betweenness'].sum()
    soma_closeness = df['Weighted_closeness'].sum()
    soma_eignv = df['Weighted_eignv'].sum()
    
    degree_interp = np.interp(point_indices, date_indices, metrics_table['Degree Accumulated']) / soma_degree
    clustering_interp = np.interp(point_indices, date_indices, metrics_table['Clustering Accumulated']) / soma_clustering
    strength_interp = np.interp(point_indices, date_indices, metrics_table['Strength Accumulated']) / soma_strength
    betweenness_interp = np.interp(point_indices, date_indices, metrics_table['Betweenness Accumulated']) / soma_betweenness
    closeness_interp = np.interp(point_indices, date_indices, metrics_table['Closeness Accumulated']) / soma_closeness
    eignv_interp = np.interp(point_indices, date_indices, metrics_table['Eignv Accumulated']) / soma_eignv
    
    # Store the interpolation data in a pandas DataFrame
    interp_data_df = pd.DataFrame({
        'Degree': degree_interp,
        'Weighted Betweenness': betweenness_interp,
        'Clustering': clustering_interp,
        'Weighted Strength': strength_interp,
        'Weighted Closeness': closeness_interp,
        'Weighted Eigenvector': eignv_interp
    })
    
    # Calculate the area under each curve using numerical integration (trapezoidal rule)
    areas = interp_data_df.apply(lambda col: np.trapz(col, dx=1))

    plt.figure(figsize=(10, 6))

    # Ajuste da curva logística
    popt_degree, _ = curve_fit(logistic_growth, point_indices, degree_interp)
    popt_clustering, _ = curve_fit(logistic_growth, point_indices, clustering_interp)
    popt_strength, _ = curve_fit(logistic_growth, point_indices, strength_interp)
    popt_betweenness, _ = curve_fit(logistic_growth, point_indices, betweenness_interp)
    popt_closeness, _ = curve_fit(logistic_growth, point_indices, closeness_interp)
    popt_eignv, _ = curve_fit(logistic_growth, point_indices, eignv_interp)
    
    # Imprimindo os parâmetros ajustados para cada métrica


    # Curvas ajustadas usando os parâmetros estimados
    degree_fit = logistic_growth(point_indices, *popt_degree)
    clustering_fit = logistic_growth(point_indices, *popt_clustering)
    strength_fit = logistic_growth(point_indices, *popt_strength)
    betweenness_fit = logistic_growth(point_indices, *popt_betweenness)
    closeness_fit = logistic_growth(point_indices, *popt_closeness)
    eignv_fit = logistic_growth(point_indices, *popt_eignv)
    
    
    
# # Latex font --------------------
#     rc('text', usetex=True)
#     font = {'family' : 'normal',
#             'weight' : 'bold',
#             'size'   : 12}

#     rc('font', **font)
#     params = {'legend.fontsize': 14}
#     plt.rcParams.update(params)
#     # -------------------------------
    # Plotar curvas ajustadas
    plt.plot(point_indices, degree_fit, 'rs--', label=f'Logistic Fit (Degree) - A: {popt_degree[0]:.2f}, B: {popt_degree[1]:.2f}, K: {popt_degree[2]:.2f}')
    plt.plot(point_indices, clustering_fit, 'go--', label=f'Logistic Fit (Clustering) - A: {popt_clustering[0]:.2f}, B: {popt_clustering[1]:.2f}, K: {popt_clustering[2]:.2f}')
    plt.plot(point_indices, strength_fit, 'yd--', label=f'Logistic Fit (Weighted Strength) - A: {popt_strength[0]:.2f}, B: {popt_strength[1]:.2f}, K: {popt_strength[2]:.2f}')
    plt.plot(point_indices, betweenness_fit, 'b^--', label=f'Logistic Fit (Weighted Betweenness) - A: {popt_betweenness[0]:.2f}, B: {popt_betweenness[1]:.2f}, K: {popt_betweenness[2]:.2f}')
    plt.plot(point_indices, closeness_fit, 'm*--', label=f'Logistic Fit (Weighted Closeness) - A: {popt_closeness[0]:.2f}, B: {popt_closeness[1]:.2f}, K: {popt_betweenness[2]:.2f}')
    plt.plot(point_indices, eignv_fit, 'c.--', label=f'Logistic Fit (Eigenvector Closeness) - A: {popt_eignv[0]:.2f}, B: {popt_eignv[1]:.2f}, K: {popt_eignv[2]:.2f}')
    plt.fill_between(mean_table.index, lower_bound['Degree'], upper_bound['Degree'], color='gray', alpha=0.3)
    plt.fill_between(mean_table.index, lower_bound['Clustering'], upper_bound['Clustering'], color='gray', alpha=0.3)
    plt.fill_between(mean_table.index, lower_bound['Weighted Strength'], upper_bound['Weighted Strength'], color='gray', alpha=0.3)
    plt.fill_between([], [], [], color='gray', alpha=0.3, label='Development with Random Cases')
    plt.legend(fontsize='large', title=f'amount of cities: {leng}')

    visible_indices = np.arange(0, len(metrics_table), step=10)
    visible_dates = metrics_table['date'].iloc[visible_indices]
    plt.xticks(visible_indices, visible_dates, rotation=45)
    
    plt.xlabel('Date')
    plt.ylabel(f'Normalized Values')
    plt.tight_layout()
    plt.xlim(0, len(metrics_table) - 1)  
    plt.ylim(0, 1) 
    plt.savefig(f'Datas/results/LogisticFit_{minimum_cases}cases.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

def plotGraph(leng,minimum_cases):
    upper_bound = mean_table + 2 * std_table
    lower_bound = mean_table - 2 * std_table

    point_indices = np.linspace(0, len(dates) - 1, num=20)
    date_indices = np.arange(len(dates))
    soma_degree = df['degree'].sum()
    soma_strength = df['Weighted_strength'].sum()
    soma_clustering = df['clustering'].sum()
    soma_betweenness = df['Weighted_betweenness'].sum()
    soma_closeness = df['Weighted_closeness'].sum()
    soma_eignv = df['Weighted_eignv'].sum()
    degree_interp = np.interp(point_indices, date_indices, metrics_table['Degree Accumulated']) / soma_degree
    clustering_interp = np.interp(point_indices, date_indices, metrics_table['Clustering Accumulated']) / soma_clustering
    strength_interp = np.interp(point_indices, date_indices, metrics_table['Strength Accumulated']) / soma_strength
    betweenness_interp = np.interp(point_indices, date_indices, metrics_table['Betweenness Accumulated'])/soma_betweenness
    closeness_interp = np.interp(point_indices, date_indices, metrics_table['Closeness Accumulated'])/ soma_closeness
    eignv_interp = np.interp(point_indices, date_indices, metrics_table['Eignv Accumulated'])/soma_eignv
    # Store the interpolation data in a pandas DataFrame
    interp_data_df = pd.DataFrame({
    'Degree': degree_interp,
    'Weighted Betweenness': betweenness_interp,
    'Clustering': clustering_interp,
    'Weighted Strength': strength_interp,
    'Weighted Closeness': closeness_interp,
    'Weighted Eigenvector': eignv_interp })
    # Calculate the area under each curve using numerical integration (trapezoidal rule)
    areas = interp_data_df.apply(lambda col: np.trapz(col, dx=1))
    popt_clustering, _ = curve_fit(logistic_growth, point_indices, clustering_interp)
    popt_betweenness, _ = curve_fit(logistic_growth, point_indices, betweenness_interp)
    # Curvas ajustadas usando os parâmetros estimados
    clustering_fit = logistic_growth(point_indices, *popt_clustering)    
    betweenness_fit = logistic_growth(point_indices, *popt_betweenness)

    plt.figure(figsize=(10, 6))
    plt.plot(point_indices, degree_interp, 'ro-', label=f'Degree: {areas["Degree"]:.2f}', marker='s')
    plt.plot(point_indices, clustering_interp, 'gs-', label=f'Clustering: {areas["Clustering"]:.2f}', marker='^')
    plt.plot(point_indices, strength_interp, 'yd-', label=f'Weighted Strength: {areas["Weighted Strength"]:.2f}', marker='v')
    plt.plot(point_indices, betweenness_interp, 'b^-', label=f'Weighted Betweenness: {areas["Weighted Betweenness"]:.2f}', marker='x')
    plt.plot(point_indices, closeness_interp, 'mo-', label=f'Weighted Closeness: {areas["Weighted Closeness"]:.2f}', marker='+')
    plt.plot(point_indices, eignv_interp, 'c*-', label=f'Weighted Eignv: {areas["Weighted Eigenvector"]:.2f}',marker='o')
    plt.fill_between(mean_table.index, lower_bound['Degree'], upper_bound['Degree'], color='gray', alpha=0.3)
    plt.fill_between(mean_table.index, lower_bound['Clustering'], upper_bound['Clustering'], color='gray', alpha=0.3)
    plt.fill_between(mean_table.index, lower_bound['Weighted Strength'], upper_bound['Weighted Strength'], color='gray', alpha=0.3)
    plt.fill_between([], [], [], color='gray', alpha=0.3, label='Development with Random Cases')
    
    plt.plot(point_indices, clustering_fit, 'g--')
    plt.plot(point_indices, betweenness_fit, 'b--')
   # Annotate the plot with logistic fit information and arrows
    plt.annotate(f'Logistic Fit (Clustering)\nA: {popt_clustering[0]:.2f}, B: {popt_clustering[1]:.2f}, K: {popt_clustering[2]:.2f}',
                 xy=(point_indices[5], clustering_fit[5]), xycoords='data',
                 xytext=(point_indices[8], clustering_fit[6]), textcoords='data',
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    
    plt.annotate(f'Logistic Fit (Weighted Betweenness)\nA: {popt_betweenness[0]:.2f}, B: {popt_betweenness[1]:.2f}, K: {popt_betweenness[2]:.2f}',
                 xy=(point_indices[4], betweenness_fit[4]), xycoords='data',
                 xytext=(point_indices[0], betweenness_fit[5]), textcoords='data',
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.legend(fontsize='medium')
    plt.text(0.8, 0.90, f'Number of Cities: {leng}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
  
    # -------------------------------   
    visible_indices = np.arange(0, len(metrics_table), step=10)
    visible_dates = metrics_table['date'].iloc[visible_indices]
    plt.xticks(visible_indices, visible_dates, rotation=45)
    
    plt.xlabel('Date')
    plt.ylabel(f'Normalized accumulated metric')
    plt.tight_layout()
    plt.xlim(0, len(metrics_table) - 1)  
    plt.ylim(0, 1) 
    plt.savefig(f'Datas/results/Accumulated_metricas_{minimum_cases}cases.pdf', bbox_inches='tight')
    plt.show()
    plt.close()


def graph_plot_minimum_casesXintegral():      
    #plt.figure(figsize=(10, 8))
    labels = ['Degrees', 'Weighted Betweenness', 'Clustering', 'Weighted Strength', 'Weighted Closeness ', 'Weighted Eigenvector ','Mean cases']
    #markers = ['o', 's', '^', '+', 'v', 'x','~']  
    data = [degrees_avg, betweenness_avg, clustering_avg, strength_avg, closeness_avg, eignv_avg,mean_random_avg]

    # for label, marker, values in zip(labels, markers, data):
    #     plt.plot(min_cases, values, marker + '-', label=f'Avg: {round(sum(values) / len(values), 2)} - {label}')
    #     for x, y in zip(min_cases, values):
    #         plt.text(x, y, str(round(y, 2)), ha='center', va='bottom')

    # plt.xlabel('Minimum number of cases')
    # plt.ylabel('Area under curve')
    # plt.autoscale(axis='x', tight=True)
    # plt.autoscale(axis='y', tight=True)
    
    # # Ajuste da legenda para a posição inferior quase no meio
    # plt.legend(title='Accumulated', loc='lower center', ncol=2)
    
    # plt.tight_layout()
    
    # # Salvar o gráfico como um arquivo de imagem
    # plt.savefig('Datas/results/graph_MinimumIntegral_SUM.pdf')
    
    # # Exibir o gráfico
    # plt.show()
     # Gerar a tabela CSV
    with open('Datas/results/table_MinimumIntegral_SUM.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Escrever o cabeçalho da tabela
        writer.writerow(['Minimum number of cases'] + labels)
        
        # Escrever os dados
        for min_case, values in zip(min_cases, zip(*data)):
            writer.writerow([min_case] + list(values))
    print("FEito")
    
import csv 

graph = Graph.Read_GraphML("Datas/networks/grafo_Peso_Geral.GraphML")
geocodes = list(map(int, graph.vs["geocode"]))
degrees = graph.degree()
clustering = graph.transitivity_local_undirected()
weighted_strength = graph.strength(weights="weight")
graph.es['w_inv'] = 1.0 / np.array(graph.es['weight'])
weighted_betweenness = graph.betweenness(vertices=None, directed=False, cutoff=None, weights='w_inv')
weighted_closeness = graph.closeness(vertices=None, mode='all', cutoff=None, weights='w_inv', normalized=True)
weighted_eignv = graph.evcent(directed=False, scale=True, weights='w_inv', return_eigenvalue=False)

metrics_df = pd.DataFrame({
    "geocode": geocodes,
    "degree": degrees,
    "clustering": clustering,
    "Weighted_strength": weighted_strength,
    "Weighted_betweenness": weighted_betweenness,
    "Weighted_closeness": weighted_closeness,
    "Weighted_eignv": weighted_eignv
})
degrees_avg = []
betweenness_avg = []
clustering_avg = []
strength_avg = []
closeness_avg = []
eignv_avg = []
mean_random_avg = []
std_random_avg = []
min_cases = list(range(1, 61))
leng = []
if len(min_cases)>1:
    for minimum_cases in min_cases :   
        df = filter_cases("Datas/Pre-processed/cases-brazil-cities-time_2020.csv", minimum_cases)
        leng.append(len(df))
        df = df.merge(metrics_df, left_on='ibgeID', right_on='geocode')
        df_sum = df.groupby('date').sum().reset_index()
        
        soma_degree = df['degree'].sum()
        soma_strength = df['Weighted_strength'].sum()
        soma_clustering = df['clustering'].sum()
        soma_betweenness = df['Weighted_betweenness'].sum()
        soma_closeness = df['Weighted_closeness'].sum()
        soma_eignv = df['Weighted_eignv'].sum()
        # Calculate the accumulated values for each metric
        df_sum['Degree Accumulated'] = df_sum['degree'].cumsum() / soma_degree
        df_sum['Clustering Accumulated'] = df_sum['clustering'].cumsum() / soma_clustering
        df_sum['Strength Accumulated'] = df_sum['Weighted_strength'].cumsum() / soma_strength
        df_sum['Betweenness Accumulated'] = df_sum['Weighted_betweenness'].cumsum() / soma_betweenness
        df_sum['Closeness Accumulated'] = df_sum['Weighted_closeness'].cumsum() / soma_closeness
        df_sum['Eignv Accumulated'] = df_sum['Weighted_eignv'].cumsum() / soma_eignv
        result_list = [calculate_random_metric_averages(df) for _ in range(700)]
        combined_table = pd.concat(result_list)
        mean_table = combined_table.groupby('Date').mean()
        std_table = combined_table.groupby('Date').std()
        
        # Calculate the accumulated area under each curve
        degrees_area = np.trapz(df_sum['Degree Accumulated'], dx=1)
        betweenness_area = np.trapz(df_sum['Betweenness Accumulated'], dx=1)
        clustering_area = np.trapz(df_sum['Clustering Accumulated'], dx=1)
        strength_area = np.trapz(df_sum['Strength Accumulated'], dx=1)
        closeness_w_area = np.trapz(df_sum['Closeness Accumulated'], dx=1)
        eignv_w_area = np.trapz(df_sum['Eignv Accumulated'], dx=1)
        mean_table_area = np.trapz(mean_table,dx=1)
        std_table_area = np.trapz(std_table,dx=1)
        # Append the calculated areas to the respective lists
        degrees_avg.append(degrees_area)
        betweenness_avg.append(betweenness_area)
        clustering_avg.append(clustering_area)
        strength_avg.append(strength_area)
        closeness_avg.append(closeness_w_area)
        eignv_avg.append(eignv_w_area)
        mean_random_avg.append(mean_table_area)
        std_random_avg.append(std_table_area)
    graph_plot_minimum_casesXintegral()
else:
    df = filter_cases("Datas/Pre-processed/cases-brazil-cities-time_2020.csv", min_cases[0])
    leng.append(len(df))
    df = df.merge(metrics_df, left_on='ibgeID', right_on='geocode')
    df = df.sort_values("date")
    df_sum = df.groupby('date').sum().reset_index()

    # Calculate the accumulated values for each metric
    df_sum['Degree Accumulated'] = df_sum['degree'].cumsum() 
    df_sum['Clustering Accumulated'] = df_sum['clustering'].cumsum() 
    df_sum['Strength Accumulated'] = df_sum['Weighted_strength'].cumsum() 
    df_sum['Betweenness Accumulated'] = df_sum['Weighted_betweenness'].cumsum() 
    df_sum['Closeness Accumulated'] = df_sum['Weighted_closeness'].cumsum()
    df_sum['Eignv Accumulated'] = df_sum['Weighted_eignv'].cumsum() 
        
    metrics_table = df_sum[['date', 'Degree Accumulated', 'Clustering Accumulated', 'Strength Accumulated',
                                'Betweenness Accumulated', 'Closeness Accumulated', 'Eignv Accumulated']]

    metrics_table = metrics_table.reset_index(drop=True)
    dates = df_sum['date']
    result_list = [calculate_random_metric_averages(df) for _ in range(500)]
    combined_table = pd.concat(result_list)
    mean_table = combined_table.groupby('Date').mean()
    std_table = combined_table.groupby('Date').std()
    
    plotGraph(leng[0],min_cases[0])