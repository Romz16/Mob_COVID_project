from igraph import Graph
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
import matplotlib.pyplot as plt
#metrics = ['degrees', 'betweenness', 'clustering', 'strength', 'closeness_w', 'eignv_w']
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
    "Weighted_eignv": weighted_eignv,
    "Ordered_covidcases":
})




# Suponhamos que você tenha um DataFrame com os dados
# Exemplo fictício para ilustração, substitua pelos seus dados reais
data = {
    'Geocode': [1, 2, 3, 4, 5],
    'Metrica_1': [10, 15, 8, 20, 12],
    'Metrica_2': [30, 25, 40, 15, 35],
    'Casos_COVID': [3, 5, 1, 4, 2],  # Ordenei de acordo com a ordem de surgimento
}
df = pd.DataFrame(data)

# Calcula os coeficientes de Kendall e Spearman para cada métrica
kendall_corr_metrica_1, _ = kendalltau(df['Metrica_1'], df['Casos_COVID'])
spearman_corr_metrica_1, _ = spearmanr(df['Metrica_1'], df['Casos_COVID'])

kendall_corr_metrica_2, _ = kendalltau(df['Metrica_2'], df['Casos_COVID'])
spearman_corr_metrica_2, _ = spearmanr(df['Metrica_2'], df['Casos_COVID'])

# Cria um DataFrame para os resultados
resultados = pd.DataFrame({
    'Métrica': ['Métrica 1', 'Métrica 2'],
    'Kendall': [kendall_corr_metrica_1, kendall_corr_metrica_2],
    'Spearman': [spearman_corr_metrica_1, spearman_corr_metrica_2],
})

# Plota um gráfico de barras
resultados.plot(x='Métrica', y=['Kendall', 'Spearman'], kind='bar', rot=0)
plt.ylabel('Coeficiente de Correlação')
plt.title('Comparação de Coeficientes de Correlação para Métrica 1 e Métrica 2')
plt.show()
