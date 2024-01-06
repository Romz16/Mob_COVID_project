import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

# Função para calcular a correlação de Kendall entre o vetor original e as métricas
def calculate_kendall_correlations(data_frames):
    metrics = ['degrees', 'betweenness', 'clustering', 'strength', 'closeness_w', 'eignv_w']
    kendall_correlations = []

    for metric in metrics:
        correlations = []
        for data in data_frames:
            kendall, _ = kendalltau(data['DATES'], data[metric])
            correlations.append(kendall)
        kendall_correlations.append(correlations)

    return kendall_correlations

def plot_kendall_correlations(kendall_correlations, delays):
    metrics = ['Degrees', 'Weighted Betweenness', 'Clustering', 'Weighted Strength', 'Weighted Closeness', 'Weighted Eigenvector']

    plt.figure(figsize=(12, 6))  # Tamanho do gráfico

    for i, metric in enumerate(metrics):
        plt.plot(delays, kendall_correlations[i], marker='o', label=f'{metric}')

    plt.title('Kendall Correlations for Metrics')
    plt.xlabel('Delays (Cases)')
    plt.ylabel('Kendall Correlation')
    plt.legend()
    plt.grid()
    plt.show()

# Loading data from each file into separate DataFrames
data_1_df = pd.read_excel("Datas/results/results_tables/result_table-0cases.xlsx")
data_2_df = pd.read_excel("Datas/results/results_tables/result_table-20cases.xlsx")
data_3_df = pd.read_excel("Datas/results/results_tables/result_table-40cases.xlsx")
data_4_df = pd.read_excel("Datas/results/results_tables/result_table-60cases.xlsx")
data_5_df = pd.read_excel("Datas/results/results_tables/result_table-80cases.xlsx")
data_6_df = pd.read_excel("Datas/results/results_tables/result_table-100cases.xlsx")

data_frames = [data_1_df, data_2_df, data_3_df, data_4_df, data_5_df, data_6_df]

# Calculate Kendall correlations
kendall_correlations = calculate_kendall_correlations(data_frames)

# Values for delays (cases)
delays = [0, 20, 40, 60, 80, 100]

# Plot Kendall correlations for each metric
plot_kendall_correlations(kendall_correlations, delays)
