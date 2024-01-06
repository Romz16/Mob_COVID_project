import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc

def graph_plot_minimum_casesXCorrelation(data_0, data_20,data_40,data_60,data_80, data_100):

    minimum = [0,20,40,60,80,100]
    # Kendall correlations for each minimum value
    Kendalls = []  
    for data in [data_0, data_20,data_40,data_60,data_80, data_100]:
        metrics_df = data.drop(columns=['DATES'])
        # Calculate Kendall and Kendall correlations
        correlations_Kendall = metrics_df.corrwith(data['DATES'], method='kendall')
        Kendalls.append(correlations_Kendall)

    metrics = ['degrees', 'betweenness', 'clustering', 'strength', 'closeness_w', 'eignv_w']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    legend_labels = []

    for i, metric in enumerate(metrics):
        color = colors[i % len(colors)]  # Select a color from the color list
        metric_values = [correlations[metric] for correlations in Kendalls]
        plt.plot(minimum, metric_values, marker='o', linestyle='-', label=metric, color=color)
        legend_labels.append(metric)  # Add the metric to the legend label list

        # Add labels to the data points
        for j, value in enumerate(metric_values):
            plt.text(minimum[j], value, f'{value:.4f}', ha='left', va='bottom', fontsize=10)

    # Add labels to the axes
    plt.xlabel('Minimum Values')
    plt.ylabel('Kendall Correlation')
    plt.tight_layout()  
    plt.autoscale(axis='x', tight=True)
    plt.autoscale(axis='y', tight=True)  # Also adjust the Y-axis automatically
    # Define a mapping of old names to new names
    name_mapping = {
        'degrees': 'Degrees',
        'betweenness': 'Weighted Betweenness',
        'clustering': 'Clustering',
        'strength': 'Weighted Strength',
        'closeness_w': 'Weighted Closeness',
        'eignv_w': 'Weighted Eigenvector'
    }
    # Replace the legend labels with the new names
    legend_labels = [name_mapping.get(label, label) for label in legend_labels]
    # Add a legend for the metrics with the labels
    plt.legend(legend_labels)
    #plt.savefig("Datas/results/Kendall.png")
    plt.show()

# Loading data from each file into separate DataFrames
data_1_df = pd.read_excel("Datas/results/results_tables/result_table-0cases.xlsx")
data_2_df = pd.read_excel("Datas/results/results_tables/result_table-20cases.xlsx")
data_3_df = pd.read_excel("Datas/results/results_tables/result_table-40cases.xlsx")
data_4_df = pd.read_excel("Datas/results/results_tables/result_table-60cases.xlsx")
data_5_df = pd.read_excel("Datas/results/results_tables/result_table-80cases.xlsx")
data_6_df = pd.read_excel("Datas/results/results_tables/result_table-100cases.xlsx")

graph_plot_minimum_casesXCorrelation(data_1_df, data_2_df,data_3_df,data_4_df,data_5_df, data_6_df)
