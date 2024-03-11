from igraph import Graph
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import vulnerability as vn

# Used to transform vertex data from 'graph' into a matrix
def get_matrix():
    # Get vertex information
    # Calculate the inverse of edge weights and store them in the 'w_inv' property
    graph.es['w_inv'] = 1.0 / np.array(graph.es['weight'])
    vertex_info = []
    geocodes = graph.vs["geocode"]
    degrees = graph.degree()
    betweenness = graph.betweenness(vertices=None, directed=False, cutoff=None, weights='w_inv')
    clustering = graph.transitivity_local_undirected()
    strength = graph.strength(weights="weight")
    closeness_w = graph.closeness(vertices=None, mode='all', cutoff=None, weights='w_inv', normalized=True)
    eignv_w = graph.evcent(directed=False, scale=True, weights='w_inv', return_eigenvalue=False)
    #prank_w = nx.pagerank(g_nx, alpha=0.85, weight='weight')
    #vuln_w = vn.vulnerability(graph, weights='w_inv')
    geocodes_int = list(map(int, geocodes))
    geocodes_strings = list(map(str,geocodes_int))
    # Construct the vertex information matrix
    vertex_info = list(zip(geocodes_strings, degrees, betweenness, clustering, strength, closeness_w, eignv_w))

    return vertex_info

# Filter cases from a CSV file based on a minimum number of cases
# and return a list of filtered elements and their data
def filter_cases(csv_file, n):
    # Read the CSV file and define the columns to be considered
    df = pd.read_csv(
        csv_file,
        encoding='utf-8',
        sep=',',
        usecols=['ibgeID', 'newCases', 'totalCases', 'date'],
        dtype={'ibgeID': int}  # Define the appropriate data type for ibgeID if possible
    )
    # Filter records that meet the requirements
    filtered_df = df[(df['totalCases'] >= n) & (df['newCases'] >= 1) & (df['ibgeID'] != 0) & (df['ibgeID'] > 1000)]
    # Remove duplicate records based on ibgeID
    filtered_df = filtered_df.drop_duplicates(subset='ibgeID')
    # Return the results as a list of tuples
    filtered_cases = list(zip(filtered_df['date'], filtered_df['ibgeID'].apply(repr)))
    return filtered_cases

# Filter records from list A based on a filtered list of cities with more than N Covid cases
def filter_records(list_A, list_B):
    # Create a set of all ids in list B for easy verification
    id_set = set(record[1] for record in list_B)
    # Filter records in list A that meet the criteria
    filtered_list = [record for record in list_A if record[0] in id_set]
    return filtered_list

# Verify the similarity between an ordered list of cities based on a certain metric
# and the list of cities with B Covid cases over time
def compare_columns(metrics_matrix, col_idx):
    # Select the columns from the matrices
    result = []
    aux = metrics_matrix[metrics_matrix[:, col_idx].argsort()[::-1]]
    for i in range(1, matrix_size + 1):
        col1_elements = set(Id_matrix_covid[:i])
        col2_elements = set(aux[:i, 0].astype(float))
        intersection = col1_elements.intersection(col2_elements)
        similarity_percentage = (len(intersection) / len(col1_elements))

        result.append(similarity_percentage)
    return result

#Creates a table with the similarity rate of each metric over time
def setUp_comparsion_table():
# Set up the comparison matrix
    comparison_matrix = np.array([matrix_covid_dates, degrees_similarity, betweenness_similarity, clustering_similarity,
                              strength_similarity, closeness_w_similarity, eignv_w_similarity])
    Table_Names = ["DATES", "degrees", "betweenness", "clustering", "strength", "closeness_w", "eignv_w"]
    Final_Table = pd.DataFrame(comparison_matrix.T, columns=Table_Names)
# Save the final table to an Excel file
    excel_name = "Datas/results/result_table2.xlsx"
    Final_Table.to_excel(excel_name, index=False)
    
def graph_plot_DateXMetrics(step,cases):
    
    # Indices for points along the x-axis intervals
    point_indices = np.linspace(0, len(matrix_covid_dates)-1, num=100)
    date_indices = np.arange(len(matrix_covid_dates))

    # Interpolation of similarity points
    degrees_interp = np.interp(point_indices, date_indices, degrees_similarity)
    betweenness_interp = np.interp(point_indices, date_indices, betweenness_similarity)
    clustering_interp = np.interp(point_indices, date_indices, clustering_similarity)
    strength_interp = np.interp(point_indices, date_indices, strength_similarity)
    closeness_w_interp = np.interp(point_indices, date_indices, closeness_w_similarity)
    eignv_w_interp = np.interp(point_indices, date_indices, eignv_w_similarity)

    # Configure the plot
    plt.figure(figsize=(10, 6))
    plt.plot(point_indices, degrees_interp, 'ro-', label='Degree')
    plt.plot(point_indices, betweenness_interp, 'go-', label='Betweenness')
    plt.plot(point_indices, clustering_interp, 'yo-', label='Clustering')
    plt.plot(point_indices, strength_interp, 'mo-', label='Strength')
    plt.plot(point_indices, closeness_w_interp, 'co-', label=' Weighted Closeness ')
    plt.plot(point_indices, eignv_w_interp, 'ko-', label='Weighted Eigenvector ')

    # Adjust the display of the x-axis
    visible_indices = np.arange(0, len(matrix_covid_dates), step=step)
    visible_dates = matrix_covid_dates[visible_indices]
    plt.xticks(visible_indices, visible_dates, rotation=45)

    # Configure the plot title and axis labels
    plt.title(f'Similarity between Metrics and Covid Cases over Time: Delay of {cases} cases')
    plt.xlabel('Date')
    plt.ylabel('Similarity')

    # Display the legend
    plt.legend()

     # Save the plot as an image file
    plt.savefig(f'Datas/results/graph_RODO_{cases}cases.png')

    # Close the plot to release resources
    plt.close()
    
def graph_plot_DelayXMedia():
    # Plotting the graph
    plt.figure(figsize=(8, 6))
    labels = ['Degrees', 'Betweenness', 'Clustering', 'Strength', ' Weighted Closeness ', ' Weighted Eigenvector ']
    data = [degrees_avg, betweenness_avg, clustering_avg, strength_avg, closeness_w_avg, eignv_w_avg]

    for label, values in zip(labels, data):
        plt.plot(delays, values, 'o-', label=f'Avg: {round(sum(values) / len(values), 2)} - {label}')
        for x, y in zip(delays, values):
            plt.text(x, y, str(round(y, 2)), ha='center', va='bottom')

    plt.xlabel('Delay')
    plt.ylabel('Average Similarity')
    plt.title('AVERAGE NETWORK CENTRALITY')
    plt.ylim(0, 1)
    plt.legend(title='Average Similarity', loc='best')

    plt.tight_layout()
     # Save the plot as an image file
    plt.savefig('Datas/results/graph_dalay-Media_RODO.png')

    # Close the plot to release resources
    plt.close()
 
# Open the GraphML file and create a Graph object from it
# There are some metrics implemented in Networkx and not in igraph
# This is why we use Networkx below
graph = Graph.Read_GraphML("Datas/networks/grafo_Peso_Geral.GraphML")
#g_nx = nx.readwrite.graphml.read_graphml("weighted_road_graph.graphml")

delays = [10,20,40,60,80,100]
boolQ= True
# Initialize lists to store average similarity values for each column
degrees_avg = []
betweenness_avg = []
clustering_avg = []
strength_avg = []
closeness_w_avg = []
eignv_w_avg = []

if len(delays)>1:
    for delay in delays:
        covidID_list = filter_cases("Datas/Pre-processed/cases-brazil-cities-time_2020.csv",delay)
        metrics_list = filter_records(get_matrix(), covidID_list)
        metrics_matrix = np.array(metrics_list, dtype=float)
        covid_matrix = np.array(covidID_list)
        # Receive the size of the metrics matrix to match the size of the matrices
        matrix_size = len(metrics_matrix)
        #matrix_size = 100
        matrix_covid_dates = covid_matrix[:matrix_size, 0]
        Id_matrix_covid = covid_matrix[:matrix_size, 1].astype(float)
        # Calculate the similarity percentage for each metric
        degrees_similarity = compare_columns(metrics_matrix, 1)
        betweenness_similarity = compare_columns(metrics_matrix, 2)
        clustering_similarity = compare_columns(metrics_matrix, 3)
        strength_similarity = compare_columns(metrics_matrix, 4)
        closeness_w_similarity = compare_columns(metrics_matrix, 5)
        eignv_w_similarity = compare_columns(metrics_matrix, 6)
        #graph_plot_DateXMetrics(250,delay)
        if boolQ:
            #Calculate average similarity values for each column
            degrees_avg.append(np.mean(degrees_similarity))
            betweenness_avg.append(np.mean(betweenness_similarity))
            clustering_avg.append(np.mean(clustering_similarity))
            strength_avg.append(np.mean(strength_similarity))
            closeness_w_avg.append(np.mean(closeness_w_similarity))
            eignv_w_avg.append(np.mean(eignv_w_similarity))
    graph_plot_DelayXMedia()
else:
    # Receive lists with filtered IDs
    covidID_list = filter_cases("Datas/Pre-processed/cases-brazil-cities-time_2020.csv", delays[0])
    metrics_list = filter_records(get_matrix(), covidID_list)

    metrics_matrix = np.array(metrics_list, dtype=float)
    covid_matrix = np.array(covidID_list)

    # Receive the size of the metrics matrix to match the size of the matrices
    #matrix_size = len(metrics_matrix)
    matrix_size = 100

    matrix_covid_dates = covid_matrix[:matrix_size, 0]
    Id_matrix_covid = covid_matrix[:matrix_size, 1].astype(float)
    

    # Calculate the similarity percentage for each metric
    degrees_similarity = compare_columns(metrics_matrix, 1)
    betweenness_similarity = compare_columns(metrics_matrix, 2)
    clustering_similarity = compare_columns(metrics_matrix, 3)
    strength_similarity = compare_columns(metrics_matrix, 4)
    closeness_w_similarity = compare_columns(metrics_matrix, 5)
    eignv_w_similarity = compare_columns(metrics_matrix, 6)


# setUp_comparsion_table()