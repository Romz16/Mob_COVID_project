!pip install igraph
from igraph import Graph
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib import rc
# Open the GraphML file and create a Graph object from it
graph2 = Graph.Read_GraphML("grafo_Peso_aerio.GraphML")
graph3 = Graph.Read_GraphML("grafo_Peso_hidro.GraphML")
graph4 = Graph.Read_GraphML("grafo_Peso_rodo.GraphML")
graph = Graph.Read_GraphML("grafo_Peso_Geral.GraphML")

# Calcular métricas para cada rede
metricas_graph2 = {
    'nós': graph2.vcount(),
    'arestas': graph2.ecount(),
    'grau_médio': np.mean(graph2.degree()),
    'coeficiente_de_heterogeneidade': graph2.transitivity_undirected(),
    'densidade': graph2.density()
}

metricas_graph3 = {
    'nós': graph3.vcount(),
    'arestas': graph3.ecount(),
    'grau_médio': np.mean(graph3.degree()),
    'coeficiente_de_heterogeneidade': graph3.transitivity_undirected(),
    'densidade': graph3.density()
}
metricas_graph4 = {
    'nós': graph4.vcount(),
    'arestas': graph4.ecount(),
    'grau_médio': np.mean(graph4.degree()),
    'coeficiente_de_heterogeneidade': graph4.transitivity_undirected(),
    'densidade': graph4.density()
}

metricas_graph = {
    'nós': graph.vcount(),
    'arestas': graph.ecount(),
    'grau_médio': np.mean(graph.degree()),
    'coeficiente_de_heterogeneidade': graph.transitivity_undirected(),
    'densidade': graph.density()
}

# Exportar as métricas para um arquivo GRAPHML (opcional)
# Você pode adicionar essas métricas como atributos dos nós ou da rede
# e salvar o arquivo GRAPHML com as métricas calculadas

# Apresentar os números e métricas
print("\nMétricas para Rede aeria:")
print(metricas_graph2)

print("\nMétricas para Rede hidro:")
print(metricas_graph3)

print("\nMétricas para Rede rodo:")
print(metricas_graph4)

print("\nMétricas para Rede Fundida:")
print(metricas_graph)