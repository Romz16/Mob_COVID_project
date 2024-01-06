from scipy.stats import kendalltau

# Dados de classificações de dois avaliadores (exemplo fictício)
avaliador1 = [4, 2, 1, 2.5, 4]
avaliador2 = [4, 2, 1, 3, 5]

# Calcula a correlação de Kendall
correlacao, p_valor = kendalltau(avaliador1, avaliador2)

# Exibe os resultados
print("Correlação de Kendall:", correlacao)
print("Valor-p:", p_valor)
