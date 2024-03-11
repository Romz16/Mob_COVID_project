from scipy.stats import kendalltau
import pandas as pd

# Seu DataFrame
data = {
    'lutador': ['Lutador A', 'Lutador B', 'Lutador C', 'Lutador D', 'Lutador E'],
    'altura': [175, 180, 170, 170, 185],
    'peso': [70, 55, 90, 75, 95],
}

# Ordens fornecidas por CV19
ordem_cv19_corrigida = ['Lutador D', 'Lutador B', 'Lutador A', 'Lutador C', 'Lutador E']

# Adiciona a ordem corrigida ao DataFrame
df = pd.DataFrame(data)

# Cria um mapeamento de lutadores para índices sequenciais
lutadores_indexados = {lutador: idx + 1 for idx, lutador in enumerate(df['lutador'])}

# Substitui os nomes dos lutadores por seus índices em 'CV19-ORD'
df['CV19-ORD'] = [lutadores_indexados[lutador] for lutador in ordem_cv19_corrigida]

print(df['CV19-ORD'])
# Calcule a correlação de Kendall entre 'altura' e 'CV19-ORD'
kendall_corr_altura, _ = kendalltau(df['altura'], df['CV19-ORD'])

# Calcule a correlação de Kendall entre 'peso' e 'CV19-ORD'
kendall_corr_peso, _ = kendalltau(df['peso'], df['CV19-ORD'])


print(f"Correlação de Kendall entre 'altura' e 'CV19-ORD': {kendall_corr_altura}")
print(f"Correlação de Kendall entre 'peso' e 'CV19-ORD': {kendall_corr_peso}")
