import pandas as pd

# Ler o arquivo de médias
df = pd.read_csv('medias_por_escola_planilha.csv', encoding='utf-8-sig')

print('=== RESUMO DOS DADOS PROCESSADOS ===')
print(f'Total de registros: {len(df)}')
print(f'Planilhas processadas: {list(df["Planilha"].unique())}')

print('\n=== PRIMEIRAS 10 LINHAS ===')
print(df.head(10).to_string(index=False))

print('\n=== ESTATÍSTICAS POR PLANILHA ===')
for planilha in df['Planilha'].unique():
    dados = df[df['Planilha'] == planilha]
    print(f'\n{planilha}:')
    print(f'  Escolas: {len(dados)}')
    print(f'  Média: {dados["Media"].mean():.2f}')
    print(f'  Menor: {dados["Media"].min():.2f}')
    print(f'  Maior: {dados["Media"].max():.2f}')
