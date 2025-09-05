import pandas as pd
import numpy as np

# Analisar a primeira planilha
df = pd.read_excel('ALURA 2025  por semana inicio na semana 1 .xlsx')

print('=== ANÁLISE DA PLANILHA ALURA ===')
print('Forma dos dados:', df.shape)
print('\nPrimeiras linhas da coluna escola:')
print(df['1 - NOME DA ESCOLA'].head(10))

print('\nColunas com dados válidos (>5 valores):')
for col in df.columns:
    non_null_count = df[col].notna().sum()
    if non_null_count > 5:
        print(f'{col}: {df[col].dtype} - {non_null_count} valores válidos')

print('\nColunas relacionadas a ALURA ou BIM:')
alura_cols = [col for col in df.columns if 'ALURA' in str(col).upper() or 'BIM' in str(col).upper()]
for col in alura_cols:
    print(f'{col}:')
    valid_data = df[col].dropna()
    if len(valid_data) > 0:
        print(f'  Tipo: {valid_data.dtype}')
        print(f'  Primeiros valores: {valid_data.head().tolist()}')
    print()

# Tentar converter colunas de objeto para numérico
print('\nTentando converter colunas para numéricas:')
for col in alura_cols:
    if df[col].dtype == 'object':
        try:
            # Tentar converter para numérico
            df_converted = pd.to_numeric(df[col], errors='coerce')
            valid_count = df_converted.notna().sum()
            if valid_count > 0:
                print(f'{col}: {valid_count} valores numéricos válidos')
                print(f'  Valores: {df_converted.dropna().head().tolist()}')
        except:
            pass

print('\n=== DADOS DAS ESCOLAS ===')
escolas_validas = df[df['1 - NOME DA ESCOLA'].notna()]
print(f'Número de escolas: {len(escolas_validas)}')
print('Nomes das escolas:')
for escola in escolas_validas['1 - NOME DA ESCOLA'].tolist():
    print(f'  - {escola}')
