import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import numpy as np

# Configurar matplotlib para usar uma fonte que suporte caracteres especiais
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 8)

def converter_excel_para_csv(arquivo_excel):
    """
    Converte arquivo Excel para CSV
    """
    try:
        # Ler o arquivo Excel
        df = pd.read_excel(arquivo_excel)
        
        # Criar nome do arquivo CSV
        nome_csv = arquivo_excel.replace('.xlsx', '.csv').replace('.xls', '.csv')
        
        # Salvar como CSV
        df.to_csv(nome_csv, index=False, encoding='utf-8-sig')
        print(f"Arquivo convertido: {arquivo_excel} -> {nome_csv}")
        
        return nome_csv, df
    except Exception as e:
        print(f"Erro ao converter {arquivo_excel}: {str(e)}")
        return None, None

def identificar_colunas_escola_e_dados(df):
    """
    Identifica automaticamente as colunas de escola e dados numéricos
    """
    # Procurar colunas que podem ser escolas (strings com "escola", "unidade", etc.)
    colunas_texto = df.select_dtypes(include=['object']).columns
    coluna_escola = None
    
    for col in colunas_texto:
        col_lower = str(col).lower()
        if any(palavra in col_lower for palavra in ['escola', 'unidade', 'instituição', 'nome']):
            coluna_escola = col
            break
    
    # Se não encontrar, usar a primeira coluna de texto
    if coluna_escola is None and len(colunas_texto) > 0:
        coluna_escola = colunas_texto[0]
    
    # Identificar colunas que contêm dados relevantes (ALURA, BIM, etc.)
    colunas_dados = []
    for col in df.columns:
        col_upper = str(col).upper()
        if any(palavra in col_upper for palavra in ['ALURA', 'BIM', 'LEIA', 'SPEAK', 'PLATAFORMA']):
            # Verificar se a coluna tem dados válidos
            if df[col].notna().sum() > 5:  # Pelo menos 5 valores válidos
                colunas_dados.append(col)
    
    return coluna_escola, colunas_dados

def calcular_media_por_escola(df, nome_planilha):
    """
    Calcula a média por escola dos dados numéricos
    """
    # Identificar colunas automaticamente
    coluna_escola, colunas_dados = identificar_colunas_escola_e_dados(df)
    
    if coluna_escola is None:
        print(f"Não foi possível identificar a coluna de escola em {nome_planilha}")
        return None
    
    if not colunas_dados:
        print(f"Não foram encontradas colunas de dados relevantes em {nome_planilha}")
        return None
    
    print(f"\n{nome_planilha}:")
    print(f"Coluna escola: {coluna_escola}")
    print(f"Colunas de dados: {colunas_dados}")
    
    # Filtrar apenas linhas com nomes de escola válidos
    df_limpo = df[df[coluna_escola].notna() & (df[coluna_escola] != '')].copy()
    
    if len(df_limpo) == 0:
        print(f"Nenhuma escola encontrada em {nome_planilha}")
        return None
    
    # Calcular média por escola
    try:
        medias_por_escola = []
        
        for escola in df_limpo[coluna_escola].unique():
            dados_escola = df_limpo[df_limpo[coluna_escola] == escola]
            valores_numericos = []
            
            for col in colunas_dados:
                for valor in dados_escola[col]:
                    if pd.notna(valor):
                        try:
                            # Tentar converter para float
                            valor_num = pd.to_numeric(valor, errors='coerce')
                            if pd.notna(valor_num):
                                valores_numericos.append(valor_num)
                        except:
                            continue
            
            if valores_numericos:
                media = np.mean(valores_numericos)
                medias_por_escola.append({
                    'Escola': escola,
                    'Media': media,
                    'Planilha': nome_planilha,
                    'Num_Valores': len(valores_numericos)
                })
        
        if medias_por_escola:
            resultado = pd.DataFrame(medias_por_escola)
            print(f"Médias calculadas para {len(resultado)} escolas")
            return resultado
        else:
            print(f"Nenhuma média foi calculada para {nome_planilha}")
            return None
    
    except Exception as e:
        print(f"Erro ao calcular média para {nome_planilha}: {str(e)}")
        return None

def main():
    # Lista dos arquivos Excel
    arquivos_excel = [
        'ALURA 2025  por semana inicio na semana 1 .xlsx',
        'LEIA 2025 inicio na semana 1  OK.xlsx',
        'plataforma SPeak 2025 por semana até semana 43.xlsx'
    ]
    
    print("=== PROCESSAMENTO DE PLANILHAS ===\n")
    
    # Verificar se os arquivos existem
    arquivos_existentes = []
    for arquivo in arquivos_excel:
        if os.path.exists(arquivo):
            arquivos_existentes.append(arquivo)
            print(f"✓ Arquivo encontrado: {arquivo}")
        else:
            print(f"✗ Arquivo não encontrado: {arquivo}")
    
    if not arquivos_existentes:
        print("Nenhum arquivo foi encontrado!")
        return
    
    print(f"\n=== CONVERSÃO PARA CSV ===")
    
    # Converter arquivos para CSV e processar dados
    todas_medias = []
    dados_originais = {}
    
    for arquivo in arquivos_existentes:
        print(f"\nProcessando: {arquivo}")
        
        # Converter para CSV
        arquivo_csv, df = converter_excel_para_csv(arquivo)
        
        if df is not None:
            # Armazenar dados originais
            dados_originais[arquivo] = df
            
            # Calcular média por escola
            nome_planilha = arquivo.replace('.xlsx', '').replace('.xls', '')
            media_escola = calcular_media_por_escola(df, nome_planilha)
            
            if media_escola is not None:
                todas_medias.append(media_escola)
    
    if not todas_medias:
        print("Nenhuma média foi calculada!")
        return
    
    # Combinar todas as médias
    df_medias = pd.concat(todas_medias, ignore_index=True)
    
    print(f"\n=== RESUMO DAS MÉDIAS ===")
    print(df_medias.to_string(index=False))
    
    # Salvar médias em CSV
    df_medias.to_csv('medias_por_escola_planilha.csv', index=False, encoding='utf-8-sig')
    print(f"\nMédias salvas em: medias_por_escola_planilha.csv")
    
    # Criar gráficos
    print(f"\n=== CRIANDO GRÁFICOS ===")
    
    # Verificar se há dados válidos para plotar
    if df_medias['Media'].isna().all():
        print("Erro: Todas as médias são NaN. Não é possível criar gráficos.")
        return
    
    # Remover linhas com NaN
    df_medias_limpo = df_medias.dropna(subset=['Media'])
    
    if len(df_medias_limpo) == 0:
        print("Erro: Nenhum dado válido para plotar.")
        return
    
    print(f"Dados válidos para plotar: {len(df_medias_limpo)} registros")
    
    # Gráfico 1: Médias por escola (todas as planilhas)
    plt.figure(figsize=(15, 10))
    
    # Preparar dados para o gráfico
    pivot_data = df_medias_limpo.pivot(index='Escola', columns='Planilha', values='Media')
    
    # Gráfico de barras agrupadas
    ax = pivot_data.plot(kind='bar', figsize=(15, 10))
    plt.title('Médias por Escola - Comparação entre Planilhas', fontsize=16, fontweight='bold')
    plt.xlabel('Escola', fontsize=12)
    plt.ylabel('Média', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Planilhas', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)
    
    # Salvar gráfico
    plt.savefig('medias_por_escola_comparacao.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Gráfico 2: Heatmap das médias (apenas se há dados suficientes)
    if len(pivot_data.columns) > 1 and len(pivot_data.index) > 1:
        plt.figure(figsize=(12, 8))
        
        # Criar heatmap
        sns.heatmap(pivot_data.T, annot=True, cmap='YlOrRd', fmt='.2f', cbar_kws={'label': 'Média'})
        plt.title('Heatmap das Médias por Escola e Planilha', fontsize=16, fontweight='bold')
        plt.xlabel('Escola', fontsize=12)
        plt.ylabel('Planilha', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Salvar heatmap
        plt.savefig('heatmap_medias_escolas.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Gráfico 3: Médias por planilha (boxplot) - apenas se há múltiplas planilhas
    if len(df_medias_limpo['Planilha'].unique()) > 1:
        plt.figure(figsize=(12, 8))
        
        # Boxplot das médias por planilha
        sns.boxplot(data=df_medias_limpo, x='Planilha', y='Media')
        plt.title('Distribuição das Médias por Planilha', fontsize=16, fontweight='bold')
        plt.xlabel('Planilha', fontsize=12)
        plt.ylabel('Média', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.grid(axis='y', alpha=0.3)
        
        # Salvar boxplot
        plt.savefig('distribuicao_medias_planilhas.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Gráfico 4: Gráfico de barras simples por planilha
    for planilha in df_medias_limpo['Planilha'].unique():
        dados_planilha = df_medias_limpo[df_medias_limpo['Planilha'] == planilha]
        
        plt.figure(figsize=(15, 8))
        plt.bar(range(len(dados_planilha)), dados_planilha['Media'])
        plt.title(f'Médias por Escola - {planilha}', fontsize=16, fontweight='bold')
        plt.xlabel('Escola', fontsize=12)
        plt.ylabel('Média', fontsize=12)
        plt.xticks(range(len(dados_planilha)), dados_planilha['Escola'], rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Salvar gráfico individual
        nome_arquivo = f'medias_{planilha.replace(" ", "_").replace(".", "")}.png'
        plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
        plt.show()
    
    # Estatísticas resumidas
    print(f"\n=== ESTATÍSTICAS RESUMIDAS ===")
    
    if len(df_medias_limpo) > 0:
        for planilha in df_medias_limpo['Planilha'].unique():
            dados_planilha = df_medias_limpo[df_medias_limpo['Planilha'] == planilha]
            print(f"\n{planilha}:")
            print(f"  Número de escolas: {len(dados_planilha)}")
            print(f"  Média geral: {dados_planilha['Media'].mean():.2f}")
            print(f"  Desvio padrão: {dados_planilha['Media'].std():.2f}")
            print(f"  Menor média: {dados_planilha['Media'].min():.2f}")
            print(f"  Maior média: {dados_planilha['Media'].max():.2f}")
    
    print(f"\n=== PROCESSAMENTO CONCLUÍDO ===")
    print("Arquivos gerados:")
    print("- Arquivos CSV convertidos")
    print("- medias_por_escola_planilha.csv")
    
    if len(df_medias_limpo) > 0:
        print("- medias_por_escola_comparacao.png")
        if len(df_medias_limpo['Planilha'].unique()) > 1:
            print("- heatmap_medias_escolas.png")
            print("- distribuicao_medias_planilhas.png")
        for planilha in df_medias_limpo['Planilha'].unique():
            nome_arquivo = f'medias_{planilha.replace(" ", "_").replace(".", "")}.png'
            print(f"- {nome_arquivo}")
    else:
        print("- Nenhum gráfico foi gerado devido à falta de dados válidos")

if __name__ == "__main__":
    main()
