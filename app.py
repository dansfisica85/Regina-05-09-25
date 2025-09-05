from flask import Flask, request, render_template, jsonify, send_file, session
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64
import os
from werkzeug.utils import secure_filename
import tempfile
from datetime import datetime

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = 'regina-sistema-educacional-2025'  # Para usar sessões

# Configurar matplotlib para usar uma fonte que suporte caracteres especiais
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 8)

ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
    
    # Identificar colunas que contêm dados relevantes
    palavras_chave = ['alura', 'bim', 'média', 'media', 'nota', 'avaliação', 'resultado']
    colunas_dados = []
    
    for col in df.columns:
        col_lower = str(col).lower()
        if any(palavra in col_lower for palavra in palavras_chave):
            # Verificar se tem dados numéricos válidos
            try:
                dados_numericos = pd.to_numeric(df[col], errors='coerce')
                if dados_numericos.notna().sum() > 0:
                    colunas_dados.append(col)
            except:
                pass
    
    return coluna_escola, colunas_dados

def processar_planilha(df, nome_planilha):
    """
    Processa uma planilha e calcula as médias por escola
    """
    try:
        coluna_escola, colunas_dados = identificar_colunas_escola_e_dados(df)
        
        if not coluna_escola or not colunas_dados:
            return None, f"Não foi possível identificar colunas válidas em {nome_planilha}"
        
        # Filtrar apenas as linhas com escolas válidas
        df_limpo = df[df[coluna_escola].notna()].copy()
        
        if len(df_limpo) == 0:
            return None, f"Nenhuma escola válida encontrada em {nome_planilha}"
        
        resultados = []
        
        for idx, row in df_limpo.iterrows():
            escola = row[coluna_escola]
            valores = []
            
            for col in colunas_dados:
                try:
                    valor = pd.to_numeric(row[col], errors='coerce')
                    if not pd.isna(valor):
                        valores.append(valor)
                except:
                    pass
            
            if valores:
                media = np.mean(valores)
                resultados.append({
                    'Escola': escola,
                    'Planilha': nome_planilha,
                    'Media': media,
                    'Valores_Utilizados': len(valores)
                })
        
        return resultados, None
        
    except Exception as e:
        return None, f"Erro ao processar {nome_planilha}: {str(e)}"

def criar_grafico_comparacao(dados_combinados):
    """
    Cria gráfico de comparação das médias por escola
    """
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Preparar dados para o gráfico
    df_plot = pd.DataFrame(dados_combinados)
    
    # Agrupar por planilha e escola
    pivot_data = df_plot.pivot_table(index='Escola', columns='Planilha', values='Media', aggfunc='mean')
    
    # Criar gráfico de barras agrupadas
    pivot_data.plot(kind='bar', ax=ax, width=0.8)
    
    ax.set_title('Comparação de Médias por Escola e Planilha', fontsize=14, fontweight='bold')
    ax.set_xlabel('Escolas', fontsize=12)
    ax.set_ylabel('Média', fontsize=12)
    ax.legend(title='Planilhas', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Converter para base64
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return img_base64

def gerar_relatorio_markdown(dados_combinados, estatisticas, mensagens):
    """
    Gera um relatório em formato Markdown
    """
    df_dados = pd.DataFrame(dados_combinados)
    data_atual = datetime.now().strftime("%d/%m/%Y às %H:%M")
    
    # Cabeçalho do relatório
    md_content = f"""# Relatório de Análise Educacional

**Data de Geração:** {data_atual}

**Sistema:** Análise de Dados Educacionais - Região de Sertãozinho

---

## 📊 Resumo Executivo

Este relatório apresenta a análise detalhada das planilhas educacionais processadas, incluindo dados das plataformas ALURA, LEIA e SPeak.

### 📈 Estatísticas Gerais

"""
    
    # Adicionar estatísticas por planilha
    for planilha, stats in estatisticas.items():
        md_content += f"""
#### {planilha}
- **Escolas analisadas:** {stats['escolas']}
- **Média geral:** {stats['media']:.2f}
- **Menor média:** {stats['minimo']:.2f}
- **Maior média:** {stats['maximo']:.2f}

"""
    
    # Mensagens de processamento
    md_content += """
---

## 📝 Mensagens de Processamento

"""
    for msg in mensagens:
        md_content += f"- ✅ {msg}\n"
    
    # Dados detalhados por planilha
    md_content += """
---

## 📋 Dados Detalhados por Planilha

"""
    
    for planilha in df_dados['Planilha'].unique():
        dados_planilha = df_dados[df_dados['Planilha'] == planilha].sort_values('Media', ascending=False)
        
        md_content += f"""
### {planilha}

| Posição | Escola | Média | Valores Utilizados |
|---------|--------|-------|-------------------|
"""
        
        for idx, (_, row) in enumerate(dados_planilha.iterrows(), 1):
            md_content += f"| {idx}º | {row['Escola']} | {row['Media']:.2f} | {row['Valores_Utilizados']} |\n"
    
    # Análise comparativa
    md_content += """
---

## 📊 Análise Comparativa

### Ranking Geral por Média (Todas as Planilhas)

"""
    
    # Ranking geral
    ranking_geral = df_dados.groupby('Escola')['Media'].mean().sort_values(ascending=False)
    
    md_content += """
| Posição | Escola | Média Geral |
|---------|--------|-------------|
"""
    
    for idx, (escola, media) in enumerate(ranking_geral.items(), 1):
        md_content += f"| {idx}º | {escola} | {media:.2f} |\n"
    
    # Observações
    md_content += f"""
---

## 📋 Observações Técnicas

### Metodologia
- **Identificação automática** de colunas de escolas e dados numéricos
- **Cálculo de médias** baseado em valores válidos encontrados
- **Processamento de múltiplas planilhas** com consolidação de resultados

### Dados Processados
- **Total de registros:** {len(df_dados)}
- **Planilhas analisadas:** {len(df_dados['Planilha'].unique())}
- **Escolas únicas:** {len(df_dados['Escola'].unique())}

---

## 👥 Informações do Sistema

**Desenvolvido por:** PEC Tecnologia - Davi Antonino Nunes da Silva URESER

**Contato:** 
- 📧 E-mail: davi.silva@educacao.sp.gov.br
- 📱 Celular: 16 99260-4315

**Solicitado por:** Regina Aparecida Pieruchi  
**Cargo:** Chefe de Departamento – Dirigente Regional de Ensino de Sertãozinho

**Finalidade:** Análises detalhadas dos dados das Escolas

---

*Relatório gerado automaticamente pelo Sistema de Análise Educacional em {data_atual}*
"""
    
    return md_content

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'Nenhum arquivo enviado'}), 400
        
        files = request.files.getlist('files')
        
        if not files or all(file.filename == '' for file in files):
            return jsonify({'error': 'Nenhum arquivo selecionado'}), 400
        
        todos_resultados = []
        mensagens = []
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                
                # Ler arquivo diretamente da memória
                if filename.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file)
                else:
                    df = pd.read_csv(file)
                
                # Processar planilha
                resultados, erro = processar_planilha(df, filename)
                
                if erro:
                    mensagens.append(erro)
                else:
                    todos_resultados.extend(resultados)
                    mensagens.append(f"Arquivo {filename} processado com sucesso - {len(resultados)} escolas encontradas")
        
        if not todos_resultados:
            return jsonify({
                'error': 'Nenhum dado válido foi encontrado nos arquivos enviados',
                'mensagens': mensagens
            }), 400
        
        # Criar gráfico
        grafico_base64 = criar_grafico_comparacao(todos_resultados)
        
        # Calcular estatísticas
        df_resultados = pd.DataFrame(todos_resultados)
        estatisticas = {}
        
        for planilha in df_resultados['Planilha'].unique():
            dados = df_resultados[df_resultados['Planilha'] == planilha]
            estatisticas[planilha] = {
                'escolas': len(dados),
                'media': round(dados['Media'].mean(), 2),
                'minimo': round(dados['Media'].min(), 2),
                'maximo': round(dados['Media'].max(), 2)
            }
        
        # Salvar dados na sessão para download posterior
        session['dados_analise'] = {
            'dados': todos_resultados,
            'estatisticas': estatisticas,
            'mensagens': mensagens,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'mensagens': mensagens,
            'dados': todos_resultados,
            'grafico': grafico_base64,
            'estatisticas': estatisticas,
            'download_disponivel': True
        })
        
    except Exception as e:
        return jsonify({'error': f'Erro no processamento: {str(e)}'}), 500

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Sistema de análise educacional funcionando'})

if __name__ == '__main__':
    app.run(debug=True)
