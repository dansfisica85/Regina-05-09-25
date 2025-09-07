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
import json

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = 'regina_analise_educacional_2025'  # Para sessions

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
    Gera um relatório completo em formato Markdown
    """
    df_resultados = pd.DataFrame(dados_combinados)
    
    # Cabeçalho
    md_content = f"""# Relatório de Análise Educacional

**Data da Análise:** {datetime.now().strftime('%d/%m/%Y às %H:%M')}

---

## 📊 Resumo Executivo

Este relatório apresenta a análise detalhada dos dados educacionais processados pelo Sistema de Análise Educacional, desenvolvido para a Dirigente Regional de Ensino de Sertãozinho, Regina Aparecida Pieruchi.

### 📈 Estatísticas Gerais

- **Total de Escolas Analisadas:** {len(df_resultados)}
- **Planilhas Processadas:** {len(df_resultados['Planilha'].unique())}
- **Média Geral:** {df_resultados['Media'].mean():.2f}

---

## 📋 Mensagens do Processamento

"""
    
    for i, msg in enumerate(mensagens, 1):
        md_content += f"{i}. {msg}\n"
    
    md_content += "\n---\n\n## 📊 Estatísticas por Planilha\n\n"
    
    # Estatísticas detalhadas por planilha
    for planilha, stats in estatisticas.items():
        md_content += f"""### 📄 {planilha}

- **Escolas:** {stats['escolas']}
- **Média:** {stats['media']}
- **Valor Mínimo:** {stats['minimo']}
- **Valor Máximo:** {stats['maximo']}

"""
    
    md_content += "---\n\n## 📋 Dados Detalhados por Escola\n\n"
    
    # Tabela detalhada
    md_content += "| Escola | Planilha | Média | Valores Utilizados |\n"
    md_content += "|--------|----------|-------|--------------------|\n"
    
    for item in dados_combinados:
        escola = item['Escola'].replace('|', '\\|')  # Escapar pipes na tabela
        planilha = item['Planilha'].replace('|', '\\|')
        md_content += f"| {escola} | {planilha} | {item['Media']:.2f} | {item['Valores_Utilizados']} |\n"
    
    md_content += "\n---\n\n"
    
    # Análise por planilha
    for planilha in df_resultados['Planilha'].unique():
        dados_planilha = df_resultados[df_resultados['Planilha'] == planilha]
        md_content += f"""## 🏫 Análise Detalhada - {planilha}

### Ranking das Escolas (por Média)

"""
        # Ordenar por média decrescente
        dados_ordenados = dados_planilha.sort_values('Media', ascending=False)
        
        for idx, (_, row) in enumerate(dados_ordenados.iterrows(), 1):
            emoji = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else "📊"
            md_content += f"{idx}. {emoji} **{row['Escola']}** - Média: {row['Media']:.2f}\n"
        
        md_content += f"""
### Estatísticas da Planilha

- **Média da Planilha:** {dados_planilha['Media'].mean():.2f}
- **Desvio Padrão:** {dados_planilha['Media'].std():.2f}
- **Amplitude:** {dados_planilha['Media'].max() - dados_planilha['Media'].min():.2f}

---

"""
    
    # Rodapé
    md_content += f"""## 👥 Créditos

**Sistema Desenvolvido por:**
- **PEC Tecnologia**
- **Davi Antonino Nunes da Silva URESER**
- **E-mail:** davi.silva@educacao.sp.gov.br
- **Celular:** 16 99260-4315

**Solicitado por:**
- **Regina Aparecida Pieruchi**
- **Chefe de Departamento – Dirigente Regional de Ensino de Sertãozinho**

---

*Relatório gerado automaticamente em {datetime.now().strftime('%d/%m/%Y às %H:%M')}*
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
            'estatisticas': estatisticas
        })
        
    except Exception as e:
        return jsonify({'error': f'Erro no processamento: {str(e)}'}), 500

@app.route('/download_relatorio')
def download_relatorio():
    """
    Gera e faz download do relatório em formato Markdown
    """
    try:
        if 'dados_analise' not in session:
            return jsonify({'error': 'Nenhuma análise disponível para download. Faça uma nova análise primeiro.'}), 400
        
        dados_sessao = session['dados_analise']
        
        # Gerar relatório em Markdown
        md_content = gerar_relatorio_markdown(
            dados_sessao['dados'],
            dados_sessao['estatisticas'],
            dados_sessao['mensagens']
        )
        
        # Criar arquivo temporário
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8')
        temp_file.write(md_content)
        temp_file.close()
        
        # Nome do arquivo com timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'relatorio_analise_educacional_{timestamp}.md'
        
        def remove_file(response):
            try:
                os.unlink(temp_file.name)
            except Exception:
                pass
            return response
        
        return send_file(
            temp_file.name,
            as_attachment=True,
            download_name=filename,
            mimetype='text/markdown'
        )
        
    except Exception as e:
        return jsonify({'error': f'Erro ao gerar relatório: {str(e)}'}), 500

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Sistema de análise educacional funcionando'})

if __name__ == '__main__':
    app.run(debug=True)
