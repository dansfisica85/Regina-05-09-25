from flask import Flask, request, render_template, jsonify, send_file, session
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import os
from werkzeug.utils import secure_filename
import tempfile
import time
import uuid

# Database connector
from database_connector import init_database, db_connector

# Imports condicionais para evitar conflitos
try:
    import seaborn as sns
except ImportError:
    sns = None
    print("Warning: seaborn não disponível")

# Imports do Smart Analytics
try:
    from smart_analytics import (
        DataTypeDetector, RelationshipDetector, 
        SmartChartRecommender, AutoInsightGenerator
    )
    from universal_processor import UniversalSpreadsheetProcessor
    from visualization_generator import AdvancedVisualizationGenerator
    from statistical_analyzer import AutomatedStatisticalAnalyzer
    SMART_ANALYTICS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Smart Analytics não disponível: {e}")
    SMART_ANALYTICS_AVAILABLE = False

# Imports para análise automática
try:
    from auto_structure_analyzer import AutoStructureAnalyzer, detect_file_structure
    from intelligent_chart_generator import IntelligentChartGenerator
    AUTO_ANALYSIS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Auto Analysis não disponível: {e}")
    AUTO_ANALYSIS_AVAILABLE = False
from datetime import datetime
import json
from typing import List, Dict, Any, Tuple

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = 'regina_analise_educacional_2025'  # Para sessions

# Inicializar banco de dados
db = init_database()
print("✅ Banco de dados Turso conectado e inicializado")

# Configurar matplotlib para usar uma fonte que suporte caracteres especiais
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 8)

ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ============================= SUPORTE SAEB (ACRESCIMO) ============================= #
def detectar_planilha_saeb(df: pd.DataFrame) -> bool:
    """Heurística simples para detectar se a planilha segue o padrão 'APRENDIZAGEM - SAEB'
    Observações da estrutura:
    - Colunas todas como 'Unnamed'
    - Linhas 1 a 3 contêm metadados: área (LÍNGUA PORTUGUESA / MATEMÁTICA), períodos (1ª QUINZENA ...), tipos (ENGAJAMENTO / ACERTOS)
    - A partir de certa linha começam os nomes de escola em uma coluna com muitos valores de texto e ao lado valores percentuais
    """
    # Critérios: muitas colunas 'Unnamed' e presença de palavras-chave em qualquer célula
    if not all(str(c).startswith('Unnamed') for c in df.columns):
        return False
    amostra_texto = ' '.join(str(v) for v in df.head(6).fillna('').values.flatten())
    chaves = ['LÍNGUA', 'PORTUGUESA', 'MATEMÁTICA', 'QUINZENA', 'ENGAJAMENTO', 'ACERTOS']
    return sum(1 for k in chaves if k.lower() in amostra_texto.lower()) >= 2

def normalizar_valor_percentual(v: Any) -> float | None:
    """Converte strings como '75,5 +' ou '70 -' em float (75.5, 70.0)."""
    if pd.isna(v):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    # Remover símbolos + - = e espaços finais
    s = s.replace('+', '').replace('-', '').replace('=', '').strip()
    # Trocar vírgula por ponto
    s = s.replace(',', '.')
    try:
        return float(s)
    except ValueError:
        return None

def extrair_bloco_saeb(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Extrai dados estruturados da planilha SAEB.
    Estratégia:
    - Identificar linhas 'header' múltiplas (linhas 1-3 observadas) para montar hierarquia: Área -> Período -> Métrica (ENGAJAMENTO/ACERTOS)
    - A partir da primeira linha onde aparece claramente um nome de escola (texto não genérico e valores numéricos ao lado) coletar registros.
    - Para cada escola, percorrer pares de colunas (Engajamento, Acertos) por período e área.
    Retorna lista de dicts com campos: Escola, Area, Periodo, Metrica, Valor
    """
    registros = []
    if df.empty:
        return registros

    # Copiar para evitar SettingWithCopy warnings
    dfx = df.copy()
    # Preencher NaN com vazio para manipulação textual
    dfx = dfx.fillna('')

    # Capturar possíveis linhas de cabeçalho (primeiras 4 linhas)
    header_rows = dfx.iloc[:4]
    header_matrix = header_rows.values
    n_cols = dfx.shape[1]

    # Construir metadados por coluna: Área, Período, Tipo
    areas = [''] * n_cols
    periodos = [''] * n_cols
    metricas = [''] * n_cols

    for c in range(n_cols):
        col_vals = [str(header_matrix[r][c]).strip() for r in range(len(header_rows))]
        for val in col_vals:
            vlow = val.lower()
            if 'língua p' in vlow or 'portuguesa' in vlow:
                areas[c] = 'Língua Portuguesa'
            elif 'matemática' in vlow:
                areas[c] = 'Matemática'
            if 'quinzena' in vlow:
                periodos[c] = val
            if 'engajamento' in vlow:
                metricas[c] = 'Engajamento'
            elif 'acerto' in vlow:
                metricas[c] = 'Acertos'

    # Propagar valores para a direita onde vazio (forward fill manual)
    for arr in (areas, periodos):
        ultimo = ''
        for i in range(n_cols):
            if arr[i]:
                ultimo = arr[i]
            else:
                arr[i] = ultimo

    # Encontrar linha inicial de escolas: heurística texto não vazio em alguma coluna e ao menos um valor percentual nas colunas seguintes
    linha_inicio = None
    for i in range(4, len(dfx)):
        row = dfx.iloc[i]
        textos = [str(v).strip() for v in row.values]
        # Candidato a nome de escola: string alfabética maior que 3 chars
        for j, txt in enumerate(textos):
            if len(txt) >= 4 and any(ch.isalpha() for ch in txt) and not txt.lower().startswith('unnamed'):
                # Verificar se existe algum número nas próximas colunas
                proximos = textos[j+1:j+6]
                if any(any(d.isdigit() for d in p) for p in proximos):
                    linha_inicio = i
                    break
        if linha_inicio is not None:
            break

    if linha_inicio is None:
        return registros

    # Assumir que a coluna j detectada é a coluna de escola -> encontrar melhor coluna escola: a com maior contagem de strings longas nas linhas de dados
    candidatos = {}
    dados_part = dfx.iloc[linha_inicio:]
    for c in range(n_cols):
        col_series = dados_part.iloc[:, c]
        score = 0
        for val in col_series.head(40):
            sval = str(val).strip()
            if len(sval) >= 4 and any(ch.isalpha() for ch in sval) and not sval.replace('.', '').isdigit():
                score += 1
        candidatos[c] = score
    col_escola = max(candidatos, key=candidatos.get)

    # Percorrer linhas de dados até encontrar linha vazia longa (mais de 80% vazia)
    for idx in range(linha_inicio, len(dfx)):
        row = dfx.iloc[idx]
        valores = row.values
        escola = str(valores[col_escola]).strip()
        if not escola or escola.lower().startswith('unnamed'):
            # Critério de parada: linha muito vazia
            vazios = sum(1 for v in valores if (not str(v).strip()))
            if vazios / n_cols > 0.8:
                break
            continue
        # Para cada coluna de métricas coletar valor
        for c in range(n_cols):
            if c == col_escola:
                continue
            metrica = metricas[c]
            if not metrica:
                continue
            area = areas[c] or 'Geral'
            periodo = periodos[c] or 'Período'
            valor_bruto = valores[c]
            valor = normalizar_valor_percentual(valor_bruto)
            if valor is not None:
                registros.append({
                    'Escola': escola,
                    'Area': area,
                    'Periodo': periodo,
                    'Metrica': metrica,
                    'Valor': valor
                })
    return registros

def processar_planilha_saeb(df: pd.DataFrame, nome_planilha: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str | None]:
    """Processa planilha SAEB retornando:
    - registros detalhados (lista granular)
    - agregados por escola (média geral de todas as métricas)
    - mensagem de status (ou None se ok)
    Não altera as funções originais do sistema.
    """
    if not detectar_planilha_saeb(df):
        return [], [], f"Planilha {nome_planilha} não reconhecida como formato SAEB"
    registros = extrair_bloco_saeb(df)
    if not registros:
        return [], [], f"Nenhum dado estruturado extraído de {nome_planilha} (SAEB)"
    # Agregar por escola para compatibilizar com fluxo existente
    df_reg = pd.DataFrame(registros)
    agregados = (
        df_reg.groupby('Escola')['Valor']
        .mean()
        .reset_index()
        .rename(columns={'Valor': 'Media'})
    )
    agregados['Planilha'] = nome_planilha
    agregados['Valores_Utilizados'] = df_reg.groupby('Escola')['Valor'].count().values
    return registros, agregados.to_dict(orient='records'), None
# ============================= FIM SUPORTE SAEB ===================================== #

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

def criar_grafico_saeb(saeb_registros: List[Dict[str, Any]]) -> str | None:
    """Cria heatmap de Engajamento/Acertos por Escola e Área agregando períodos.
    Retorna base64 ou None se dados insuficientes."""
    if not saeb_registros:
        return None
    df = pd.DataFrame(saeb_registros)
    if df.empty:
        return None
    # Agregar média por Escola, Area, Metrica
    pivot = (
        df.groupby(['Escola', 'Area', 'Metrica'])['Valor']
        .mean()
        .reset_index()
    )
    # Criar matriz multi-métrica concatenando Area+Metrica
    pivot['Chave'] = pivot['Area'] + ' - ' + pivot['Metrica']
    mat = pivot.pivot(index='Escola', columns='Chave', values='Valor')
    if mat.shape[0] < 2 or mat.shape[1] < 2:
        return None
    plt.figure(figsize=(max(8, mat.shape[1]*1.2), max(6, mat.shape[0]*0.4)))
    sns.heatmap(mat, annot=True, fmt='.1f', cmap='Blues', cbar_kws={'label': 'Valor Médio'})
    plt.title('SAEB - Engajamento e Acertos por Escola (Média dos Períodos)', fontsize=14, fontweight='bold')
    plt.xlabel('Indicadores')
    plt.ylabel('Escola')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    return b64

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
        
        saeb_registros_detalhados = []  # manter registros SAEB para relatório futuro
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                
                # Ler arquivo diretamente da memória
                if filename.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file)
                else:
                    df = pd.read_csv(file)
                
                # Primeiro tentar SAEB (adição)
                saeb_detalhes, saeb_agregado, erro_saeb = processar_planilha_saeb(df, filename)
                if saeb_agregado:  # se reconhecido como SAEB usar agregados
                    todos_resultados.extend(saeb_agregado)
                    saeb_registros_detalhados.extend(saeb_detalhes)
                    mensagens.append(f"Arquivo {filename} (SAEB) processado - {len(saeb_agregado)} escolas")
                else:
                    # Fluxo original
                    resultados, erro = processar_planilha(df, filename)
                    if erro:
                        # Se houve tentativa SAEB sem sucesso, juntar mensagens
                        if erro_saeb and 'SAEB' in erro_saeb:
                            mensagens.append(erro_saeb)
                        mensagens.append(erro)
                    else:
                        todos_resultados.extend(resultados)
                        mensagens.append(f"Arquivo {filename} processado com sucesso - {len(resultados)} escolas encontradas")
        
        if not todos_resultados:
            return jsonify({
                'error': 'Nenhum dado válido foi encontrado nos arquivos enviados',
                'mensagens': mensagens
            }), 400
        
        # Criar gráficos (usar registros em memória; só depois salvar na sessão)
        grafico_base64 = criar_grafico_comparacao(todos_resultados)
        grafico_saeb = criar_grafico_saeb(saeb_registros_detalhados) if saeb_registros_detalhados else None
        
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
        # Armazenar somente dados essenciais na sessão para não ultrapassar limite de cookie
        sess_dados = todos_resultados
        # Para SAEB guardar apenas contagem e não todos registros detalhados (que podem ser muitos)
        info_saeb = {
            'total_registros': len(saeb_registros_detalhados)
        } if saeb_registros_detalhados else {}
        session['dados_analise'] = {
            'dados': sess_dados,
            'estatisticas': estatisticas,
            'mensagens': mensagens,
            'timestamp': datetime.now().isoformat(),
            'saeb_info': info_saeb
        }
        
        retorno = {
            'success': True,
            'mensagens': mensagens,
            'dados': todos_resultados,
            'grafico': grafico_base64,
            'estatisticas': estatisticas
        }
        if grafico_saeb:
            retorno['grafico_saeb'] = grafico_saeb
        return jsonify(retorno)
        
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

# ============================= SMART ANALYTICS ROUTES ============================= #

# Verificar se Smart Analytics está disponível (imports já feitos no topo)
if not SMART_ANALYTICS_AVAILABLE:
    print("Smart Analytics não está disponível. Verifique as dependências.")

@app.route('/smart-analytics')
def smart_analytics_home():
    """Página inicial do Smart Analytics"""
    if not SMART_ANALYTICS_AVAILABLE:
        return render_template('error.html', 
                             message="Smart Analytics não disponível. Instale as dependências necessárias.")
    
    return render_template('smart_analytics.html')

@app.route('/smart-analytics/upload', methods=['POST'])
def smart_analytics_upload():
    """Upload e processamento inteligente de planilhas"""
    
    if not SMART_ANALYTICS_AVAILABLE:
        return jsonify({'error': 'Smart Analytics não disponível'}), 500
    
    start_time = time.time()
    session_id = session.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
    
    try:
        # Verifica se arquivo foi enviado
        if 'file' not in request.files:
            return jsonify({'error': 'Nenhum arquivo enviado'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Nenhum arquivo selecionado'}), 400
        
        filename = secure_filename(file.filename)
        file_type = filename.split('.')[-1].lower() if '.' in filename else 'unknown'
        
        # Log início do processamento
        db_connector.log_event('INFO', f'Iniciando processamento: {filename}', 'smart_analytics')
        
        # Processa arquivo com processador universal
        processor = UniversalSpreadsheetProcessor()
        result = processor.process_file(file.read(), file.filename)
        
        if not result['success']:
            db_connector.log_event('ERROR', f'Erro no processamento: {result["error"]}', 'smart_analytics')
            return jsonify({'error': result['error']}), 400
        
        df = result['data']
        metadata = result['metadata']
        
        # Limpeza inteligente dos dados
        from universal_processor import SmartDataCleaner
        cleaner = SmartDataCleaner()
        df_clean, cleaning_log = cleaner.clean_dataframe(df)
        
        # Detecção de tipos de dados
        detector = DataTypeDetector()
        column_types = {}
        for col in df_clean.columns:
            column_types[col] = detector.detect_column_type(df_clean[col])
        
        # Detecção de relacionamentos
        relationship_detector = RelationshipDetector()
        relationships = relationship_detector.find_relationships(df_clean)
        
        # Recomendação de gráficos
        chart_recommender = SmartChartRecommender()
        chart_recommendations = chart_recommender.recommend_charts(df_clean, column_types)
        
        # Geração de insights automáticos
        insight_generator = AutoInsightGenerator()
        insights = insight_generator.generate_insights(df_clean, column_types, relationships)
        
        # Análise estatística completa
        statistical_analyzer = AutomatedStatisticalAnalyzer()
        statistical_analysis = statistical_analyzer.comprehensive_analysis(df_clean, column_types)
        
        # Calcula métricas de performance
        processing_time = time.time() - start_time
        data_rows, data_columns = df_clean.shape
        charts_generated = len(chart_recommendations)
        insights_generated = len(insights)
        
        # Prepara dados para salvamento
        analysis_data = {
            'dataframe_summary': {
                'rows': data_rows,
                'columns': data_columns,
                'column_names': list(df_clean.columns)
            },
            'metadata': metadata,
            'cleaning_log': cleaning_log,
            'column_types': column_types,
            'relationships': relationships
        }
        
        charts_data = {
            'recommendations': chart_recommendations,
            'total_charts': charts_generated
        }
        
        # Salva análise no banco de dados
        analysis_id = db_connector.save_analysis(
            session_id=session_id,
            filename=filename,
            file_type=file_type,
            analysis_data=analysis_data,
            insights=insights,
            charts_data=charts_data,
            statistics=statistical_analysis
        )
        
        # Salva métricas de performance
        db_connector.save_performance_metrics(
            analysis_id=analysis_id,
            processing_time=processing_time,
            data_rows=data_rows,
            data_columns=data_columns,
            charts_generated=charts_generated,
            insights_generated=insights_generated
        )
        
        # Salva dados na sessão
        session['smart_analytics_data'] = {
            'analysis_id': analysis_id,
            'dataframe': df_clean.to_json(orient='records'),
            'metadata': metadata,
            'cleaning_log': cleaning_log,
            'column_types': column_types,
            'relationships': relationships,
            'chart_recommendations': chart_recommendations,
            'insights': insights,
            'statistical_analysis': statistical_analysis
        }
        
        # Log sucesso
        db_connector.log_event('INFO', f'Análise concluída com sucesso: {filename} (ID: {analysis_id})', 'smart_analytics')
        
        return jsonify({
            'success': True,
            'message': 'Arquivo processado com sucesso',
            'analysis_id': analysis_id,
            'processing_time': round(processing_time, 2),
            'metadata': metadata,
            'cleaning_log': cleaning_log,
            'column_types': column_types,
            'relationships': relationships,
            'chart_recommendations': chart_recommendations[:5],  # Primeiras 5 recomendações
            'insights': insights[:10],  # Primeiros 10 insights
            'statistical_summary': {
                'descriptive_stats': statistical_analysis.get('descriptive_stats', {}),
                'data_quality_score': sum(info.get('quality_score', 0) for info in column_types.values()) / len(column_types) if column_types else 0
            }
        })
        
    except Exception as e:
        db_connector.log_event('ERROR', f'Erro no processamento de {filename if "filename" in locals() else "arquivo"}: {str(e)}', 'smart_analytics')
        return jsonify({'error': f'Erro no processamento: {str(e)}'}), 500

@app.route('/smart-analytics/visualize', methods=['POST'])
def smart_analytics_visualize():
    """Gera visualização baseada na configuração"""
    
    if not SMART_ANALYTICS_AVAILABLE:
        return jsonify({'error': 'Smart Analytics não disponível'}), 500
    
    try:
        # Recupera dados da sessão
        if 'smart_analytics_data' not in session:
            return jsonify({'error': 'Nenhum dado carregado. Faça upload primeiro.'}), 400
        
        data = session['smart_analytics_data']
        df = pd.read_json(data['dataframe'], orient='records')
        
        # Configuração do gráfico
        chart_config = request.json
        
        # Gera visualização
        viz_generator = AdvancedVisualizationGenerator()
        result = viz_generator.generate_visualization(df, chart_config)
        
        if result['success']:
            return jsonify({
                'success': True,
                'figure_json': result['figure_json'],
                'chart_type': result['chart_type'],
                'title': result['title'],
                'description': result['description']
            })
        else:
            return jsonify({'error': result['error']}), 400
            
    except Exception as e:
        return jsonify({'error': f'Erro na visualização: {str(e)}'}), 500

@app.route('/smart-analytics/dashboard', methods=['POST'])
def smart_analytics_dashboard():
    """Gera dashboard completo com múltiplas visualizações"""
    
    if not SMART_ANALYTICS_AVAILABLE:
        return jsonify({'error': 'Smart Analytics não disponível'}), 500
    
    try:
        # Recupera dados da sessão
        if 'smart_analytics_data' not in session:
            return jsonify({'error': 'Nenhum dado carregado. Faça upload primeiro.'}), 400
        
        data = session['smart_analytics_data']
        df = pd.read_json(data['dataframe'], orient='records')
        recommendations = data['chart_recommendations']
        
        # Gera dashboard
        viz_generator = AdvancedVisualizationGenerator()
        result = viz_generator.generate_dashboard(df, recommendations)
        
        if result['success']:
            return jsonify({
                'success': True,
                'figure_json': result['figure_json'],
                'chart_count': result['chart_count'],
                'title': result['title']
            })
        else:
            return jsonify({'error': result['error']}), 400
            
    except Exception as e:
        return jsonify({'error': f'Erro no dashboard: {str(e)}'}), 500

@app.route('/smart-analytics/insights')
def smart_analytics_insights():
    """Retorna todos os insights gerados"""
    
    if not SMART_ANALYTICS_AVAILABLE:
        return jsonify({'error': 'Smart Analytics não disponível'}), 500
    
    try:
        if 'smart_analytics_data' not in session:
            return jsonify({'error': 'Nenhum dado carregado. Faça upload primeiro.'}), 400
        
        data = session['smart_analytics_data']
        
        return jsonify({
            'success': True,
            'insights': data['insights'],
            'statistical_analysis': data['statistical_analysis'],
            'relationships': data['relationships'],
            'column_types': data['column_types']
        })
        
    except Exception as e:
        return jsonify({'error': f'Erro ao recuperar insights: {str(e)}'}), 500

@app.route('/smart-analytics/export', methods=['POST'])
def smart_analytics_export():
    """Exporta análise completa como relatório"""
    
    if not SMART_ANALYTICS_AVAILABLE:
        return jsonify({'error': 'Smart Analytics não disponível'}), 500
    
    try:
        if 'smart_analytics_data' not in session:
            return jsonify({'error': 'Nenhum dado carregado. Faça upload primeiro.'}), 400
        
        data = session['smart_analytics_data']
        export_format = request.json.get('format', 'json')
        
        if export_format == 'json':
            # Exporta como JSON
            return jsonify({
                'success': True,
                'data': {
                    'metadata': data['metadata'],
                    'cleaning_log': data['cleaning_log'],
                    'column_types': data['column_types'],
                    'relationships': data['relationships'],
                    'insights': data['insights'],
                    'statistical_analysis': data['statistical_analysis'],
                    'chart_recommendations': data['chart_recommendations']
                }
            })
        
        elif export_format == 'markdown':
            # Gera relatório em Markdown
            report = generate_smart_analytics_report(data)
            
            # Salva arquivo temporário
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8')
            temp_file.write(report)
            temp_file.close()
            
            filename = f'smart_analytics_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
            
            return send_file(
                temp_file.name,
                as_attachment=True,
                download_name=filename,
                mimetype='text/markdown'
            )
        
        else:
            return jsonify({'error': 'Formato não suportado'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Erro na exportação: {str(e)}'}), 500

def generate_smart_analytics_report(data: Dict[str, Any]) -> str:
    """Gera relatório detalhado em Markdown"""
    
    report = f"""# Relatório de Smart Analytics
    
**Data de Geração:** {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}

## 📊 Metadados do Arquivo

- **Tipo de Fonte:** {data['metadata'].get('source_type', 'Desconhecido')}
- **Linhas:** {data['metadata'].get('rows', 0):,}
- **Colunas:** {data['metadata'].get('columns', 0)}

## 🧹 Log de Limpeza

"""
    
    # Adiciona informações de limpeza
    cleaning_ops = data['cleaning_log'].get('operations', [])
    if cleaning_ops:
        for op in cleaning_ops:
            report += f"- **{op.get('operation', 'Operação')}:** {op.get('description', '')}\n"
    else:
        report += "- Nenhuma operação de limpeza necessária\n"
    
    report += "\n## 📈 Tipos de Dados Detectados\n\n"
    
    # Tipos de colunas
    for col, info in data['column_types'].items():
        report += f"- **{col}:** {info['type']}"
        if info.get('subtype'):
            report += f" ({info['subtype']})"
        report += f" - Qualidade: {info.get('quality_score', 0):.2f}\n"
    
    report += "\n## 🔍 Insights Principais\n\n"
    
    # Top 5 insights
    insights = data['insights'][:5]
    for i, insight in enumerate(insights, 1):
        report += f"### {i}. {insight['title']}\n\n"
        report += f"{insight['description']}\n\n"
        if insight.get('recommendation'):
            report += f"**Recomendação:** {insight['recommendation']}\n\n"
    
    report += "\n## 📊 Relacionamentos Detectados\n\n"
    
    # Correlações
    correlations = data['relationships'].get('correlations', [])
    if correlations:
        report += "### Correlações Fortes\n\n"
        for corr in correlations[:5]:
            report += f"- **{corr['column1']} ↔ {corr['column2']}:** {corr['correlation']:.3f} ({corr['strength']})\n"
    
    # Clusters
    clusters = data['relationships'].get('clusters', [])
    if clusters:
        report += "\n### Grupos de Colunas Similares\n\n"
        for cluster in clusters:
            report += f"- **Grupo {cluster['cluster_id']}:** {', '.join(cluster['columns'])}\n"
    
    report += "\n## 📈 Análise Estatística\n\n"
    
    # Estatísticas descritivas
    desc_stats = data['statistical_analysis'].get('descriptive_stats', {})
    if 'numeric' in desc_stats:
        report += "### Estatísticas Numéricas\n\n"
        basic_stats = desc_stats['numeric'].get('basic_stats', {})
        for col, stats in list(basic_stats.items())[:3]:  # Primeiras 3 colunas
            report += f"**{col}:**\n"
            if isinstance(stats, dict):
                report += f"- Média: {stats.get('mean', 0):.2f}\n"
                report += f"- Mediana: {stats.get('50%', 0):.2f}\n"
                report += f"- Desvio Padrão: {stats.get('std', 0):.2f}\n\n"
    
    report += "\n## 🎯 Recomendações de Visualização\n\n"
    
    # Top 3 recomendações de gráficos
    chart_recs = data['chart_recommendations'][:3]
    for i, rec in enumerate(chart_recs, 1):
        report += f"{i}. **{rec['title']}** ({rec['chart_type']})\n"
        report += f"   - {rec['description']}\n"
        report += f"   - Relevância: {rec['relevance_score']:.2f}\n\n"
    
    report += f"\n---\n*Relatório gerado pelo Sistema Smart Analytics*"
    
    return report

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Sistema de análise educacional funcionando'})

# ============================= ROTAS DO BANCO DE DADOS ============================= #

@app.route('/api/dashboard')
def api_dashboard():
    """API para dados do dashboard"""
    try:
        stats = db_connector.get_system_stats()
        recent_analyses = db_connector.get_recent_analyses(10)
        
        return jsonify({
            'success': True,
            'stats': stats,
            'recent_analyses': recent_analyses,
            'database_status': 'connected'
        })
    except Exception as e:
        db_connector.log_event('ERROR', f'Erro na API dashboard: {str(e)}', 'api')
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analysis/<int:analysis_id>')
def api_get_analysis(analysis_id):
    """Busca análise específica por ID"""
    try:
        analysis = db_connector.get_analysis_by_id(analysis_id)
        if analysis:
            return jsonify({
                'success': True,
                'analysis': analysis
            })
        else:
            return jsonify({'success': False, 'error': 'Análise não encontrada'}), 404
    except Exception as e:
        db_connector.log_event('ERROR', f'Erro ao buscar análise {analysis_id}: {str(e)}', 'api')
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/database-status')
def database_status():
    """Página de status do banco de dados"""
    try:
        stats = db_connector.get_system_stats()
        recent_analyses = db_connector.get_recent_analyses(5)
        
        return render_template('database_status.html', 
                             stats=stats, 
                             recent_analyses=recent_analyses,
                             turso_url=os.getenv('TURSO_DATABASE_URL', 'N/A'))
    except Exception as e:
        return f"Erro ao acessar banco de dados: {str(e)}", 500

@app.route('/auto-analysis')
def auto_analysis_page():
    """Página de análise automática inteligente"""
    return render_template('auto_analysis.html')

@app.route('/auto-analysis', methods=['POST'])
def auto_analysis_process():
    """Processa dados automaticamente e gera gráficos inteligentes"""
    if not AUTO_ANALYSIS_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'Sistema de análise automática não disponível'
        }), 500
    
    session_id = session.get('session_id', str(uuid.uuid4()))
    session['session_id'] = session_id
    start_time = time.time()
    
    try:
        df = None
        filename = 'dados_colados'
        
        # Verifica se é upload de arquivo ou dados colados
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'Nenhum arquivo selecionado'}), 400
            
            filename = secure_filename(file.filename)
            
            # Processa arquivo baseado na extensão
            if filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(io.BytesIO(file.read()))
            elif filename.endswith('.csv'):
                # Tenta diferentes separadores para CSV
                file_content = file.read().decode('utf-8', errors='ignore')
                for sep in [',', ';', '\t', '|']:
                    try:
                        df = pd.read_csv(io.StringIO(file_content), sep=sep)
                        if df.shape[1] > 1:  # Se conseguiu separar em múltiplas colunas
                            break
                    except:
                        continue
                
                if df is None or df.shape[1] <= 1:
                    # Fallback para separador padrão
                    df = pd.read_csv(io.StringIO(file_content))
            
        elif request.is_json:
            # Dados colados
            data = request.get_json()
            paste_data = data.get('paste_data', '')
            
            if not paste_data.strip():
                return jsonify({'success': False, 'error': 'Nenhum dado fornecido'}), 400
            
            # Tenta interpretar dados colados como CSV
            for sep in [',', ';', '\t', '|']:
                try:
                    df = pd.read_csv(io.StringIO(paste_data), sep=sep)
                    if df.shape[1] > 1:
                        break
                except:
                    continue
            
            if df is None or df.shape[1] <= 1:
                # Tenta como linhas separadas por quebra de linha
                lines = paste_data.strip().split('\n')
                if len(lines) > 1:
                    # Primeira linha como cabeçalho
                    headers = lines[0].split()
                    data_rows = []
                    for line in lines[1:]:
                        row = line.split()
                        if len(row) == len(headers):
                            data_rows.append(row)
                    
                    if data_rows:
                        df = pd.DataFrame(data_rows, columns=headers)
        
        if df is None or df.empty:
            return jsonify({
                'success': False,
                'error': 'Não foi possível interpretar os dados. Verifique o formato.'
            }), 400
        
        # Remove colunas completamente vazias
        df = df.dropna(axis=1, how='all')
        
        # Se ainda estiver vazio
        if df.empty:
            return jsonify({
                'success': False,
                'error': 'Dados estão vazios após limpeza'
            }), 400
        
        # Análise automática da estrutura
        analyzer = AutoStructureAnalyzer()
        analysis_results = analyzer.analyze_dataframe(df)
        
        # Gera gráficos inteligentes
        chart_generator = IntelligentChartGenerator()
        charts = chart_generator.generate_all_charts(df, analysis_results)
        
        # Calcula tempo de processamento
        processing_time = time.time() - start_time
        
        # Salva no banco de dados
        try:
            analysis_data = {
                'file_name': filename,
                'data_shape': analysis_results['data_shape'],
                'column_types': {col: info['type'] for col, info in analysis_results['column_analysis'].items()},
                'data_quality_score': analysis_results['data_quality']['completeness_score'],
                'charts_generated': len(charts),
                'processing_time': processing_time,
                'insights_count': len(analysis_results.get('patterns', [])),
                'session_id': session_id
            }
            
            analysis_id = db_connector.save_analysis(
                session_id=session_id,
                file_name=filename,
                analysis_type='auto_analysis',
                results=analysis_data,
                processing_time=processing_time
            )
            
            db_connector.log_event('SUCCESS', 
                f'Análise automática concluída: {len(charts)} gráficos gerados', 
                'auto_analysis')
                
        except Exception as db_error:
            print(f"Erro ao salvar no banco: {db_error}")
            # Continua sem falhar se houver erro no banco
        
        return jsonify({
            'success': True,
            'analysis': analysis_results,
            'charts': charts,
            'processing_time': processing_time,
            'message': f'Análise concluída! {len(charts)} gráficos gerados automaticamente.'
        })
        
    except Exception as e:
        db_connector.log_event('ERROR', f'Erro na análise automática: {str(e)}', 'auto_analysis')
        return jsonify({
            'success': False,
            'error': f'Erro durante análise: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Inicializar configurações do sistema
    db_connector.set_config('system_version', '1.0.0')
    db_connector.set_config('smart_analytics_enabled', True)
    db_connector.set_config('auto_analysis_enabled', AUTO_ANALYSIS_AVAILABLE)
    
    print("🚀 Iniciando Regina Smart Analytics...")
    print(f"📊 Banco de dados: {'Conectado' if db_connector else 'Erro'}")
    print(f"🧠 Smart Analytics: {'Ativo' if SMART_ANALYTICS_AVAILABLE else 'Inativo'}")
    print(f"🤖 Auto Analysis: {'Ativo' if AUTO_ANALYSIS_AVAILABLE else 'Inativo'}")
    
    print("🌐 Servidor iniciando em http://localhost:5001")
    print("📈 Acesse /auto-analysis para análise automática de planilhas")
    app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)
