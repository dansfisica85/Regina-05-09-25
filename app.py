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
from typing import List, Dict, Any, Tuple

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = 'regina_analise_educacional_2025'  # Para sessions

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

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Sistema de análise educacional funcionando'})

if __name__ == '__main__':
    app.run(debug=True)
