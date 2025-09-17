# Sistema de Análise Educacional - Regina

Sistema web para análise automatizada de planilhas educacionais (ALURA, LEIA, SPeak).

## � Novo: Suporte SAEB (Aprendizagem)

Agora o sistema também processa automaticamente planilhas no modelo "APRENDIZAGEM - SAEB.xlsx" (formato consolidado com múltiplas linhas de cabeçalho e colunas sem nomes). O parser:

- Detecta o layout caracterizado por colunas "Unnamed" e cabeçalhos distribuídos em várias linhas (Área, Quinzena, Métrica).
- Extrai métricas de Engajamento e Acertos por Escola, Área (Língua Portuguesa / Matemática) e Períodos (Quinzenas).
- Normaliza valores percentuais mesmo com símbolos (+, -, =) e vírgulas.
- Calcula médias agregadas por escola para integrar ao fluxo existente sem modificar as funções originais.
- Gera um heatmap adicional (Engajamento / Acertos por Área) exibido automaticamente quando uma planilha SAEB é enviada.

### Como usar com SAEB

1. Inclua o arquivo no upload normal (não é necessário renomear).
2. O sistema identifica o formato e mostra mensagem: `(SAEB) processado`.
3. Um segundo gráfico (heatmap) aparecerá abaixo do gráfico de comparação geral.
4. O relatório Markdown inclui as médias agregadas por escola (o detalhamento período a período é utilizado apenas para gerar o heatmap e otimizar tamanho de sessão).

### Limitações atuais SAEB

- O relatório não lista cada quinzena separadamente (focado em média agregada por Escola/Área/Métrica).
- Se a estrutura for alterada (ex.: remoção de linhas de cabeçalho), a detecção pode falhar.

Se precisar de exportação detalhada (todas as quinzenas e métricas em tabela), abrir uma issue pedindo "Export detalhado SAEB".

## �🚀 Deploy no Vercel

### Pré-requisitos

- Conta no [Vercel](https://vercel.com)
- Vercel CLI instalado ou usar o dashboard web

### Opção 1: Deploy via Dashboard Vercel (Recomendado)

1. **Acesse o Vercel Dashboard**
   - Vá para https://vercel.com/dashboard
   - Faça login na sua conta

2. **Conectar Repositório**
   - Clique em "New Project"
   - Conecte com GitHub/GitLab/Bitbucket
   - Selecione o repositório "Regina-05-09-25"

3. **Configurações de Deploy**
   - Framework Preset: Selecione "Other"
   - Build and Output Settings: Deixe em branco (usar vercel.json)
   - Environment Variables: Não necessário para este projeto

4. **Deploy**
   - Clique em "Deploy"
   - O Vercel detectará automaticamente o `vercel.json` e fará o build

### Opção 2: Deploy via CLI

1. **Instalar Vercel CLI**
   ```bash
   npm i -g vercel
   ```

2. **Login no Vercel**
   ```bash
   vercel login
   ```

3. **Deploy do Projeto**
   ```bash
   # Na pasta do projeto
   vercel --prod
   ```

### 📁 Estrutura de Arquivos Necessários

O projeto já está configurado com:

- `app.py` - Aplicação Flask principal
- `requirements.txt` - Dependências Python
- `vercel.json` - Configuração do Vercel
- `templates/index.html` - Interface web

### 🔧 Funcionamento

1. **Upload de Planilhas**: Interface web para upload de múltiplos arquivos Excel/CSV
2. **Processamento Automático**: Identifica automaticamente colunas de escolas e dados
3. **Análise de Dados**: Calcula médias por escola e planilha
4. **Visualização**: Gera gráficos de comparação automaticamente
5. **Relatórios**: Tabelas detalhadas com estatísticas

### 📊 Recursos

- ✅ Suporte a Excel (.xlsx, .xls) e CSV
- ✅ Identificação automática de colunas
- ✅ Gráficos de comparação interativos
- ✅ Interface responsiva e moderna
- ✅ Processamento em tempo real
- ✅ Drag & drop para upload

### 🛠️ Tecnologias

- **Backend**: Flask (Python)
- **Frontend**: HTML, Tailwind CSS, JavaScript
- **Análise**: Pandas, NumPy, Matplotlib
- **Deploy**: Vercel

### 📝 Como Usar

1. Acesse a aplicação no Vercel
2. Faça upload das planilhas (.xlsx, .csv)
3. Aguarde o processamento
4. Visualize os resultados: gráficos, estatísticas e tabelas

### 🔍 Detecção Automática

O sistema identifica automaticamente:
- Colunas de escolas (nome, unidade, instituição)
- Colunas de dados (ALURA, BIM, médias, notas)
- Converte dados para formato numérico
- Calcula médias por escola

### 📞 Suporte

Em caso de problemas:
1. Verifique se os arquivos estão no formato correto
2. Confirme se as colunas têm nomes descritivos
3. Verifique se há dados numéricos válidos nas planilhas

### 🔄 Atualizações

Para atualizar o sistema:
1. Faça push das alterações para o repositório
2. O Vercel fará deploy automático das mudanças
