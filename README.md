# Sistema de Análise Educacional - Regina

Sistema web para análise automatizada de planilhas educacionais (ALURA, LEIA, SPeak).

## 🚀 Deploy no Vercel

### Pré-requisitos
- Conta no Vercel (https://vercel.com)
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
