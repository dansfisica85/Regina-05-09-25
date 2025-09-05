# ⚡ Como Fazer Deploy no Vercel - Guia Passo a Passo

## 🎯 Seu projeto está pronto para o Vercel!

### ✅ Arquivos Configurados:
- `app.py` - Aplicação web principal
- `requirements.txt` - Dependências Python
- `vercel.json` - Configuração do Vercel
- `templates/index.html` - Interface web moderna

---

## 🚀 Deploy no Vercel (3 maneiras)

### 🌐 Método 1: Dashboard Web (Mais Fácil)

1. **Acesse**: https://vercel.com/dashboard
2. **Novo Projeto**: Clique em "New Project"
3. **Conectar GitHub**: Autorize o Vercel a acessar seus repositórios
4. **Selecionar Repo**: Escolha "Regina-05-09-25"
5. **Deploy**: Clique em "Deploy" (as configurações já estão no `vercel.json`)

### 💻 Método 2: Vercel CLI

```bash
# 1. Instalar Vercel CLI
npm i -g vercel

# 2. Login
vercel login

# 3. Deploy (na pasta do projeto)
vercel --prod
```

### 🔄 Método 3: Git Push Automático

1. **Primeira vez**: Use Método 1 ou 2
2. **Próximas atualizações**: Apenas faça `git push`
3. **Auto-deploy**: Vercel automaticamente faz novo deploy

---

## 🛡️ Verificação de Deploy

1. **URL do projeto**: Vercel fornece uma URL como `https://regina-05-09-25.vercel.app`
2. **Teste**: Acesse a URL e faça upload de uma planilha
3. **Health Check**: Acesse `/health` para verificar se está funcionando

---

## 📊 Funcionalidades do Sistema

### 🎯 O que o sistema faz:
- ✅ Upload de múltiplas planilhas (Excel/CSV)
- ✅ Detecção automática de colunas de escolas
- ✅ Identificação automática de dados (ALURA, BIM, etc.)
- ✅ Cálculo de médias por escola
- ✅ Gráficos comparativos automáticos
- ✅ Interface moderna e responsiva
- ✅ Drag & drop para upload

### 📁 Tipos de arquivo suportados:
- `.xlsx` (Excel)
- `.xls` (Excel legado)
- `.csv` (Comma Separated Values)

---

## 🔧 Configuração Técnica

### 📋 Estrutura do `vercel.json`:
```json
{
  "version": 2,
  "builds": [
    {
      "src": "app.py",
      "use": "@vercel/python",
      "config": {
        "maxLambdaSize": "15mb"
      }
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "app.py"
    }
  ]
}
```

### 📦 Dependências (`requirements.txt`):
```
Flask>=2.3.0
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
openpyxl>=3.0.0
Werkzeug>=2.3.0
```

---

## 🌐 URLs do Sistema

Após o deploy, você terá:

- **🏠 Página Principal**: `https://seu-projeto.vercel.app/`
- **📤 Upload**: `https://seu-projeto.vercel.app/upload` (POST)
- **💚 Health Check**: `https://seu-projeto.vercel.app/health`

---

## 🎨 Interface Web

A interface inclui:

1. **📤 Área de Upload**:
   - Drag & drop
   - Seleção múltipla de arquivos
   - Validação de tipos

2. **📊 Resultados**:
   - Gráficos de comparação
   - Estatísticas resumidas
   - Tabela de dados detalhada

3. **🎯 Design Responsivo**:
   - Funciona em desktop e mobile
   - Interface moderna com Tailwind CSS

---

## 🐛 Solução de Problemas

### ❌ Deploy falhou?
1. Verifique se `vercel.json` está na raiz
2. Confirme se `app.py` e `requirements.txt` estão corretos
3. Verifique logs no dashboard do Vercel

### ⚠️ Erro no processamento?
1. Verifique se as planilhas têm colunas com nomes descritivos
2. Confirme se há dados numéricos válidos
3. Teste com planilhas menores primeiro

### 🔧 Limite de tamanho?
- Arquivos: máximo 16MB por upload
- Lambda: configurado para 15MB

---

## 📞 Próximos Passos

1. **✅ Faça o deploy no Vercel**
2. **🧪 Teste com suas planilhas reais**
3. **🔄 Compartilhe a URL com sua equipe**
4. **📈 Monitore o uso no dashboard Vercel**

---

**🎉 Seu sistema está pronto para produção no Vercel!**
