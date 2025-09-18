"""
Análise Regina - Smart | Analisador Automático de Estrutura de Dados
Sistema inteligente que detecta automaticamente a estrutura de qualquer planilha
e gera visualizações apropriadas baseadas nos tipos de dados encontrados.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import seaborn as sns
import json

def convert_numpy_types(obj):
    """Converte tipos NumPy para tipos Python nativos para serialização JSON"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif pd.isna(obj):
        return None
    return obj

class AutoStructureAnalyzer:
    """
    Analisador automático de estrutura de dados que detecta:
    - Tipos de colunas (numérico, categórico, data, texto)
    - Relações entre variáveis
    - Padrões nos dados
    - Melhores visualizações para cada tipo de dado
    """
    
    def __init__(self):
        self.column_types = {}
        self.data_insights = {}
        self.recommended_charts = []
        
    def analyze_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analisa completamente um DataFrame e retorna insights estruturais
        """
        results = {
            'data_shape': df.shape,
            'column_analysis': self._analyze_columns(df),
            'data_quality': self._analyze_data_quality(df),
            'relationships': self._analyze_relationships(df),
            'recommended_visualizations': self._recommend_visualizations(df),
            'summary_stats': self._generate_summary_stats(df),
            'patterns': self._detect_patterns(df)
        }
        
        # Converter todos os tipos NumPy para tipos Python nativos
        return convert_numpy_types(results)
    
    def _analyze_columns(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Analisa cada coluna individualmente e determina seu tipo e características
        """
        column_analysis = {}
        
        for col in df.columns:
            series = df[col].dropna()
            
            analysis = {
                'name': col,
                'type': self._detect_column_type(series),
                'null_count': df[col].isnull().sum(),
                'null_percentage': (df[col].isnull().sum() / len(df)) * 100,
                'unique_count': series.nunique(),
                'unique_percentage': (series.nunique() / len(series)) * 100 if len(series) > 0 else 0,
                'sample_values': series.head(5).tolist() if len(series) > 0 else [],
                'description': self._describe_column(series)
            }
            
            # Adiciona estatísticas específicas por tipo
            if analysis['type'] == 'numeric':
                analysis.update(self._analyze_numeric_column(series))
            elif analysis['type'] == 'categorical':
                analysis.update(self._analyze_categorical_column(series))
            elif analysis['type'] == 'datetime':
                analysis.update(self._analyze_datetime_column(series))
            
            column_analysis[col] = analysis
            
        return column_analysis
    
    def _detect_column_type(self, series: pd.Series) -> str:
        """
        Detecta automaticamente o tipo de uma coluna
        """
        # Remove valores nulos para análise
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return 'empty'
        
        # Tenta detectar datas primeiro
        if self._is_datetime_column(clean_series):
            return 'datetime'
        
        # Verifica se é numérico
        if self._is_numeric_column(clean_series):
            return 'numeric'
        
        # Verifica se é categórico (poucas categorias únicas)
        unique_ratio = clean_series.nunique() / len(clean_series)
        if unique_ratio < 0.5 and clean_series.nunique() < 50:
            return 'categorical'
        
        # Verifica se contém principalmente texto
        if self._is_text_column(clean_series):
            return 'text'
        
        # Default para categórico se não conseguir determinar
        return 'categorical'
    
    def _is_datetime_column(self, series: pd.Series) -> bool:
        """
        Verifica se uma coluna contém datas
        """
        # Padrões comuns de data
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # DD/MM/YYYY ou MM/DD/YYYY
            r'\d{2,4}-\d{1,2}-\d{1,2}',   # YYYY-MM-DD
            r'\d{1,2}-\d{1,2}-\d{2,4}',   # DD-MM-YYYY
            r'\w+ \d{1,2}, \d{4}',        # Month DD, YYYY
        ]
        
        sample_size = min(100, len(series))
        sample = series.head(sample_size).astype(str)
        
        # Verifica se pelo menos 70% dos valores seguem padrões de data
        matches = 0
        for value in sample:
            for pattern in date_patterns:
                if re.match(pattern, value.strip()):
                    matches += 1
                    break
        
        return matches / sample_size > 0.7
    
    def _is_numeric_column(self, series: pd.Series) -> bool:
        """
        Verifica se uma coluna é numérica
        """
        try:
            pd.to_numeric(series, errors='raise')
            return True
        except:
            # Tenta converter removendo caracteres não numéricos
            try:
                cleaned = series.astype(str).str.replace(r'[^\d.,\-+]', '', regex=True)
                cleaned = cleaned.str.replace(',', '.', regex=False)
                pd.to_numeric(cleaned, errors='raise')
                return True
            except:
                return False
    
    def _is_text_column(self, series: pd.Series) -> bool:
        """
        Verifica se uma coluna contém principalmente texto
        """
        sample_size = min(100, len(series))
        sample = series.head(sample_size).astype(str)
        
        # Verifica se os valores têm mais de 3 caracteres em média
        avg_length = sample.str.len().mean()
        return avg_length > 3
    
    def _analyze_numeric_column(self, series: pd.Series) -> Dict:
        """
        Análise específica para colunas numéricas
        """
        numeric_series = pd.to_numeric(series, errors='coerce').dropna()
        
        if len(numeric_series) == 0:
            return {}
        
        return {
            'min': float(numeric_series.min()),
            'max': float(numeric_series.max()),
            'mean': float(numeric_series.mean()),
            'median': float(numeric_series.median()),
            'std': float(numeric_series.std()),
            'outliers_count': int(self._count_outliers(numeric_series)),
            'distribution_type': self._detect_distribution(numeric_series)
        }
    
    def _analyze_categorical_column(self, series: pd.Series) -> Dict:
        """
        Análise específica para colunas categóricas
        """
        value_counts = series.value_counts()
        
        return {
            'top_categories': value_counts.head(10).to_dict(),
            'category_distribution': 'balanced' if value_counts.std() < value_counts.mean() else 'skewed',
            'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
            'frequency_of_most': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
        }
    
    def _analyze_datetime_column(self, series: pd.Series) -> Dict:
        """
        Análise específica para colunas de data
        """
        try:
            datetime_series = pd.to_datetime(series, errors='coerce').dropna()
            
            return {
                'min_date': datetime_series.min().strftime('%Y-%m-%d') if len(datetime_series) > 0 else None,
                'max_date': datetime_series.max().strftime('%Y-%m-%d') if len(datetime_series) > 0 else None,
                'date_range_days': (datetime_series.max() - datetime_series.min()).days if len(datetime_series) > 0 else 0,
                'frequency_pattern': self._detect_date_frequency(datetime_series)
            }
        except:
            return {'error': 'Could not parse dates'}
    
    def _count_outliers(self, series: pd.Series) -> int:
        """
        Conta outliers usando o método IQR
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        return len(outliers)
    
    def _detect_distribution(self, series: pd.Series) -> str:
        """
        Detecta o tipo de distribuição dos dados numéricos
        """
        skewness = series.skew()
        
        if abs(skewness) < 0.5:
            return 'normal'
        elif skewness > 0.5:
            return 'right_skewed'
        else:
            return 'left_skewed'
    
    def _detect_date_frequency(self, datetime_series: pd.Series) -> str:
        """
        Detecta a frequência dos dados de data
        """
        if len(datetime_series) < 2:
            return 'insufficient_data'
        
        # Calcula diferenças entre datas consecutivas
        sorted_dates = datetime_series.sort_values()
        differences = sorted_dates.diff().dropna()
        
        # Determina frequência mais comum
        mode_diff = differences.mode()
        if len(mode_diff) > 0:
            days = mode_diff.iloc[0].days
            
            if days == 1:
                return 'daily'
            elif 6 <= days <= 8:
                return 'weekly'
            elif 28 <= days <= 31:
                return 'monthly'
            elif 88 <= days <= 95:
                return 'quarterly'
            elif 360 <= days <= 370:
                return 'yearly'
        
        return 'irregular'
    
    def _describe_column(self, series: pd.Series) -> str:
        """
        Gera uma descrição textual da coluna
        """
        col_type = self._detect_column_type(series)
        unique_count = series.nunique()
        total_count = len(series)
        
        if col_type == 'numeric':
            return f"Coluna numérica com {unique_count} valores únicos de {total_count} registros"
        elif col_type == 'categorical':
            return f"Coluna categórica com {unique_count} categorias distintas"
        elif col_type == 'datetime':
            return f"Coluna de datas com {unique_count} datas únicas"
        else:
            return f"Coluna de texto com {unique_count} valores únicos"
    
    def _analyze_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        Analisa a qualidade geral dos dados
        """
        total_cells = df.shape[0] * df.shape[1]
        null_cells = df.isnull().sum().sum()
        
        return {
            'completeness_score': ((total_cells - null_cells) / total_cells) * 100,
            'missing_data_percentage': (null_cells / total_cells) * 100,
            'columns_with_nulls': df.columns[df.isnull().any()].tolist(),
            'duplicate_rows': df.duplicated().sum(),
            'data_consistency_score': self._calculate_consistency_score(df)
        }
    
    def _calculate_consistency_score(self, df: pd.DataFrame) -> float:
        """
        Calcula um score de consistência dos dados
        """
        consistency_factors = []
        
        for col in df.columns:
            series = df[col].dropna()
            if len(series) > 0:
                # Fator de consistência baseado na variação de tipos
                type_consistency = 1.0  # Implementação simplificada
                consistency_factors.append(type_consistency)
        
        return np.mean(consistency_factors) * 100 if consistency_factors else 0
    
    def _analyze_relationships(self, df: pd.DataFrame) -> Dict:
        """
        Analisa relações entre colunas
        """
        relationships = {}
        
        # Identifica colunas numéricas para correlação
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 1:
            correlation_matrix = df[numeric_cols].corr()
            
            # Encontra correlações fortes (> 0.7 ou < -0.7)
            strong_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:
                        strong_correlations.append({
                            'col1': correlation_matrix.columns[i],
                            'col2': correlation_matrix.columns[j],
                            'correlation': float(corr_value),
                            'strength': 'strong' if abs(corr_value) > 0.8 else 'moderate'
                        })
            
            relationships['correlations'] = strong_correlations
            relationships['correlation_matrix'] = correlation_matrix.round(3).to_dict()
        
        return relationships
    
    def _recommend_visualizations(self, df: pd.DataFrame) -> List[Dict]:
        """
        Recomenda visualizações baseadas na estrutura dos dados
        """
        recommendations = []
        column_analysis = self._analyze_columns(df)
        
        # Conta tipos de colunas
        numeric_cols = [col for col, info in column_analysis.items() if info['type'] == 'numeric']
        categorical_cols = [col for col, info in column_analysis.items() if info['type'] == 'categorical']
        datetime_cols = [col for col, info in column_analysis.items() if info['type'] == 'datetime']
        
        # Recomendações baseadas nos tipos de dados disponíveis
        
        # Gráficos para colunas numéricas
        for col in numeric_cols:
            recommendations.append({
                'type': 'histogram',
                'title': f'Distribuição de {col}',
                'columns': [col],
                'description': f'Histograma mostrando a distribuição dos valores de {col}',
                'priority': 'high'
            })
            
            recommendations.append({
                'type': 'box',
                'title': f'Box Plot de {col}',
                'columns': [col],
                'description': f'Box plot para identificar outliers em {col}',
                'priority': 'medium'
            })
        
        # Gráficos para colunas categóricas
        for col in categorical_cols:
            unique_count = column_analysis[col]['unique_count']
            if unique_count <= 20:  # Limite para visualização clara
                recommendations.append({
                    'type': 'bar',
                    'title': f'Contagem por {col}',
                    'columns': [col],
                    'description': f'Gráfico de barras mostrando a distribuição de categorias em {col}',
                    'priority': 'high'
                })
                
                recommendations.append({
                    'type': 'pie',
                    'title': f'Proporção de {col}',
                    'columns': [col],
                    'description': f'Gráfico de pizza mostrando as proporções em {col}',
                    'priority': 'medium'
                })
        
        # Gráficos de série temporal para datas
        if datetime_cols and numeric_cols:
            for date_col in datetime_cols:
                for num_col in numeric_cols:
                    recommendations.append({
                        'type': 'line',
                        'title': f'{num_col} ao longo do tempo',
                        'columns': [date_col, num_col],
                        'description': f'Série temporal mostrando {num_col} por {date_col}',
                        'priority': 'high'
                    })
        
        # Gráficos de correlação
        if len(numeric_cols) >= 2:
            recommendations.append({
                'type': 'correlation_heatmap',
                'title': 'Matriz de Correlação',
                'columns': numeric_cols,
                'description': 'Heatmap mostrando correlações entre variáveis numéricas',
                'priority': 'high'
            })
            
            # Scatter plots para pares de variáveis numéricas
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    recommendations.append({
                        'type': 'scatter',
                        'title': f'{col1} vs {col2}',
                        'columns': [col1, col2],
                        'description': f'Gráfico de dispersão entre {col1} e {col2}',
                        'priority': 'medium'
                    })
        
        # Gráficos categórico vs numérico
        if categorical_cols and numeric_cols:
            for cat_col in categorical_cols:
                for num_col in numeric_cols:
                    if column_analysis[cat_col]['unique_count'] <= 10:
                        recommendations.append({
                            'type': 'box_by_category',
                            'title': f'{num_col} por {cat_col}',
                            'columns': [cat_col, num_col],
                            'description': f'Box plot de {num_col} agrupado por {cat_col}',
                            'priority': 'high'
                        })
        
        # Ordena por prioridade
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 0), reverse=True)
        
        return recommendations
    
    def _generate_summary_stats(self, df: pd.DataFrame) -> Dict:
        """
        Gera estatísticas resumidas dos dados
        """
        return {
            'total_rows': int(df.shape[0]),
            'total_columns': int(df.shape[1]),
            'memory_usage_mb': float(df.memory_usage(deep=True).sum() / 1024 / 1024),
            'numeric_columns': int(len(df.select_dtypes(include=[np.number]).columns)),
            'text_columns': int(len(df.select_dtypes(include=['object']).columns)),
            'datetime_columns': int(len(df.select_dtypes(include=['datetime64']).columns))
        }
    
    def _detect_patterns(self, df: pd.DataFrame) -> List[str]:
        """
        Detecta padrões interessantes nos dados
        """
        patterns = []
        
        # Padrão: colunas com muitos valores únicos (possíveis IDs)
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.95:
                patterns.append(f"'{col}' parece ser um identificador único (ID)")
        
        # Padrão: colunas com poucos valores únicos
        for col in df.columns:
            unique_count = df[col].nunique()
            if unique_count <= 3 and len(df) > 10:
                patterns.append(f"'{col}' tem apenas {unique_count} valores únicos - possível variável binária/categórica")
        
        # Padrão: dados faltantes concentrados
        missing_percentage = (df.isnull().sum() / len(df)) * 100
        high_missing_cols = missing_percentage[missing_percentage > 50].index.tolist()
        if high_missing_cols:
            patterns.append(f"Colunas com muitos dados faltantes: {', '.join(high_missing_cols)}")
        
        # Padrão: possíveis séries temporais
        datetime_like_cols = []
        for col in df.columns:
            if 'data' in col.lower() or 'date' in col.lower() or 'tempo' in col.lower():
                datetime_like_cols.append(col)
        
        if datetime_like_cols:
            patterns.append(f"Possíveis colunas de data/tempo detectadas: {', '.join(datetime_like_cols)}")
        
        return patterns

def detect_file_structure(file_content, filename: str) -> Dict[str, Any]:
    """
    Função principal para detectar estrutura de arquivo automaticamente
    """
    analyzer = AutoStructureAnalyzer()
    
    try:
        # Tenta ler como diferentes formatos
        if filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_content)
        elif filename.endswith('.csv'):
            # Tenta diferentes separadores
            for sep in [',', ';', '\t', '|']:
                try:
                    df = pd.read_csv(file_content, sep=sep)
                    if df.shape[1] > 1:  # Se conseguiu separar em múltiplas colunas
                        break
                except:
                    continue
        else:
            # Tenta como CSV genérico
            df = pd.read_csv(file_content)
        
        # Realiza análise completa
        analysis_results = analyzer.analyze_dataframe(df)
        analysis_results['success'] = True
        analysis_results['dataframe_preview'] = df.head(10).to_dict('records')
        analysis_results['columns'] = df.columns.tolist()
        
        return analysis_results
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': 'Não foi possível analisar a estrutura do arquivo'
        }