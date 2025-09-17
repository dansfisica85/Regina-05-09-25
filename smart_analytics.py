"""
Módulo de Machine Learning para Interpretação Inteligente de Dados
Sistema avançado para análise automática de planilhas e geração de insights
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
import nltk
from typing import Dict, List, Tuple, Any, Optional
import re
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataTypeDetector:
    """Detecta tipos de dados avançados usando ML"""
    
    def __init__(self):
        self.patterns = {
            'date': [
                r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
                r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',
                r'\d{1,2}\s+(jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez)',
                r'(segunda|terça|quarta|quinta|sexta|sábado|domingo)'
            ],
            'email': [r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'],
            'phone': [
                r'\(\d{2}\)\s*\d{4,5}[-\s]*\d{4}',
                r'\d{2}\s*\d{4,5}[-\s]*\d{4}',
                r'\+55\s*\d{2}\s*\d{4,5}[-\s]*\d{4}'
            ],
            'cpf': [r'\d{3}\.\d{3}\.\d{3}-\d{2}', r'\d{11}'],
            'currency': [
                r'R\$\s*\d+[.,]\d{2}',
                r'\$\s*\d+[.,]\d{2}',
                r'\d+[.,]\d{2}\s*(reais|real|R\$)'
            ],
            'percentage': [
                r'\d+[.,]?\d*\s*%',
                r'\d+[.,]?\d*\s*(por\s*cento|porcento)'
            ],
            'code': [
                r'^[A-Z]{2,5}\d{3,8}$',
                r'^[A-Z]\d{3,8}[A-Z]?$'
            ]
        }
    
    def detect_column_type(self, series: pd.Series) -> Dict[str, Any]:
        """Detecta o tipo de uma coluna usando múltiplas heurísticas"""
        result = {
            'type': 'unknown',
            'subtype': None,
            'confidence': 0.0,
            'patterns_found': [],
            'statistics': {},
            'quality_score': 0.0
        }
        
        # Remove valores nulos para análise
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return result
        
        # Converte para string para análise de padrões
        str_series = clean_series.astype(str)
        
        # Detecta padrões textuais
        for pattern_type, patterns in self.patterns.items():
            matches = 0
            for pattern in patterns:
                matches += str_series.str.contains(pattern, regex=True, case=False).sum()
            
            if matches > 0:
                confidence = matches / len(str_series)
                if confidence > result['confidence']:
                    result['type'] = pattern_type
                    result['confidence'] = confidence
                    result['patterns_found'].append({
                        'type': pattern_type,
                        'matches': matches,
                        'confidence': confidence
                    })
        
        # Análise numérica
        numeric_result = self._analyze_numeric_column(series)
        if numeric_result['confidence'] > result['confidence']:
            result.update(numeric_result)
        
        # Análise categórica
        categorical_result = self._analyze_categorical_column(series)
        if categorical_result['confidence'] > result['confidence']:
            result.update(categorical_result)
        
        # Calcula score de qualidade
        result['quality_score'] = self._calculate_quality_score(series)
        
        return result
    
    def _analyze_numeric_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analisa colunas numéricas"""
        result = {'type': 'numeric', 'confidence': 0.0, 'statistics': {}}
        
        # Tenta converter para numérico
        numeric_series = pd.to_numeric(series, errors='coerce')
        non_null_count = numeric_series.count()
        
        if non_null_count == 0:
            return result
        
        confidence = non_null_count / len(series)
        
        if confidence > 0.8:  # 80% dos valores são numéricos
            result['confidence'] = confidence
            
            # Estatísticas descritivas
            stats_dict = {
                'mean': float(numeric_series.mean()),
                'median': float(numeric_series.median()),
                'std': float(numeric_series.std()),
                'min': float(numeric_series.min()),
                'max': float(numeric_series.max()),
                'range': float(numeric_series.max() - numeric_series.min()),
                'skewness': float(stats.skew(numeric_series.dropna())),
                'kurtosis': float(stats.kurtosis(numeric_series.dropna()))
            }
            result['statistics'] = stats_dict
            
            # Subtipo numérico
            if all(numeric_series.dropna() == numeric_series.dropna().astype(int)):
                result['subtype'] = 'integer'
            else:
                result['subtype'] = 'float'
            
            # Detecta se pode ser uma escala (0-100, 0-10, etc.)
            min_val, max_val = numeric_series.min(), numeric_series.max()
            if 0 <= min_val and max_val <= 100:
                result['subtype'] = 'scale_0_100'
            elif 0 <= min_val and max_val <= 10:
                result['subtype'] = 'scale_0_10'
        
        return result
    
    def _analyze_categorical_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analisa colunas categóricas"""
        result = {'type': 'categorical', 'confidence': 0.0, 'statistics': {}}
        
        unique_count = series.nunique()
        total_count = len(series)
        
        if unique_count == 0:
            return result
        
        # Calcula razão de cardinalidade
        cardinality_ratio = unique_count / total_count
        
        # Se tem poucos valores únicos em relação ao total, é provavelmente categórico
        if cardinality_ratio < 0.5:
            result['confidence'] = 1 - cardinality_ratio
            
            # Estatísticas categóricas
            value_counts = series.value_counts()
            result['statistics'] = {
                'unique_count': int(unique_count),
                'most_frequent': str(value_counts.index[0]),
                'most_frequent_count': int(value_counts.iloc[0]),
                'distribution': value_counts.head(10).to_dict(),
                'cardinality_ratio': float(cardinality_ratio)
            }
            
            # Subtipo categórico
            if unique_count <= 10:
                result['subtype'] = 'low_cardinality'
            elif unique_count <= 50:
                result['subtype'] = 'medium_cardinality'
            else:
                result['subtype'] = 'high_cardinality'
        
        return result
    
    def _calculate_quality_score(self, series: pd.Series) -> float:
        """Calcula um score de qualidade dos dados"""
        total_count = len(series)
        if total_count == 0:
            return 0.0
        
        # Penaliza valores nulos
        null_ratio = series.isnull().sum() / total_count
        
        # Penaliza valores duplicados se for uma coluna que deveria ser única
        duplicate_ratio = series.duplicated().sum() / total_count
        
        # Score base
        quality_score = 1.0 - (null_ratio * 0.5) - (duplicate_ratio * 0.3)
        
        return max(0.0, quality_score)


class RelationshipDetector:
    """Detecta relacionamentos entre colunas usando ML"""
    
    def __init__(self):
        self.correlation_threshold = 0.7
        self.mutual_info_threshold = 0.3
    
    def find_relationships(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Encontra relacionamentos entre todas as colunas"""
        relationships = {
            'correlations': [],
            'dependencies': [],
            'clusters': [],
            'anomalies': []
        }
        
        # Prepara dados numéricos para análise
        numeric_df = self._prepare_numeric_data(df)
        
        if not numeric_df.empty:
            # Correlações
            relationships['correlations'] = self._find_correlations(numeric_df)
            
            # Clustering de colunas similares
            relationships['clusters'] = self._cluster_similar_columns(numeric_df)
            
            # Detecção de anomalias
            relationships['anomalies'] = self._detect_anomalies(numeric_df)
        
        # Dependências funcionais
        relationships['dependencies'] = self._find_functional_dependencies(df)
        
        return relationships
    
    def _prepare_numeric_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepara dados para análise numérica"""
        numeric_df = pd.DataFrame()
        
        for col in df.columns:
            series = df[col]
            
            # Tenta converter para numérico
            numeric_series = pd.to_numeric(series, errors='coerce')
            
            if numeric_series.count() > len(series) * 0.5:  # Pelo menos 50% numérico
                numeric_df[col] = numeric_series
            else:
                # Para categóricas, usa encoding
                if series.nunique() < 20:  # Baixa cardinalidade
                    le = LabelEncoder()
                    encoded = le.fit_transform(series.astype(str).fillna('missing'))
                    numeric_df[f"{col}_encoded"] = encoded
        
        return numeric_df.fillna(0)
    
    def _find_correlations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Encontra correlações significativas"""
        correlations = []
        corr_matrix = df.corr()
        
        for i, col1 in enumerate(df.columns):
            for j, col2 in enumerate(df.columns[i+1:], i+1):
                corr_value = corr_matrix.loc[col1, col2]
                
                if abs(corr_value) >= self.correlation_threshold:
                    correlations.append({
                        'column1': col1,
                        'column2': col2,
                        'correlation': float(corr_value),
                        'strength': 'strong' if abs(corr_value) > 0.8 else 'moderate',
                        'direction': 'positive' if corr_value > 0 else 'negative'
                    })
        
        return sorted(correlations, key=lambda x: abs(x['correlation']), reverse=True)
    
    def _cluster_similar_columns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Agrupa colunas similares"""
        if len(df.columns) < 3:
            return []
        
        # Transpõe para agrupar colunas como observações
        df_transposed = df.T
        
        # Standardiza os dados
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_transposed.fillna(0))
        
        # K-means clustering
        n_clusters = min(5, len(df.columns) // 2)
        if n_clusters < 2:
            return []
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        # Organiza resultados
        clusters = []
        for cluster_id in range(n_clusters):
            cluster_columns = [col for i, col in enumerate(df.columns) 
                             if cluster_labels[i] == cluster_id]
            
            if len(cluster_columns) > 1:
                clusters.append({
                    'cluster_id': int(cluster_id),
                    'columns': cluster_columns,
                    'size': len(cluster_columns),
                    'description': f"Grupo de {len(cluster_columns)} colunas similares"
                })
        
        return clusters
    
    def _detect_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detecta anomalias nos dados"""
        anomalies = []
        
        # Isolation Forest para detecção de outliers
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        
        for col in df.columns:
            series = df[col].dropna()
            if len(series) < 10:  # Precisa de dados suficientes
                continue
            
            # Detecção de outliers
            outliers = iso_forest.fit_predict(series.values.reshape(-1, 1))
            outlier_indices = np.where(outliers == -1)[0]
            
            if len(outlier_indices) > 0:
                anomalies.append({
                    'column': col,
                    'anomaly_type': 'outliers',
                    'count': len(outlier_indices),
                    'percentage': len(outlier_indices) / len(series) * 100,
                    'values': series.iloc[outlier_indices].tolist()[:5]  # Primeiros 5
                })
        
        return anomalies
    
    def _find_functional_dependencies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detecta dependências funcionais simples"""
        dependencies = []
        
        for col1 in df.columns:
            for col2 in df.columns:
                if col1 == col2:
                    continue
                
                # Verifica se col1 determina col2 (col1 -> col2)
                grouped = df.groupby(col1)[col2].nunique()
                
                # Se cada valor de col1 corresponde a apenas um valor de col2
                if all(grouped == 1):
                    dependencies.append({
                        'determinant': col1,
                        'dependent': col2,
                        'type': 'functional_dependency',
                        'strength': 'complete',
                        'description': f"{col1} determina completamente {col2}"
                    })
        
        return dependencies


class SmartChartRecommender:
    """Recomenda tipos de gráficos baseado nos dados usando ML"""
    
    def __init__(self):
        self.chart_rules = {
            'numeric_single': ['histogram', 'box_plot', 'density'],
            'numeric_multiple': ['correlation_matrix', 'scatter_matrix', 'pca'],
            'categorical_single': ['bar_chart', 'pie_chart', 'donut'],
            'categorical_multiple': ['stacked_bar', 'grouped_bar', 'heatmap'],
            'time_series': ['line_chart', 'area_chart', 'candlestick'],
            'numeric_vs_categorical': ['box_plot_grouped', 'violin_plot', 'bar_chart'],
            'numeric_vs_numeric': ['scatter_plot', 'regression_plot', 'bubble_chart'],
            'mixed_analysis': ['parallel_coordinates', 'radar_chart', 'treemap']
        }
    
    def recommend_charts(self, df: pd.DataFrame, column_types: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """Recomenda gráficos baseado na estrutura dos dados"""
        recommendations = []
        
        # Classifica colunas por tipo
        numeric_cols = [col for col, info in column_types.items() 
                       if info['type'] == 'numeric']
        categorical_cols = [col for col, info in column_types.items() 
                           if info['type'] == 'categorical']
        date_cols = [col for col, info in column_types.items() 
                    if info['type'] == 'date']
        
        # Recomendações baseadas na composição dos dados
        if len(numeric_cols) == 1 and len(categorical_cols) == 0:
            recommendations.extend(self._recommend_single_numeric(numeric_cols[0], df))
        
        elif len(numeric_cols) > 1 and len(categorical_cols) == 0:
            recommendations.extend(self._recommend_multiple_numeric(numeric_cols, df))
        
        elif len(categorical_cols) == 1 and len(numeric_cols) == 0:
            recommendations.extend(self._recommend_single_categorical(categorical_cols[0], df))
        
        elif len(categorical_cols) > 1 and len(numeric_cols) == 0:
            recommendations.extend(self._recommend_multiple_categorical(categorical_cols, df))
        
        elif len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
            recommendations.extend(self._recommend_mixed_analysis(numeric_cols, categorical_cols, df))
        
        if date_cols:
            recommendations.extend(self._recommend_time_series(date_cols, numeric_cols, df))
        
        # Ordena por score de relevância
        return sorted(recommendations, key=lambda x: x['relevance_score'], reverse=True)
    
    def _recommend_single_numeric(self, column: str, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Recomendações para uma coluna numérica"""
        recommendations = []
        series = df[column].dropna()
        
        if len(series) == 0:
            return recommendations
        
        # Histograma - sempre relevante para distribuição
        recommendations.append({
            'chart_type': 'histogram',
            'title': f'Distribuição de {column}',
            'columns': [column],
            'description': f'Mostra a distribuição de frequência dos valores de {column}',
            'relevance_score': 0.9,
            'config': {
                'bins': min(30, len(series.unique()))
            }
        })
        
        # Box plot para identificar outliers
        recommendations.append({
            'chart_type': 'box_plot',
            'title': f'Box Plot de {column}',
            'columns': [column],
            'description': f'Identifica outliers e quartis de {column}',
            'relevance_score': 0.8,
            'config': {}
        })
        
        return recommendations
    
    def _recommend_multiple_numeric(self, columns: List[str], df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Recomendações para múltiplas colunas numéricas"""
        recommendations = []
        
        # Matriz de correlação
        recommendations.append({
            'chart_type': 'correlation_matrix',
            'title': 'Matriz de Correlação',
            'columns': columns,
            'description': 'Mostra correlações entre variáveis numéricas',
            'relevance_score': 0.95,
            'config': {'method': 'pearson'}
        })
        
        # Scatter plot para pares de variáveis
        if len(columns) >= 2:
            for i, col1 in enumerate(columns):
                for col2 in columns[i+1:]:
                    recommendations.append({
                        'chart_type': 'scatter_plot',
                        'title': f'{col1} vs {col2}',
                        'columns': [col1, col2],
                        'description': f'Relação entre {col1} e {col2}',
                        'relevance_score': 0.7,
                        'config': {'x': col1, 'y': col2}
                    })
        
        return recommendations
    
    def _recommend_single_categorical(self, column: str, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Recomendações para uma coluna categórica"""
        recommendations = []
        unique_count = df[column].nunique()
        
        # Gráfico de barras
        recommendations.append({
            'chart_type': 'bar_chart',
            'title': f'Distribuição de {column}',
            'columns': [column],
            'description': f'Frequência das categorias em {column}',
            'relevance_score': 0.9,
            'config': {}
        })
        
        # Gráfico de pizza se não há muitas categorias
        if unique_count <= 8:
            recommendations.append({
                'chart_type': 'pie_chart',
                'title': f'Proporção de {column}',
                'columns': [column],
                'description': f'Proporção relativa das categorias em {column}',
                'relevance_score': 0.8,
                'config': {}
            })
        
        return recommendations
    
    def _recommend_multiple_categorical(self, columns: List[str], df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Recomendações para múltiplas colunas categóricas"""
        recommendations = []
        
        # Heatmap de contingência
        if len(columns) >= 2:
            recommendations.append({
                'chart_type': 'contingency_heatmap',
                'title': f'Tabela de Contingência',
                'columns': columns[:2],  # Pega as duas primeiras
                'description': 'Mostra a relação entre categorias',
                'relevance_score': 0.85,
                'config': {}
            })
        
        return recommendations
    
    def _recommend_mixed_analysis(self, numeric_cols: List[str], categorical_cols: List[str], df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Recomendações para análise mista"""
        recommendations = []
        
        # Box plot agrupado
        for num_col in numeric_cols:
            for cat_col in categorical_cols:
                if df[cat_col].nunique() <= 10:  # Não muitas categorias
                    recommendations.append({
                        'chart_type': 'grouped_box_plot',
                        'title': f'{num_col} por {cat_col}',
                        'columns': [num_col, cat_col],
                        'description': f'Distribuição de {num_col} em cada categoria de {cat_col}',
                        'relevance_score': 0.85,
                        'config': {'x': cat_col, 'y': num_col}
                    })
        
        return recommendations
    
    def _recommend_time_series(self, date_cols: List[str], numeric_cols: List[str], df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Recomendações para séries temporais"""
        recommendations = []
        
        for date_col in date_cols:
            for num_col in numeric_cols:
                recommendations.append({
                    'chart_type': 'time_series',
                    'title': f'{num_col} ao longo do tempo',
                    'columns': [date_col, num_col],
                    'description': f'Evolução temporal de {num_col}',
                    'relevance_score': 0.9,
                    'config': {'x': date_col, 'y': num_col}
                })
        
        return recommendations


class AutoInsightGenerator:
    """Gera insights automáticos usando análise estatística e ML"""
    
    def __init__(self):
        self.significance_level = 0.05
        self.effect_size_threshold = 0.5
    
    def generate_insights(self, df: pd.DataFrame, column_types: Dict[str, Dict], relationships: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Gera insights automáticos sobre os dados"""
        insights = []
        
        # Insights sobre qualidade dos dados
        insights.extend(self._data_quality_insights(df, column_types))
        
        # Insights estatísticos
        insights.extend(self._statistical_insights(df, column_types))
        
        # Insights sobre relacionamentos
        insights.extend(self._relationship_insights(relationships))
        
        # Insights sobre anomalias
        insights.extend(self._anomaly_insights(relationships.get('anomalies', [])))
        
        # Ordena por importância
        return sorted(insights, key=lambda x: x['importance'], reverse=True)
    
    def _data_quality_insights(self, df: pd.DataFrame, column_types: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """Insights sobre qualidade dos dados"""
        insights = []
        
        # Valores faltantes
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            worst_columns = missing_data[missing_data > 0].sort_values(ascending=False)
            insights.append({
                'type': 'data_quality',
                'category': 'missing_data',
                'title': 'Dados Faltantes Detectados',
                'description': f'Encontrados valores faltantes em {len(worst_columns)} colunas. '
                             f'A coluna "{worst_columns.index[0]}" tem {worst_columns.iloc[0]} valores faltantes '
                             f'({worst_columns.iloc[0]/len(df)*100:.1f}% dos dados).',
                'importance': 0.8,
                'actionable': True,
                'recommendation': 'Considere estratégias de imputação ou remoção de dados faltantes.',
                'affected_columns': worst_columns.index.tolist()
            })
        
        # Colunas com baixa qualidade
        low_quality_cols = [col for col, info in column_types.items() 
                           if info.get('quality_score', 1.0) < 0.7]
        
        if low_quality_cols:
            insights.append({
                'type': 'data_quality',
                'category': 'low_quality',
                'title': 'Colunas com Qualidade Baixa',
                'description': f'As colunas {", ".join(low_quality_cols)} apresentam problemas de qualidade '
                             'como muitos valores duplicados ou formatos inconsistentes.',
                'importance': 0.7,
                'actionable': True,
                'recommendation': 'Revise e limpe os dados dessas colunas.',
                'affected_columns': low_quality_cols
            })
        
        return insights
    
    def _statistical_insights(self, df: pd.DataFrame, column_types: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """Insights estatísticos"""
        insights = []
        
        # Busca por distribuições interessantes
        for col, info in column_types.items():
            if info['type'] == 'numeric' and 'statistics' in info:
                stats_dict = info['statistics']
                
                # Distribuição altamente enviesada
                if abs(stats_dict.get('skewness', 0)) > 2:
                    skew_direction = 'direita' if stats_dict['skewness'] > 0 else 'esquerda'
                    insights.append({
                        'type': 'statistical',
                        'category': 'distribution',
                        'title': f'Distribuição Enviesada em {col}',
                        'description': f'A coluna "{col}" apresenta forte assimetria para a {skew_direction} '
                                     f'(skewness = {stats_dict["skewness"]:.2f}).',
                        'importance': 0.6,
                        'actionable': True,
                        'recommendation': 'Considere transformações logarítmicas ou outras técnicas de normalização.',
                        'affected_columns': [col]
                    })
                
                # Outliers extremos
                q1 = np.percentile(df[col].dropna(), 25)
                q3 = np.percentile(df[col].dropna(), 75)
                iqr = q3 - q1
                outlier_threshold = q3 + 1.5 * iqr
                outliers = df[col][df[col] > outlier_threshold].count()
                
                if outliers > len(df) * 0.05:  # Mais de 5% são outliers
                    insights.append({
                        'type': 'statistical',
                        'category': 'outliers',
                        'title': f'Muitos Outliers em {col}',
                        'description': f'A coluna "{col}" contém {outliers} outliers ({outliers/len(df)*100:.1f}% dos dados), '
                                     'que podem indicar problemas de qualidade ou eventos excepcionais.',
                        'importance': 0.7,
                        'actionable': True,
                        'recommendation': 'Investigue os outliers para determinar se são válidos ou erros.',
                        'affected_columns': [col]
                    })
        
        return insights
    
    def _relationship_insights(self, relationships: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Insights sobre relacionamentos"""
        insights = []
        
        # Correlações fortes
        strong_correlations = [corr for corr in relationships.get('correlations', [])
                             if abs(corr['correlation']) > 0.8]
        
        if strong_correlations:
            top_corr = strong_correlations[0]
            insights.append({
                'type': 'relationship',
                'category': 'correlation',
                'title': 'Correlação Forte Detectada',
                'description': f'As variáveis "{top_corr["column1"]}" e "{top_corr["column2"]}" '
                             f'apresentam correlação {top_corr["strength"]} '
                             f'({top_corr["correlation"]:.3f}).',
                'importance': 0.9,
                'actionable': True,
                'recommendation': 'Investigue se existe causalidade ou se uma variável pode predizer a outra.',
                'affected_columns': [top_corr["column1"], top_corr["column2"]]
            })
        
        # Dependências funcionais
        dependencies = relationships.get('dependencies', [])
        if dependencies:
            dep = dependencies[0]
            insights.append({
                'type': 'relationship',
                'category': 'dependency',
                'title': 'Dependência Funcional Encontrada',
                'description': f'A coluna "{dep["determinant"]}" determina completamente "{dep["dependent"]}", '
                             'indicando uma relação funcional entre elas.',
                'importance': 0.8,
                'actionable': True,
                'recommendation': 'Considere usar esta relação para validação de dados ou criação de chaves.',
                'affected_columns': [dep["determinant"], dep["dependent"]]
            })
        
        return insights
    
    def _anomaly_insights(self, anomalies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Insights sobre anomalias"""
        insights = []
        
        for anomaly in anomalies:
            if anomaly['percentage'] > 5:  # Mais de 5% são anomalias
                insights.append({
                    'type': 'anomaly',
                    'category': 'outliers',
                    'title': f'Anomalias em {anomaly["column"]}',
                    'description': f'Detectadas {anomaly["count"]} anomalias na coluna "{anomaly["column"]}" '
                                 f'({anomaly["percentage"]:.1f}% dos dados).',
                    'importance': 0.6,
                    'actionable': True,
                    'recommendation': 'Analise estes valores anômalos para entender se representam erros ou eventos especiais.',
                    'affected_columns': [anomaly["column"]]
                })
        
        return insights