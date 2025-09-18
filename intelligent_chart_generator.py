"""
Análise Regina - Smart | Gerador Inteligente de Gráficos
Sistema que cria automaticamente gráficos otimizados baseados na estrutura dos dados
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import json
from typing import Dict, List, Any, Optional

class IntelligentChartGenerator:
    """
    Gerador inteligente que cria gráficos automaticamente baseado nos dados
    """
    
    def __init__(self):
        self.color_palette = [
            '#667eea', '#764ba2', '#f093fb', '#f5576c',
            '#4facfe', '#00f2fe', '#43e97b', '#38f9d7',
            '#ffecd2', '#fcb69f', '#a8edea', '#fed6e3',
            '#ff9a9e', '#fecfef', '#ffeaa7', '#fab1a0'
        ]
    
    def generate_all_charts(self, df: pd.DataFrame, analysis_results: Dict) -> List[Dict]:
        """
        Gera todos os gráficos recomendados automaticamente
        """
        charts = []
        recommendations = analysis_results.get('recommended_visualizations', [])
        
        # Limita a 12 gráficos principais para performance
        top_recommendations = recommendations[:12]
        
        for i, rec in enumerate(top_recommendations):
            try:
                chart_data = self._create_chart(df, rec, i)
                if chart_data:
                    charts.append(chart_data)
            except Exception as e:
                print(f"Erro ao criar gráfico {rec['type']}: {str(e)}")
                continue
        
        return charts
    
    def _create_chart(self, df: pd.DataFrame, recommendation: Dict, index: int) -> Optional[Dict]:
        """
        Cria um gráfico específico baseado na recomendação
        """
        chart_type = recommendation['type']
        columns = recommendation['columns']
        title = recommendation['title']
        
        # Verifica se as colunas existem no DataFrame
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            return None
        
        color = self.color_palette[index % len(self.color_palette)]
        
        # Roteamento para diferentes tipos de gráfico
        if chart_type == 'histogram':
            return self._create_histogram(df, columns[0], title, color)
        elif chart_type == 'box':
            return self._create_box_plot(df, columns[0], title, color)
        elif chart_type == 'bar':
            return self._create_bar_chart(df, columns[0], title, color)
        elif chart_type == 'pie':
            return self._create_pie_chart(df, columns[0], title, color)
        elif chart_type == 'line':
            return self._create_line_chart(df, columns[0], columns[1], title, color)
        elif chart_type == 'scatter':
            return self._create_scatter_plot(df, columns[0], columns[1], title, color)
        elif chart_type == 'correlation_heatmap':
            return self._create_correlation_heatmap(df, columns, title)
        elif chart_type == 'box_by_category':
            return self._create_box_by_category(df, columns[0], columns[1], title, color)
        
        return None
    
    def _create_histogram(self, df: pd.DataFrame, column: str, title: str, color: str) -> Dict:
        """
        Cria histograma para variável numérica
        """
        try:
            # Remove valores nulos
            clean_data = df[column].dropna()
            
            # Determina número de bins automaticamente
            n_bins = min(50, max(10, int(len(clean_data) ** 0.5)))
            
            fig = px.histogram(
                x=clean_data,
                nbins=n_bins,
                title=title,
                labels={'x': column, 'count': 'Frequência'},
                color_discrete_sequence=[color]
            )
            
            # Adiciona estatísticas
            mean_val = clean_data.mean()
            median_val = clean_data.median()
            
            fig.add_vline(
                x=mean_val, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Média: {mean_val:.2f}",
                annotation_position="top"
            )
            
            fig.add_vline(
                x=median_val, 
                line_dash="dot", 
                line_color="orange",
                annotation_text=f"Mediana: {median_val:.2f}",
                annotation_position="bottom"
            )
            
            self._apply_theme(fig)
            
            return {
                'id': f'hist_{column}',
                'type': 'histogram',
                'title': title,
                'description': f'Distribuição dos valores de {column}',
                'plot_json': fig.to_json(),
                'insights': self._generate_histogram_insights(clean_data)
            }
            
        except Exception as e:
            return None
    
    def _create_box_plot(self, df: pd.DataFrame, column: str, title: str, color: str) -> Dict:
        """
        Cria box plot para detectar outliers
        """
        try:
            clean_data = df[column].dropna()
            
            fig = px.box(
                y=clean_data,
                title=title,
                labels={'y': column},
                color_discrete_sequence=[color]
            )
            
            self._apply_theme(fig)
            
            # Calcula estatísticas de outliers
            Q1 = clean_data.quantile(0.25)
            Q3 = clean_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = clean_data[(clean_data < Q1 - 1.5 * IQR) | (clean_data > Q3 + 1.5 * IQR)]
            
            return {
                'id': f'box_{column}',
                'type': 'box',
                'title': title,
                'description': f'Box plot para identificar outliers em {column}',
                'plot_json': fig.to_json(),
                'insights': {
                    'outliers_count': len(outliers),
                    'outliers_percentage': (len(outliers) / len(clean_data)) * 100,
                    'quartiles': {'Q1': Q1, 'Q2': clean_data.median(), 'Q3': Q3}
                }
            }
            
        except Exception as e:
            return None
    
    def _create_bar_chart(self, df: pd.DataFrame, column: str, title: str, color: str) -> Dict:
        """
        Cria gráfico de barras para variáveis categóricas
        """
        try:
            # Conta valores e pega top 20 para legibilidade
            value_counts = df[column].value_counts().head(20)
            
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=title,
                labels={'x': column, 'y': 'Contagem'},
                color_discrete_sequence=[color]
            )
            
            # Rotaciona labels se necessário
            if len(str(value_counts.index[0])) > 10:
                fig.update_xaxes(tickangle=45)
            
            self._apply_theme(fig)
            
            return {
                'id': f'bar_{column}',
                'type': 'bar',
                'title': title,
                'description': f'Distribuição de categorias em {column}',
                'plot_json': fig.to_json(),
                'insights': {
                    'total_categories': len(df[column].unique()),
                    'top_category': value_counts.index[0],
                    'top_count': int(value_counts.iloc[0]),
                    'distribution_balance': 'equilibrada' if value_counts.std() < value_counts.mean() else 'desbalanceada'
                }
            }
            
        except Exception as e:
            return None
    
    def _create_pie_chart(self, df: pd.DataFrame, column: str, title: str, color: str) -> Dict:
        """
        Cria gráfico de pizza para proporções
        """
        try:
            # Pega top 10 categorias para legibilidade
            value_counts = df[column].value_counts().head(10)
            
            # Se há muitas categorias pequenas, agrupa em "Outros"
            if len(df[column].unique()) > 10:
                others_count = df[column].value_counts().iloc[10:].sum()
                if others_count > 0:
                    value_counts['Outros'] = others_count
            
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=title,
                color_discrete_sequence=self.color_palette
            )
            
            # Melhora layout do pie chart
            fig.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Contagem: %{value}<br>Porcentagem: %{percent}<extra></extra>'
            )
            
            self._apply_theme(fig)
            
            return {
                'id': f'pie_{column}',
                'type': 'pie',
                'title': title,
                'description': f'Proporções das categorias em {column}',
                'plot_json': fig.to_json(),
                'insights': {
                    'total_categories': len(value_counts),
                    'largest_segment': value_counts.index[0],
                    'largest_percentage': (value_counts.iloc[0] / value_counts.sum()) * 100
                }
            }
            
        except Exception as e:
            return None
    
    def _create_line_chart(self, df: pd.DataFrame, x_col: str, y_col: str, title: str, color: str) -> Dict:
        """
        Cria gráfico de linha para séries temporais
        """
        try:
            # Ordena por coluna X (assumindo que é data/tempo)
            clean_df = df[[x_col, y_col]].dropna()
            
            # Tenta converter x_col para datetime se não for
            try:
                clean_df[x_col] = pd.to_datetime(clean_df[x_col])
                clean_df = clean_df.sort_values(x_col)
            except:
                # Se não conseguir converter, usa como está
                pass
            
            fig = px.line(
                clean_df,
                x=x_col,
                y=y_col,
                title=title,
                labels={x_col: x_col, y_col: y_col},
                color_discrete_sequence=[color]
            )
            
            # Adiciona pontos para melhor visualização
            fig.update_traces(mode='lines+markers', marker=dict(size=4))
            
            self._apply_theme(fig)
            
            return {
                'id': f'line_{x_col}_{y_col}',
                'type': 'line',
                'title': title,
                'description': f'Evolução de {y_col} ao longo de {x_col}',
                'plot_json': fig.to_json(),
                'insights': {
                    'trend': self._detect_trend(clean_df[y_col].values),
                    'data_points': len(clean_df),
                    'date_range': f"{clean_df[x_col].min()} até {clean_df[x_col].max()}" if clean_df[x_col].dtype.name.startswith('datetime') else None
                }
            }
            
        except Exception as e:
            return None
    
    def _create_scatter_plot(self, df: pd.DataFrame, x_col: str, y_col: str, title: str, color: str) -> Dict:
        """
        Cria gráfico de dispersão para correlação
        """
        try:
            clean_df = df[[x_col, y_col]].dropna()
            
            # Calcula correlação
            correlation = clean_df[x_col].corr(clean_df[y_col])
            
            fig = px.scatter(
                clean_df,
                x=x_col,
                y=y_col,
                title=f"{title} (Correlação: {correlation:.3f})",
                labels={x_col: x_col, y_col: y_col},
                color_discrete_sequence=[color]
            )
            
            # Adiciona linha de tendência se correlação for significativa
            if abs(correlation) > 0.3:
                fig.add_scatter(
                    x=clean_df[x_col],
                    y=clean_df[x_col] * correlation * (clean_df[y_col].std() / clean_df[x_col].std()) + 
                      clean_df[y_col].mean() - correlation * (clean_df[y_col].std() / clean_df[x_col].std()) * clean_df[x_col].mean(),
                    mode='lines',
                    name='Tendência',
                    line=dict(color='red', dash='dash')
                )
            
            self._apply_theme(fig)
            
            return {
                'id': f'scatter_{x_col}_{y_col}',
                'type': 'scatter',
                'title': title,
                'description': f'Relação entre {x_col} e {y_col}',
                'plot_json': fig.to_json(),
                'insights': {
                    'correlation': correlation,
                    'correlation_strength': self._interpret_correlation(correlation),
                    'data_points': len(clean_df)
                }
            }
            
        except Exception as e:
            return None
    
    def _create_correlation_heatmap(self, df: pd.DataFrame, columns: List[str], title: str) -> Dict:
        """
        Cria heatmap de correlação
        """
        try:
            # Seleciona apenas colunas numéricas
            numeric_df = df[columns].select_dtypes(include=['number'])
            
            if numeric_df.empty or len(numeric_df.columns) < 2:
                return None
            
            corr_matrix = numeric_df.corr()
            
            fig = px.imshow(
                corr_matrix,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                color_continuous_scale='RdBu_r',
                aspect="auto",
                title=title,
                text_auto=True
            )
            
            # Formata valores para 2 casas decimais
            fig.update_traces(
                texttemplate='%{z:.2f}',
                textfont=dict(size=10)
            )
            
            self._apply_theme(fig)
            
            return {
                'id': 'correlation_heatmap',
                'type': 'heatmap',
                'title': title,
                'description': 'Matriz de correlação entre variáveis numéricas',
                'plot_json': fig.to_json(),
                'insights': {
                    'strongest_positive': self._find_strongest_correlation(corr_matrix, positive=True),
                    'strongest_negative': self._find_strongest_correlation(corr_matrix, positive=False),
                    'variables_count': len(corr_matrix.columns)
                }
            }
            
        except Exception as e:
            return None
    
    def _create_box_by_category(self, df: pd.DataFrame, cat_col: str, num_col: str, title: str, color: str) -> Dict:
        """
        Cria box plot agrupado por categoria
        """
        try:
            clean_df = df[[cat_col, num_col]].dropna()
            
            # Limita categorias para legibilidade
            top_categories = clean_df[cat_col].value_counts().head(10).index
            filtered_df = clean_df[clean_df[cat_col].isin(top_categories)]
            
            fig = px.box(
                filtered_df,
                x=cat_col,
                y=num_col,
                title=title,
                labels={cat_col: cat_col, num_col: num_col},
                color_discrete_sequence=[color]
            )
            
            # Rotaciona labels se necessário
            if len(str(top_categories[0])) > 10:
                fig.update_xaxes(tickangle=45)
            
            self._apply_theme(fig)
            
            return {
                'id': f'box_cat_{cat_col}_{num_col}',
                'type': 'box_by_category',
                'title': title,
                'description': f'Distribuição de {num_col} por categorias de {cat_col}',
                'plot_json': fig.to_json(),
                'insights': {
                    'categories_shown': len(top_categories),
                    'category_with_highest_median': self._find_category_with_highest_median(filtered_df, cat_col, num_col),
                    'total_data_points': len(filtered_df)
                }
            }
            
        except Exception as e:
            return None
    
    def _apply_theme(self, fig):
        """
        Aplica tema consistente aos gráficos
        """
        fig.update_layout(
            font_family="Inter, sans-serif",
            font_size=12,
            title_font_size=16,
            title_font_family="Inter, sans-serif",
            title_x=0.5,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=60, b=50),
            height=400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Grid style
        fig.update_xaxes(
            gridcolor='lightgray',
            gridwidth=0.5,
            showline=True,
            linewidth=1,
            linecolor='lightgray'
        )
        fig.update_yaxes(
            gridcolor='lightgray',
            gridwidth=0.5,
            showline=True,
            linewidth=1,
            linecolor='lightgray'
        )
    
    def _generate_histogram_insights(self, data) -> Dict:
        """
        Gera insights para histograma
        """
        return {
            'mean': float(data.mean()),
            'median': float(data.median()),
            'std': float(data.std()),
            'skewness': float(data.skew()),
            'distribution_type': 'normal' if abs(data.skew()) < 0.5 else 'skewed'
        }
    
    def _detect_trend(self, values) -> str:
        """
        Detecta tendência em série temporal
        """
        if len(values) < 3:
            return 'insufficient_data'
        
        # Regressão linear simples para detectar tendência
        x = range(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if abs(slope) < 0.01:
            return 'stable'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def _interpret_correlation(self, correlation: float) -> str:
        """
        Interpreta força da correlação
        """
        abs_corr = abs(correlation)
        
        if abs_corr >= 0.8:
            return 'muito forte'
        elif abs_corr >= 0.6:
            return 'forte'
        elif abs_corr >= 0.4:
            return 'moderada'
        elif abs_corr >= 0.2:
            return 'fraca'
        else:
            return 'muito fraca'
    
    def _find_strongest_correlation(self, corr_matrix, positive=True) -> Dict:
        """
        Encontra a correlação mais forte na matriz
        """
        # Remove diagonal (correlação de uma variável consigo mesma)
        mask = corr_matrix != 1.0
        masked_corr = corr_matrix.where(mask)
        
        if positive:
            max_corr = masked_corr.max().max()
            if pd.isna(max_corr):
                return None
            
            # Encontra posição do valor máximo
            max_pos = masked_corr.stack().idxmax()
        else:
            min_corr = masked_corr.min().min()
            if pd.isna(min_corr):
                return None
            
            # Encontra posição do valor mínimo
            max_pos = masked_corr.stack().idxmin()
            max_corr = min_corr
        
        return {
            'variables': list(max_pos),
            'correlation': float(max_corr),
            'strength': self._interpret_correlation(max_corr)
        }
    
    def _find_category_with_highest_median(self, df, cat_col, num_col) -> str:
        """
        Encontra categoria com maior mediana
        """
        medians = df.groupby(cat_col)[num_col].median()
        return str(medians.idxmax())

# Funções auxiliares para importar numpy no escopo global
import numpy as np