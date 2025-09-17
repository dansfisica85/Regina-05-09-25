"""
Gerador Avançado de Visualizações
Sistema inteligente para criar gráficos interativos usando Plotly
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Dict, List, Any, Optional, Tuple
import json
import base64
import io
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AdvancedVisualizationGenerator:
    """Gerador avançado de visualizações interativas"""
    
    def __init__(self):
        self.color_palettes = {
            'default': px.colors.qualitative.Set1,
            'viridis': px.colors.sequential.Viridis,
            'plasma': px.colors.sequential.Plasma,
            'categorical': px.colors.qualitative.Plotly,
            'diverging': px.colors.diverging.RdBu
        }
        
        self.chart_configs = {
            'histogram': self._create_histogram,
            'box_plot': self._create_box_plot,
            'scatter_plot': self._create_scatter_plot,
            'line_chart': self._create_line_chart,
            'bar_chart': self._create_bar_chart,
            'pie_chart': self._create_pie_chart,
            'heatmap': self._create_heatmap,
            'correlation_matrix': self._create_correlation_matrix,
            'violin_plot': self._create_violin_plot,
            'density_plot': self._create_density_plot,
            'parallel_coordinates': self._create_parallel_coordinates,
            'radar_chart': self._create_radar_chart,
            'treemap': self._create_treemap,
            'sunburst': self._create_sunburst,
            '3d_scatter': self._create_3d_scatter,
            'animated_scatter': self._create_animated_scatter,
            'statistical_summary': self._create_statistical_summary,
            'distribution_comparison': self._create_distribution_comparison
        }
    
    def generate_visualization(self, df: pd.DataFrame, chart_config: Dict[str, Any]) -> Dict[str, Any]:
        """Gera visualização baseada na configuração"""
        
        try:
            chart_type = chart_config.get('chart_type', 'bar_chart')
            
            if chart_type not in self.chart_configs:
                raise ValueError(f"Tipo de gráfico não suportado: {chart_type}")
            
            # Gera o gráfico
            fig = self.chart_configs[chart_type](df, chart_config)
            
            # Aplica configurações gerais
            self._apply_general_styling(fig, chart_config)
            
            # Converte para JSON
            fig_json = fig.to_json()
            
            return {
                'success': True,
                'figure_json': fig_json,
                'chart_type': chart_type,
                'title': chart_config.get('title', 'Gráfico'),
                'description': chart_config.get('description', ''),
                'config': chart_config
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'chart_type': chart_config.get('chart_type', 'unknown')
            }
    
    def generate_dashboard(self, df: pd.DataFrame, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Gera dashboard com múltiplas visualizações"""
        
        try:
            # Cria subplots baseado no número de recomendações
            n_charts = min(len(recommendations), 6)  # Máximo 6 gráficos
            
            if n_charts <= 2:
                rows, cols = 1, n_charts
            elif n_charts <= 4:
                rows, cols = 2, 2
            else:
                rows, cols = 2, 3
            
            # Cria subplots
            subplot_titles = [rec.get('title', f'Chart {i+1}') for i, rec in enumerate(recommendations[:n_charts])]
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=subplot_titles,
                specs=[[{"secondary_y": False} for _ in range(cols)] for _ in range(rows)]
            )
            
            # Adiciona cada gráfico
            for i, recommendation in enumerate(recommendations[:n_charts]):
                row = (i // cols) + 1
                col = (i % cols) + 1
                
                try:
                    # Gera gráfico individual
                    single_fig = self.chart_configs[recommendation['chart_type']](df, recommendation)
                    
                    # Adiciona traces ao subplot
                    for trace in single_fig.data:
                        fig.add_trace(trace, row=row, col=col)
                        
                except Exception as e:
                    print(f"Erro ao adicionar gráfico {i}: {str(e)}")
            
            # Aplica layout geral
            fig.update_layout(
                title_text="Dashboard de Análise de Dados",
                showlegend=True,
                height=600 * rows,
                template="plotly_white"
            )
            
            return {
                'success': True,
                'figure_json': fig.to_json(),
                'chart_count': n_charts,
                'title': 'Dashboard Interativo'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def _create_histogram(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Cria histograma"""
        columns = config.get('columns', [])
        if not columns:
            raise ValueError("Colunas não especificadas para histograma")
        
        column = columns[0]
        bins = config.get('config', {}).get('bins', 30)
        
        fig = px.histogram(
            df, 
            x=column,
            nbins=bins,
            title=config.get('title', f'Distribuição de {column}'),
            template="plotly_white"
        )
        
        # Adiciona estatísticas
        mean_val = df[column].mean()
        median_val = df[column].median()
        
        fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                     annotation_text=f"Média: {mean_val:.2f}")
        fig.add_vline(x=median_val, line_dash="dash", line_color="blue", 
                     annotation_text=f"Mediana: {median_val:.2f}")
        
        return fig
    
    def _create_box_plot(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Cria box plot"""
        columns = config.get('columns', [])
        if not columns:
            raise ValueError("Colunas não especificadas para box plot")
        
        if len(columns) == 1:
            # Box plot simples
            fig = px.box(
                df, 
                y=columns[0],
                title=config.get('title', f'Box Plot de {columns[0]}'),
                template="plotly_white"
            )
        else:
            # Box plot agrupado
            fig = px.box(
                df, 
                x=columns[1] if len(columns) > 1 else None,
                y=columns[0],
                title=config.get('title', f'{columns[0]} por {columns[1]}'),
                template="plotly_white"
            )
        
        return fig
    
    def _create_scatter_plot(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Cria scatter plot"""
        columns = config.get('columns', [])
        if len(columns) < 2:
            raise ValueError("Pelo menos 2 colunas necessárias para scatter plot")
        
        x_col = config.get('config', {}).get('x', columns[0])
        y_col = config.get('config', {}).get('y', columns[1])
        color_col = columns[2] if len(columns) > 2 else None
        size_col = columns[3] if len(columns) > 3 else None
        
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            size=size_col,
            title=config.get('title', f'{x_col} vs {y_col}'),
            template="plotly_white",
            trendline="ols" if config.get('show_trendline', True) else None
        )
        
        # Adiciona informações de correlação
        if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
            corr = df[x_col].corr(df[y_col])
            fig.add_annotation(
                x=0.02, y=0.98,
                xref="paper", yref="paper",
                text=f"Correlação: {corr:.3f}",
                showarrow=False,
                bgcolor="white",
                bordercolor="black"
            )
        
        return fig
    
    def _create_line_chart(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Cria gráfico de linha"""
        columns = config.get('columns', [])
        if len(columns) < 2:
            raise ValueError("Pelo menos 2 colunas necessárias para line chart")
        
        x_col = config.get('config', {}).get('x', columns[0])
        y_col = config.get('config', {}).get('y', columns[1])
        
        fig = px.line(
            df,
            x=x_col,
            y=y_col,
            title=config.get('title', f'{y_col} ao longo de {x_col}'),
            template="plotly_white"
        )
        
        # Adiciona marcadores
        fig.update_traces(mode='lines+markers')
        
        return fig
    
    def _create_bar_chart(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Cria gráfico de barras"""
        columns = config.get('columns', [])
        if not columns:
            raise ValueError("Colunas não especificadas para bar chart")
        
        # Calcula contagens ou usa valores diretos
        if len(columns) == 1:
            # Gráfico de frequência
            value_counts = df[columns[0]].value_counts().head(20)  # Top 20
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=config.get('title', f'Distribuição de {columns[0]}'),
                template="plotly_white"
            )
            fig.update_xaxes(title=columns[0])
            fig.update_yaxes(title='Frequência')
        else:
            # Gráfico com valores específicos
            fig = px.bar(
                df,
                x=columns[0],
                y=columns[1] if len(columns) > 1 else None,
                color=columns[2] if len(columns) > 2 else None,
                title=config.get('title', f'{columns[1]} por {columns[0]}'),
                template="plotly_white"
            )
        
        return fig
    
    def _create_pie_chart(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Cria gráfico de pizza"""
        columns = config.get('columns', [])
        if not columns:
            raise ValueError("Colunas não especificadas para pie chart")
        
        column = columns[0]
        value_counts = df[column].value_counts().head(10)  # Top 10
        
        fig = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title=config.get('title', f'Distribuição de {column}'),
            template="plotly_white"
        )
        
        return fig
    
    def _create_heatmap(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Cria heatmap"""
        columns = config.get('columns', [])
        
        if len(columns) >= 2:
            # Heatmap de contingência
            contingency = pd.crosstab(df[columns[0]], df[columns[1]])
            
            fig = px.imshow(
                contingency,
                title=config.get('title', f'Heatmap: {columns[0]} vs {columns[1]}'),
                template="plotly_white",
                aspect="auto"
            )
        else:
            # Heatmap de dados numéricos
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:10]
            
            fig = px.imshow(
                df[numeric_cols].corr(),
                title=config.get('title', 'Heatmap de Correlação'),
                template="plotly_white",
                aspect="auto"
            )
        
        return fig
    
    def _create_correlation_matrix(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Cria matriz de correlação"""
        columns = config.get('columns', [])
        if columns:
            numeric_data = df[columns].select_dtypes(include=[np.number])
        else:
            numeric_data = df.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            raise ValueError("Nenhuma coluna numérica encontrada")
        
        # Calcula correlação
        corr_matrix = numeric_data.corr()
        
        # Cria heatmap
        fig = px.imshow(
            corr_matrix,
            title=config.get('title', 'Matriz de Correlação'),
            template="plotly_white",
            aspect="auto",
            color_continuous_scale="RdBu",
            zmin=-1, zmax=1
        )
        
        # Adiciona valores de correlação
        fig.update_traces(
            text=np.around(corr_matrix.values, decimals=2),
            texttemplate="%{text}",
            textfont={"size": 10}
        )
        
        return fig
    
    def _create_violin_plot(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Cria violin plot"""
        columns = config.get('columns', [])
        if not columns:
            raise ValueError("Colunas não especificadas para violin plot")
        
        if len(columns) == 1:
            fig = px.violin(
                df,
                y=columns[0],
                title=config.get('title', f'Violin Plot de {columns[0]}'),
                template="plotly_white"
            )
        else:
            fig = px.violin(
                df,
                x=columns[1],
                y=columns[0],
                title=config.get('title', f'{columns[0]} por {columns[1]}'),
                template="plotly_white"
            )
        
        return fig
    
    def _create_density_plot(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Cria gráfico de densidade"""
        columns = config.get('columns', [])
        if not columns:
            raise ValueError("Colunas não especificadas para density plot")
        
        column = columns[0]
        
        # Remove valores nulos
        clean_data = df[column].dropna()
        
        # Cria densidade
        hist_data = [clean_data.values]
        group_labels = [column]
        
        fig = ff.create_distplot(hist_data, group_labels, show_hist=False)
        
        fig.update_layout(
            title=config.get('title', f'Densidade de {column}'),
            template="plotly_white"
        )
        
        return fig
    
    def _create_parallel_coordinates(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Cria gráfico de coordenadas paralelas"""
        columns = config.get('columns', [])
        if len(columns) < 3:
            # Usa todas as colunas numéricas
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            columns = numeric_cols[:6]  # Máximo 6 colunas
        
        if len(columns) < 3:
            raise ValueError("Pelo menos 3 colunas numéricas necessárias")
        
        fig = px.parallel_coordinates(
            df,
            dimensions=columns,
            title=config.get('title', 'Coordenadas Paralelas'),
            template="plotly_white"
        )
        
        return fig
    
    def _create_radar_chart(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Cria gráfico radar"""
        columns = config.get('columns', [])
        if not columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            columns = numeric_cols[:6]  # Máximo 6 dimensões
        
        if len(columns) < 3:
            raise ValueError("Pelo menos 3 colunas necessárias para radar chart")
        
        # Calcula médias ou usa primeira linha
        if len(df) > 1:
            values = df[columns].mean().values
        else:
            values = df[columns].iloc[0].values
        
        # Normaliza valores (0-1)
        values_norm = (values - values.min()) / (values.max() - values.min())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values_norm,
            theta=columns,
            fill='toself',
            name='Valores'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title=config.get('title', 'Gráfico Radar'),
            template="plotly_white"
        )
        
        return fig
    
    def _create_treemap(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Cria treemap"""
        columns = config.get('columns', [])
        if len(columns) < 2:
            raise ValueError("Pelo menos 2 colunas necessárias para treemap")
        
        # Agrupa dados
        grouped = df.groupby(columns[0]).size().reset_index(name='count')
        
        fig = px.treemap(
            grouped,
            path=[columns[0]],
            values='count',
            title=config.get('title', f'Treemap de {columns[0]}'),
            template="plotly_white"
        )
        
        return fig
    
    def _create_sunburst(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Cria gráfico sunburst"""
        columns = config.get('columns', [])
        if len(columns) < 2:
            raise ValueError("Pelo menos 2 colunas necessárias para sunburst")
        
        # Agrupa dados hierarquicamente
        path_columns = columns[:3]  # Máximo 3 níveis
        grouped = df.groupby(path_columns).size().reset_index(name='count')
        
        fig = px.sunburst(
            grouped,
            path=path_columns,
            values='count',
            title=config.get('title', 'Gráfico Sunburst'),
            template="plotly_white"
        )
        
        return fig
    
    def _create_3d_scatter(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Cria scatter plot 3D"""
        columns = config.get('columns', [])
        if len(columns) < 3:
            raise ValueError("Pelo menos 3 colunas necessárias para 3D scatter")
        
        fig = px.scatter_3d(
            df,
            x=columns[0],
            y=columns[1],
            z=columns[2],
            color=columns[3] if len(columns) > 3 else None,
            size=columns[4] if len(columns) > 4 else None,
            title=config.get('title', 'Scatter Plot 3D'),
            template="plotly_white"
        )
        
        return fig
    
    def _create_animated_scatter(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Cria scatter plot animado"""
        columns = config.get('columns', [])
        if len(columns) < 3:
            raise ValueError("Pelo menos 3 colunas necessárias para scatter animado")
        
        # Assume que a terceira coluna é temporal ou categórica
        animation_frame = columns[2]
        
        fig = px.scatter(
            df,
            x=columns[0],
            y=columns[1],
            animation_frame=animation_frame,
            title=config.get('title', 'Scatter Plot Animado'),
            template="plotly_white"
        )
        
        return fig
    
    def _create_statistical_summary(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Cria resumo estatístico visual"""
        columns = config.get('columns', [])
        if not columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            columns = numeric_cols[:5]  # Máximo 5 colunas
        
        if not columns:
            raise ValueError("Nenhuma coluna numérica encontrada")
        
        # Calcula estatísticas
        stats_df = df[columns].describe().T
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Médias', 'Desvios Padrão', 'Mínimos e Máximos', 'Quartis'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Gráfico 1: Médias
        fig.add_trace(
            go.Bar(x=stats_df.index, y=stats_df['mean'], name='Média'),
            row=1, col=1
        )
        
        # Gráfico 2: Desvios padrão
        fig.add_trace(
            go.Bar(x=stats_df.index, y=stats_df['std'], name='Desvio Padrão'),
            row=1, col=2
        )
        
        # Gráfico 3: Min e Max
        fig.add_trace(
            go.Scatter(x=stats_df.index, y=stats_df['min'], name='Mínimo', mode='markers+lines'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=stats_df.index, y=stats_df['max'], name='Máximo', mode='markers+lines'),
            row=2, col=1
        )
        
        # Gráfico 4: Quartis
        fig.add_trace(
            go.Bar(x=stats_df.index, y=stats_df['25%'], name='Q1'),
            row=2, col=2
        )
        fig.add_trace(
            go.Bar(x=stats_df.index, y=stats_df['75%'], name='Q3'),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text=config.get('title', 'Resumo Estatístico'),
            showlegend=True,
            height=600,
            template="plotly_white"
        )
        
        return fig
    
    def _create_distribution_comparison(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Cria comparação de distribuições"""
        columns = config.get('columns', [])
        if len(columns) < 2:
            raise ValueError("Pelo menos 2 colunas necessárias para comparação")
        
        numeric_col = columns[0]
        category_col = columns[1]
        
        # Cria distribuições por categoria
        categories = df[category_col].unique()[:5]  # Máximo 5 categorias
        
        fig = go.Figure()
        
        for category in categories:
            data = df[df[category_col] == category][numeric_col].dropna()
            
            fig.add_trace(go.Histogram(
                x=data,
                name=str(category),
                opacity=0.7,
                nbinsx=20
            ))
        
        fig.update_layout(
            title=config.get('title', f'Distribuição de {numeric_col} por {category_col}'),
            barmode='overlay',
            template="plotly_white"
        )
        
        return fig
    
    def _apply_general_styling(self, fig: go.Figure, config: Dict[str, Any]):
        """Aplica estilização geral ao gráfico"""
        
        # Configurações de layout
        fig.update_layout(
            font=dict(size=12),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=80, b=50),
            hovermode='closest'
        )
        
        # Adiciona descrição se disponível
        description = config.get('description', '')
        if description:
            fig.add_annotation(
                text=description,
                xref="paper", yref="paper",
                x=0, y=-0.1,
                xanchor='left', yanchor='top',
                showarrow=False,
                font=dict(size=10, color="gray")
            )
        
        # Configurações responsivas
        fig.update_layout(
            autosize=True,
            responsive=True
        )