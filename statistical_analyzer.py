"""
Módulo de Análise Estatística Automatizada
Sistema avançado para análises estatísticas, tendências e insights automáticos
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import normaltest, shapiro, anderson, kstest
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

class AutomatedStatisticalAnalyzer:
    """Analisador estatístico automatizado com ML"""
    
    def __init__(self):
        self.significance_level = 0.05
        self.confidence_level = 0.95
        
    def comprehensive_analysis(self, df: pd.DataFrame, column_types: Dict[str, Dict]) -> Dict[str, Any]:
        """Executa análise estatística completa"""
        
        analysis_results = {
            'descriptive_stats': self._descriptive_statistics(df, column_types),
            'normality_tests': self._test_normality(df, column_types),
            'correlation_analysis': self._correlation_analysis(df, column_types),
            'outlier_analysis': self._outlier_detection(df, column_types),
            'trend_analysis': self._trend_analysis(df, column_types),
            'clustering_analysis': self._clustering_analysis(df, column_types),
            'dimensionality_analysis': self._dimensionality_reduction(df, column_types),
            'regression_analysis': self._automated_regression(df, column_types),
            'time_series_analysis': self._time_series_analysis(df, column_types),
            'hypothesis_tests': self._automated_hypothesis_tests(df, column_types),
            'insights': [],
            'recommendations': []
        }
        
        # Gera insights baseados nos resultados
        analysis_results['insights'] = self._generate_statistical_insights(analysis_results)
        analysis_results['recommendations'] = self._generate_recommendations(analysis_results)
        
        return analysis_results
    
    def _descriptive_statistics(self, df: pd.DataFrame, column_types: Dict[str, Dict]) -> Dict[str, Any]:
        """Estatísticas descritivas avançadas"""
        
        results = {}
        
        # Estatísticas para colunas numéricas
        numeric_cols = [col for col, info in column_types.items() if info['type'] == 'numeric']
        
        if numeric_cols:
            numeric_df = df[numeric_cols].select_dtypes(include=[np.number])
            
            results['numeric'] = {
                'basic_stats': numeric_df.describe().to_dict(),
                'additional_stats': {}
            }
            
            for col in numeric_df.columns:
                series = numeric_df[col].dropna()
                if len(series) > 0:
                    results['numeric']['additional_stats'][col] = {
                        'skewness': float(stats.skew(series)),
                        'kurtosis': float(stats.kurtosis(series)),
                        'coefficient_of_variation': float(series.std() / series.mean()) if series.mean() != 0 else 0,
                        'range': float(series.max() - series.min()),
                        'iqr': float(series.quantile(0.75) - series.quantile(0.25)),
                        'mad': float(stats.median_abs_deviation(series)),  # Median Absolute Deviation
                        'mode': float(stats.mode(series, keepdims=False)[0]) if len(series) > 0 else None,
                        'geometric_mean': float(stats.gmean(series[series > 0])) if (series > 0).any() else None,
                        'harmonic_mean': float(stats.hmean(series[series > 0])) if (series > 0).any() else None
                    }
        
        # Estatísticas para colunas categóricas
        categorical_cols = [col for col, info in column_types.items() if info['type'] == 'categorical']
        
        if categorical_cols:
            results['categorical'] = {}
            
            for col in categorical_cols:
                if col in df.columns:
                    series = df[col].dropna()
                    value_counts = series.value_counts()
                    
                    results['categorical'][col] = {
                        'unique_count': int(series.nunique()),
                        'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                        'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                        'least_frequent': str(value_counts.index[-1]) if len(value_counts) > 0 else None,
                        'least_frequent_count': int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0,
                        'frequency_distribution': value_counts.head(10).to_dict(),
                        'concentration_ratio': float(value_counts.iloc[0] / len(series)) if len(series) > 0 else 0,
                        'entropy': float(-np.sum((value_counts / len(series)) * np.log2(value_counts / len(series))))
                    }
        
        return results
    
    def _test_normality(self, df: pd.DataFrame, column_types: Dict[str, Dict]) -> Dict[str, Any]:
        """Testes de normalidade múltiplos"""
        
        results = {}
        numeric_cols = [col for col, info in column_types.items() if info['type'] == 'numeric']
        
        for col in numeric_cols:
            if col in df.columns:
                series = df[col].dropna()
                
                if len(series) < 8:  # Mínimo para testes
                    continue
                
                results[col] = {}
                
                try:
                    # Teste de Shapiro-Wilk (melhor para n < 5000)
                    if len(series) <= 5000:
                        shapiro_stat, shapiro_p = shapiro(series)
                        results[col]['shapiro_wilk'] = {
                            'statistic': float(shapiro_stat),
                            'p_value': float(shapiro_p),
                            'is_normal': shapiro_p > self.significance_level
                        }
                    
                    # Teste de D'Agostino-Pearson
                    if len(series) >= 20:
                        dagostino_stat, dagostino_p = normaltest(series)
                        results[col]['dagostino_pearson'] = {
                            'statistic': float(dagostino_stat),
                            'p_value': float(dagostino_p),
                            'is_normal': dagostino_p > self.significance_level
                        }
                    
                    # Teste de Anderson-Darling
                    anderson_result = anderson(series, dist='norm')
                    critical_value = anderson_result.critical_values[2]  # 5% significance
                    results[col]['anderson_darling'] = {
                        'statistic': float(anderson_result.statistic),
                        'critical_value': float(critical_value),
                        'is_normal': anderson_result.statistic < critical_value
                    }
                    
                    # Teste de Kolmogorov-Smirnov
                    # Padroniza os dados
                    standardized = (series - series.mean()) / series.std()
                    ks_stat, ks_p = kstest(standardized, 'norm')
                    results[col]['kolmogorov_smirnov'] = {
                        'statistic': float(ks_stat),
                        'p_value': float(ks_p),
                        'is_normal': ks_p > self.significance_level
                    }
                    
                    # Consenso
                    normal_tests = [
                        results[col].get('shapiro_wilk', {}).get('is_normal', False),
                        results[col].get('dagostino_pearson', {}).get('is_normal', False),
                        results[col].get('anderson_darling', {}).get('is_normal', False),
                        results[col].get('kolmogorov_smirnov', {}).get('is_normal', False)
                    ]
                    results[col]['consensus'] = {
                        'normal_tests_passed': sum(normal_tests),
                        'total_tests': len([t for t in normal_tests if t is not False]),
                        'likely_normal': sum(normal_tests) >= len(normal_tests) / 2
                    }
                    
                except Exception as e:
                    results[col]['error'] = str(e)
        
        return results
    
    def _correlation_analysis(self, df: pd.DataFrame, column_types: Dict[str, Dict]) -> Dict[str, Any]:
        """Análise de correlação avançada"""
        
        results = {}
        numeric_cols = [col for col, info in column_types.items() if info['type'] == 'numeric']
        
        if len(numeric_cols) < 2:
            return results
        
        numeric_df = df[numeric_cols].select_dtypes(include=[np.number])
        
        # Correlação de Pearson
        pearson_corr = numeric_df.corr(method='pearson')
        results['pearson'] = {
            'matrix': pearson_corr.to_dict(),
            'strong_correlations': self._find_strong_correlations(pearson_corr, 0.7)
        }
        
        # Correlação de Spearman (não-paramétrica)
        spearman_corr = numeric_df.corr(method='spearman')
        results['spearman'] = {
            'matrix': spearman_corr.to_dict(),
            'strong_correlations': self._find_strong_correlations(spearman_corr, 0.7)
        }
        
        # Correlação de Kendall
        kendall_corr = numeric_df.corr(method='kendall')
        results['kendall'] = {
            'matrix': kendall_corr.to_dict(),
            'strong_correlations': self._find_strong_correlations(kendall_corr, 0.7)
        }
        
        # Análise de correlação parcial
        results['partial_correlations'] = self._partial_correlations(numeric_df)
        
        return results
    
    def _find_strong_correlations(self, corr_matrix: pd.DataFrame, threshold: float) -> List[Dict[str, Any]]:
        """Encontra correlações fortes"""
        
        strong_corrs = []
        
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns[i+1:], i+1):
                corr_value = corr_matrix.loc[col1, col2]
                
                if abs(corr_value) >= threshold:
                    strong_corrs.append({
                        'variable_1': col1,
                        'variable_2': col2,
                        'correlation': float(corr_value),
                        'strength': 'very_strong' if abs(corr_value) > 0.9 else 'strong',
                        'direction': 'positive' if corr_value > 0 else 'negative'
                    })
        
        return sorted(strong_corrs, key=lambda x: abs(x['correlation']), reverse=True)
    
    def _partial_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcula correlações parciais"""
        
        try:
            from scipy.stats import pearsonr
            
            partial_corrs = {}
            cols = df.columns.tolist()
            
            if len(cols) >= 3:
                # Para cada par de variáveis, calcula correlação controlando pelas outras
                for i, col1 in enumerate(cols):
                    for j, col2 in enumerate(cols[i+1:], i+1):
                        # Controla pelas outras variáveis
                        control_vars = [c for c in cols if c not in [col1, col2]]
                        
                        if control_vars:
                            # Regressão linear para remover efeito das variáveis de controle
                            X_control = df[control_vars].fillna(0)
                            
                            # Resíduos de col1
                            y1 = df[col1].fillna(0)
                            reg1 = LinearRegression().fit(X_control, y1)
                            residuals1 = y1 - reg1.predict(X_control)
                            
                            # Resíduos de col2
                            y2 = df[col2].fillna(0)
                            reg2 = LinearRegression().fit(X_control, y2)
                            residuals2 = y2 - reg2.predict(X_control)
                            
                            # Correlação dos resíduos
                            partial_corr, p_value = pearsonr(residuals1, residuals2)
                            
                            partial_corrs[f"{col1}__{col2}"] = {
                                'correlation': float(partial_corr),
                                'p_value': float(p_value),
                                'controlled_variables': control_vars,
                                'significant': p_value < self.significance_level
                            }
            
            return partial_corrs
            
        except Exception as e:
            return {'error': str(e)}
    
    def _outlier_detection(self, df: pd.DataFrame, column_types: Dict[str, Dict]) -> Dict[str, Any]:
        """Detecção de outliers com múltiplos métodos"""
        
        results = {}
        numeric_cols = [col for col, info in column_types.items() if info['type'] == 'numeric']
        
        for col in numeric_cols:
            if col in df.columns:
                series = df[col].dropna()
                
                if len(series) < 10:
                    continue
                
                results[col] = {}
                
                # Método IQR
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                iqr_outliers = series[(series < lower_bound) | (series > upper_bound)]
                results[col]['iqr_method'] = {
                    'outlier_count': len(iqr_outliers),
                    'outlier_percentage': len(iqr_outliers) / len(series) * 100,
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'outlier_values': iqr_outliers.tolist()[:10]  # Primeiros 10
                }
                
                # Método Z-Score
                z_scores = np.abs(stats.zscore(series))
                z_outliers = series[z_scores > 3]
                results[col]['zscore_method'] = {
                    'outlier_count': len(z_outliers),
                    'outlier_percentage': len(z_outliers) / len(series) * 100,
                    'threshold': 3.0,
                    'outlier_values': z_outliers.tolist()[:10]
                }
                
                # Método Modified Z-Score
                median = series.median()
                mad = stats.median_abs_deviation(series)
                modified_z_scores = 0.6745 * (series - median) / mad if mad != 0 else np.zeros_like(series)
                modified_z_outliers = series[np.abs(modified_z_scores) > 3.5]
                results[col]['modified_zscore_method'] = {
                    'outlier_count': len(modified_z_outliers),
                    'outlier_percentage': len(modified_z_outliers) / len(series) * 100,
                    'threshold': 3.5,
                    'outlier_values': modified_z_outliers.tolist()[:10]
                }
                
                # Isolation Forest
                try:
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outlier_labels = iso_forest.fit_predict(series.values.reshape(-1, 1))
                    isolation_outliers = series[outlier_labels == -1]
                    results[col]['isolation_forest'] = {
                        'outlier_count': len(isolation_outliers),
                        'outlier_percentage': len(isolation_outliers) / len(series) * 100,
                        'outlier_values': isolation_outliers.tolist()[:10]
                    }
                except Exception as e:
                    results[col]['isolation_forest'] = {'error': str(e)}
        
        return results
    
    def _trend_analysis(self, df: pd.DataFrame, column_types: Dict[str, Dict]) -> Dict[str, Any]:
        """Análise de tendências temporais"""
        
        results = {}
        
        # Procura por colunas de data/tempo
        date_cols = [col for col, info in column_types.items() if info['type'] == 'date']
        numeric_cols = [col for col, info in column_types.items() if info['type'] == 'numeric']
        
        # Se não há colunas de data explícitas, tenta detectar
        if not date_cols:
            date_cols = self._detect_date_columns(df)
        
        if date_cols and numeric_cols:
            for date_col in date_cols[:1]:  # Pega a primeira coluna de data
                for numeric_col in numeric_cols[:3]:  # Máximo 3 colunas numéricas
                    
                    if date_col in df.columns and numeric_col in df.columns:
                        try:
                            # Prepara dados temporais
                            temp_df = df[[date_col, numeric_col]].copy()
                            temp_df = temp_df.dropna()
                            
                            # Converte data se necessário
                            if not pd.api.types.is_datetime64_any_dtype(temp_df[date_col]):
                                temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors='coerce')
                            
                            temp_df = temp_df.dropna()
                            
                            if len(temp_df) < 10:
                                continue
                            
                            # Ordena por data
                            temp_df = temp_df.sort_values(date_col)
                            
                            # Análise de tendência
                            results[f"{numeric_col}_over_{date_col}"] = self._analyze_time_trend(
                                temp_df[date_col], temp_df[numeric_col]
                            )
                            
                        except Exception as e:
                            results[f"{numeric_col}_over_{date_col}"] = {'error': str(e)}
        
        return results
    
    def _detect_date_columns(self, df: pd.DataFrame) -> List[str]:
        """Detecta colunas que podem ser datas"""
        
        date_cols = []
        
        for col in df.columns:
            # Tenta converter uma amostra para data
            sample = df[col].dropna().head(20)
            
            try:
                pd.to_datetime(sample, errors='raise')
                date_cols.append(col)
            except:
                # Verifica padrões de data em strings
                if sample.dtype == 'object':
                    date_patterns = [
                        r'\d{4}-\d{2}-\d{2}',
                        r'\d{2}/\d{2}/\d{4}',
                        r'\d{2}-\d{2}-\d{4}'
                    ]
                    
                    import re
                    for pattern in date_patterns:
                        matches = sample.astype(str).str.contains(pattern, regex=True).sum()
                        if matches > len(sample) * 0.7:  # 70% coincidem
                            date_cols.append(col)
                            break
        
        return date_cols
    
    def _analyze_time_trend(self, dates: pd.Series, values: pd.Series) -> Dict[str, Any]:
        """Analisa tendência temporal"""
        
        # Converte datas para números para regressão
        date_numeric = pd.to_numeric(dates)
        
        # Regressão linear
        X = date_numeric.values.reshape(-1, 1)
        y = values.values
        
        reg = LinearRegression().fit(X, y)
        trend_slope = reg.coef_[0]
        r2 = r2_score(y, reg.predict(X))
        
        # Teste de significância da tendência
        from scipy.stats import pearsonr
        corr, p_value = pearsonr(date_numeric, values)
        
        # Classificação da tendência
        if abs(trend_slope) < values.std() * 0.1:
            trend_type = 'stable'
        elif trend_slope > 0:
            trend_type = 'increasing'
        else:
            trend_type = 'decreasing'
        
        # Análise de sazonalidade (se há dados suficientes)
        seasonality = {}
        if len(values) >= 24:  # Pelo menos 2 ciclos completos
            try:
                # Tenta decomposição sazonal
                ts_series = pd.Series(values.values, index=dates)
                ts_series = ts_series.resample('D').mean()  # Agrupa por dia
                
                if len(ts_series) >= 14:  # Pelo menos 2 semanas
                    decomposition = seasonal_decompose(ts_series, model='additive', period=7)
                    
                    seasonality = {
                        'seasonal_strength': float(np.var(decomposition.seasonal) / np.var(values)),
                        'trend_strength': float(np.var(decomposition.trend.dropna()) / np.var(values)),
                        'residual_strength': float(np.var(decomposition.resid.dropna()) / np.var(values))
                    }
            except Exception as e:
                seasonality = {'error': str(e)}
        
        return {
            'trend_type': trend_type,
            'slope': float(trend_slope),
            'r_squared': float(r2),
            'correlation': float(corr),
            'p_value': float(p_value),
            'significant_trend': p_value < self.significance_level,
            'seasonality': seasonality,
            'data_points': len(values),
            'time_span_days': (dates.max() - dates.min()).days if len(dates) > 1 else 0
        }
    
    def _clustering_analysis(self, df: pd.DataFrame, column_types: Dict[str, Dict]) -> Dict[str, Any]:
        """Análise de clustering"""
        
        results = {}
        numeric_cols = [col for col, info in column_types.items() if info['type'] == 'numeric']
        
        if len(numeric_cols) < 2:
            return results
        
        # Prepara dados
        numeric_df = df[numeric_cols].select_dtypes(include=[np.number])
        numeric_df = numeric_df.fillna(numeric_df.mean())
        
        if len(numeric_df) < 10:
            return results
        
        # Padroniza dados
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # K-means com diferentes números de clusters
        optimal_clusters = self._find_optimal_clusters(scaled_data)
        
        # Executa K-means com número ótimo
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        # Análise dos clusters
        results['kmeans'] = {
            'optimal_clusters': optimal_clusters,
            'cluster_labels': cluster_labels.tolist(),
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'inertia': float(kmeans.inertia_),
            'cluster_summary': self._analyze_clusters(numeric_df, cluster_labels)
        }
        
        return results
    
    def _find_optimal_clusters(self, data: np.ndarray) -> int:
        """Encontra número ótimo de clusters usando método do cotovelo"""
        
        max_clusters = min(10, len(data) // 2)
        if max_clusters < 2:
            return 2
        
        inertias = []
        k_range = range(2, max_clusters + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
        
        # Método do cotovelo simples
        # Procura pelo maior decréscimo percentual
        decreases = [inertias[i-1] - inertias[i] for i in range(1, len(inertias))]
        optimal_idx = np.argmax(decreases)
        
        return k_range[optimal_idx]
    
    def _analyze_clusters(self, df: pd.DataFrame, labels: np.ndarray) -> Dict[str, Any]:
        """Analisa características dos clusters"""
        
        cluster_summary = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            cluster_data = df[labels == label]
            
            cluster_summary[f"cluster_{label}"] = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df) * 100,
                'means': cluster_data.mean().to_dict(),
                'stds': cluster_data.std().to_dict()
            }
        
        return cluster_summary
    
    def _dimensionality_reduction(self, df: pd.DataFrame, column_types: Dict[str, Dict]) -> Dict[str, Any]:
        """Análise de redução de dimensionalidade"""
        
        results = {}
        numeric_cols = [col for col, info in column_types.items() if info['type'] == 'numeric']
        
        if len(numeric_cols) < 3:
            return results
        
        # Prepara dados
        numeric_df = df[numeric_cols].select_dtypes(include=[np.number])
        numeric_df = numeric_df.fillna(numeric_df.mean())
        
        if len(numeric_df) < 10:
            return results
        
        # Padroniza dados
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # PCA
        try:
            pca = PCA()
            pca_data = pca.fit_transform(scaled_data)
            
            # Variância explicada
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            # Número de componentes para 90% da variância
            n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
            
            results['pca'] = {
                'explained_variance_ratio': explained_variance.tolist(),
                'cumulative_variance': cumulative_variance.tolist(),
                'components_for_90_percent': int(n_components_90),
                'total_components': len(explained_variance),
                'dimensionality_reduction_potential': (len(numeric_cols) - n_components_90) / len(numeric_cols),
                'principal_components': pca.components_[:3].tolist()  # Primeiros 3 componentes
            }
            
        except Exception as e:
            results['pca'] = {'error': str(e)}
        
        return results
    
    def _automated_regression(self, df: pd.DataFrame, column_types: Dict[str, Dict]) -> Dict[str, Any]:
        """Análise de regressão automatizada"""
        
        results = {}
        numeric_cols = [col for col, info in column_types.items() if info['type'] == 'numeric']
        
        if len(numeric_cols) < 2:
            return results
        
        # Para cada variável numérica, tenta prever usando as outras
        for target_col in numeric_cols:
            predictor_cols = [col for col in numeric_cols if col != target_col]
            
            if not predictor_cols:
                continue
            
            try:
                # Prepara dados
                X = df[predictor_cols].fillna(df[predictor_cols].mean())
                y = df[target_col].fillna(df[target_col].mean())
                
                # Remove linhas com NaN
                mask = ~(X.isnull().any(axis=1) | y.isnull())
                X = X[mask]
                y = y[mask]
                
                if len(X) < 10:
                    continue
                
                # Regressão linear múltipla
                reg = LinearRegression()
                reg.fit(X, y)
                
                # Predições
                y_pred = reg.predict(X)
                r2 = r2_score(y, y_pred)
                
                # Testes de diagnóstico
                residuals = y - y_pred
                
                # Teste de Durbin-Watson para autocorrelação
                dw_stat = durbin_watson(residuals)
                
                results[target_col] = {
                    'predictors': predictor_cols,
                    'r_squared': float(r2),
                    'coefficients': dict(zip(predictor_cols, reg.coef_)),
                    'intercept': float(reg.intercept_),
                    'durbin_watson': float(dw_stat),
                    'residual_mean': float(residuals.mean()),
                    'residual_std': float(residuals.std()),
                    'significant_predictors': []  # Seria necessário calcular p-values
                }
                
            except Exception as e:
                results[target_col] = {'error': str(e)}
        
        return results
    
    def _time_series_analysis(self, df: pd.DataFrame, column_types: Dict[str, Dict]) -> Dict[str, Any]:
        """Análise de séries temporais"""
        
        results = {}
        
        # Detecta colunas de data
        date_cols = [col for col, info in column_types.items() if info['type'] == 'date']
        if not date_cols:
            date_cols = self._detect_date_columns(df)
        
        numeric_cols = [col for col, info in column_types.items() if info['type'] == 'numeric']
        
        if not date_cols or not numeric_cols:
            return results
        
        for date_col in date_cols[:1]:  # Primeira coluna de data
            for numeric_col in numeric_cols[:2]:  # Primeiras 2 numéricas
                
                try:
                    # Prepara série temporal
                    ts_df = df[[date_col, numeric_col]].copy()
                    ts_df = ts_df.dropna()
                    
                    # Converte data
                    if not pd.api.types.is_datetime64_any_dtype(ts_df[date_col]):
                        ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors='coerce')
                    
                    ts_df = ts_df.dropna().sort_values(date_col)
                    
                    if len(ts_df) < 20:
                        continue
                    
                    # Cria índice temporal
                    ts_series = pd.Series(
                        ts_df[numeric_col].values, 
                        index=ts_df[date_col]
                    )
                    
                    # Análise básica da série
                    series_analysis = {
                        'length': len(ts_series),
                        'start_date': str(ts_series.index.min()),
                        'end_date': str(ts_series.index.max()),
                        'frequency': self._detect_frequency(ts_series),
                        'missing_values': ts_series.isnull().sum(),
                        'stationarity': self._test_stationarity(ts_series),
                        'autocorrelation': self._analyze_autocorrelation(ts_series)
                    }
                    
                    results[f"{numeric_col}_timeseries"] = series_analysis
                    
                except Exception as e:
                    results[f"{numeric_col}_timeseries"] = {'error': str(e)}
        
        return results
    
    def _detect_frequency(self, series: pd.Series) -> str:
        """Detecta frequência da série temporal"""
        
        if len(series) < 2:
            return 'unknown'
        
        # Calcula diferenças entre datas consecutivas
        time_diffs = series.index.to_series().diff().dropna()
        
        # Frequência mais comum
        most_common_diff = time_diffs.mode()[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
        
        # Classifica frequência
        if most_common_diff <= pd.Timedelta(days=1):
            return 'daily'
        elif most_common_diff <= pd.Timedelta(days=7):
            return 'weekly'
        elif most_common_diff <= pd.Timedelta(days=31):
            return 'monthly'
        elif most_common_diff <= pd.Timedelta(days=365):
            return 'yearly'
        else:
            return 'irregular'
    
    def _test_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        """Testa estacionariedade da série"""
        
        try:
            from statsmodels.tsa.stattools import adfuller
            
            # Teste Augmented Dickey-Fuller
            adf_result = adfuller(series.dropna())
            
            return {
                'adf_statistic': float(adf_result[0]),
                'adf_pvalue': float(adf_result[1]),
                'is_stationary': adf_result[1] < self.significance_level,
                'critical_values': {k: float(v) for k, v in adf_result[4].items()}
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_autocorrelation(self, series: pd.Series) -> Dict[str, Any]:
        """Analisa autocorrelação da série"""
        
        try:
            from statsmodels.tsa.stattools import acf, pacf
            
            # Calcula ACF e PACF
            autocorr = acf(series.dropna(), nlags=min(20, len(series)//4))
            partial_autocorr = pacf(series.dropna(), nlags=min(20, len(series)//4))
            
            return {
                'autocorrelation': autocorr.tolist(),
                'partial_autocorrelation': partial_autocorr.tolist(),
                'significant_lags': [i for i, val in enumerate(autocorr) if abs(val) > 0.2]
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _automated_hypothesis_tests(self, df: pd.DataFrame, column_types: Dict[str, Dict]) -> Dict[str, Any]:
        """Testes de hipótese automatizados"""
        
        results = {}
        
        # Testes para duas amostras
        numeric_cols = [col for col, info in column_types.items() if info['type'] == 'numeric']
        categorical_cols = [col for col, info in column_types.items() if info['type'] == 'categorical']
        
        # Teste t entre grupos
        if numeric_cols and categorical_cols:
            for numeric_col in numeric_cols[:2]:
                for cat_col in categorical_cols[:1]:
                    
                    if numeric_col in df.columns and cat_col in df.columns:
                        try:
                            # Pega as duas categorias mais frequentes
                            top_categories = df[cat_col].value_counts().head(2).index.tolist()
                            
                            if len(top_categories) == 2:
                                group1 = df[df[cat_col] == top_categories[0]][numeric_col].dropna()
                                group2 = df[df[cat_col] == top_categories[1]][numeric_col].dropna()
                                
                                if len(group1) >= 5 and len(group2) >= 5:
                                    # Teste t independente
                                    t_stat, t_p = stats.ttest_ind(group1, group2)
                                    
                                    # Teste Mann-Whitney (não-paramétrico)
                                    u_stat, u_p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
                                    
                                    results[f"{numeric_col}_by_{cat_col}"] = {
                                        'groups': top_categories,
                                        'group1_mean': float(group1.mean()),
                                        'group2_mean': float(group2.mean()),
                                        'group1_size': len(group1),
                                        'group2_size': len(group2),
                                        'ttest': {
                                            'statistic': float(t_stat),
                                            'p_value': float(t_p),
                                            'significant': t_p < self.significance_level
                                        },
                                        'mannwhitney': {
                                            'statistic': float(u_stat),
                                            'p_value': float(u_p),
                                            'significant': u_p < self.significance_level
                                        }
                                    }
                                    
                        except Exception as e:
                            results[f"{numeric_col}_by_{cat_col}"] = {'error': str(e)}
        
        return results
    
    def _generate_statistical_insights(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Gera insights baseados na análise estatística"""
        
        insights = []
        
        # Insights de normalidade
        normality = analysis_results.get('normality_tests', {})
        for col, tests in normality.items():
            consensus = tests.get('consensus', {})
            if not consensus.get('likely_normal', False):
                insights.append({
                    'type': 'statistical',
                    'category': 'normality',
                    'title': f'Distribuição Não-Normal em {col}',
                    'description': f'A variável "{col}" não segue distribuição normal. '
                                 f'Apenas {consensus.get("normal_tests_passed", 0)} de {consensus.get("total_tests", 0)} '
                                 'testes de normalidade foram aprovados.',
                    'importance': 0.7,
                    'recommendation': 'Considere transformações (log, sqrt) ou métodos não-paramétricos.',
                    'affected_columns': [col]
                })
        
        # Insights de correlação
        correlations = analysis_results.get('correlation_analysis', {})
        strong_corrs = correlations.get('pearson', {}).get('strong_correlations', [])
        if strong_corrs:
            top_corr = strong_corrs[0]
            insights.append({
                'type': 'statistical',
                'category': 'correlation',
                'title': 'Correlação Forte Detectada',
                'description': f'Correlação {top_corr["strength"]} ({top_corr["correlation"]:.3f}) '
                             f'entre "{top_corr["variable_1"]}" e "{top_corr["variable_2"]}".',
                'importance': 0.9,
                'recommendation': 'Investigate possível causalidade ou multicolinearidade.',
                'affected_columns': [top_corr["variable_1"], top_corr["variable_2"]]
            })
        
        # Insights de outliers
        outliers = analysis_results.get('outlier_analysis', {})
        for col, methods in outliers.items():
            iqr_outliers = methods.get('iqr_method', {}).get('outlier_percentage', 0)
            if iqr_outliers > 5:  # Mais de 5% são outliers
                insights.append({
                    'type': 'statistical',
                    'category': 'outliers',
                    'title': f'Muitos Outliers em {col}',
                    'description': f'A variável "{col}" contém {iqr_outliers:.1f}% outliers (método IQR).',
                    'importance': 0.8,
                    'recommendation': 'Analise se outliers são erros ou valores legítimos extremos.',
                    'affected_columns': [col]
                })
        
        # Insights de clustering
        clustering = analysis_results.get('clustering_analysis', {})
        if 'kmeans' in clustering:
            optimal_k = clustering['kmeans'].get('optimal_clusters', 0)
            if optimal_k > 1:
                insights.append({
                    'type': 'statistical',
                    'category': 'clustering',
                    'title': f'Estrutura de Grupos Detectada',
                    'description': f'Os dados podem ser agrupados em {optimal_k} clusters distintos.',
                    'importance': 0.7,
                    'recommendation': 'Analise as características de cada grupo para insights de segmentação.',
                    'affected_columns': []
                })
        
        return insights
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Gera recomendações baseadas na análise"""
        
        recommendations = []
        
        # Recomendações de qualidade de dados
        outliers = analysis_results.get('outlier_analysis', {})
        if outliers:
            recommendations.append({
                'category': 'data_quality',
                'title': 'Limpeza de Outliers',
                'description': 'Considere investigar e tratar outliers detectados.',
                'priority': 'high',
                'actions': [
                    'Visualize outliers em gráficos',
                    'Determine se são erros ou valores legítimos',
                    'Considere remoção ou transformação'
                ]
            })
        
        # Recomendações de modelagem
        normality = analysis_results.get('normality_tests', {})
        non_normal_vars = [col for col, tests in normality.items() 
                          if not tests.get('consensus', {}).get('likely_normal', True)]
        
        if non_normal_vars:
            recommendations.append({
                'category': 'modeling',
                'title': 'Uso de Métodos Não-Paramétricos',
                'description': f'Variáveis {", ".join(non_normal_vars)} não são normais.',
                'priority': 'medium',
                'actions': [
                    'Use testes não-paramétricos (Mann-Whitney, Kruskal-Wallis)',
                    'Considere transformações (log, Box-Cox)',
                    'Use algoritmos robustos a não-normalidade'
                ]
            })
        
        # Recomendações de visualização
        correlations = analysis_results.get('correlation_analysis', {})
        if correlations:
            recommendations.append({
                'category': 'visualization',
                'title': 'Visualização de Relacionamentos',
                'description': 'Crie visualizações para explorar correlações encontradas.',
                'priority': 'medium',
                'actions': [
                    'Crie matriz de correlação heatmap',
                    'Faça scatter plots das correlações fortes',
                    'Analise correlações parciais'
                ]
            })
        
        return recommendations