"""
Processador Universal de Planilhas
Sistema para importar e processar múltiplos formatos de planilhas com detecção automática
"""

import pandas as pd
import numpy as np
import io
import requests
from typing import Dict, List, Any, Optional, Tuple, Union
import os
from urllib.parse import urlparse
import tempfile
import zipfile
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Imports condicionais para Google Sheets
try:
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    GOOGLE_SHEETS_AVAILABLE = True
except ImportError:
    GOOGLE_SHEETS_AVAILABLE = False

class UniversalSpreadsheetProcessor:
    """Processador universal para diferentes formatos de planilhas"""
    
    def __init__(self):
        self.supported_formats = {
            'excel': ['.xlsx', '.xls', '.xlsm'],
            'csv': ['.csv', '.tsv', '.txt'],
            'google_sheets': ['google.com/spreadsheets', 'docs.google.com/spreadsheets'],
            'json': ['.json'],
            'parquet': ['.parquet'],
            'feather': ['.feather']
        }
        
        self.encoding_options = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        self.separator_options = [',', ';', '\t', '|']
        
    def process_file(self, file_input: Union[str, io.BytesIO, bytes], 
                    filename: Optional[str] = None,
                    options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Processa um arquivo de planilha e retorna dados estruturados
        
        Args:
            file_input: Caminho do arquivo, URL, ou dados binários
            filename: Nome do arquivo (para detecção de formato)
            options: Opções específicas de processamento
            
        Returns:
            Dict com dados processados e metadados
        """
        try:
            # Determina o tipo de entrada
            file_type = self._detect_input_type(file_input, filename)
            
            # Processa baseado no tipo
            if file_type == 'url_google_sheets':
                return self._process_google_sheets_url(file_input, options)
            elif file_type == 'url_file':
                return self._process_url_file(file_input, options)
            elif file_type == 'local_file':
                return self._process_local_file(file_input, options)
            elif file_type == 'binary_data':
                return self._process_binary_data(file_input, filename, options)
            else:
                raise ValueError(f"Tipo de entrada não suportado: {file_type}")
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'data': None,
                'metadata': {}
            }
    
    def _detect_input_type(self, file_input: Union[str, io.BytesIO, bytes], 
                          filename: Optional[str] = None) -> str:
        """Detecta o tipo de entrada"""
        
        if isinstance(file_input, str):
            if file_input.startswith(('http://', 'https://')):
                if any(sheets_url in file_input for sheets_url in self.supported_formats['google_sheets']):
                    return 'url_google_sheets'
                else:
                    return 'url_file'
            else:
                return 'local_file'
        elif isinstance(file_input, (io.BytesIO, bytes)):
            return 'binary_data'
        else:
            raise ValueError("Tipo de entrada não reconhecido")
    
    def _process_google_sheets_url(self, url: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Processa URL do Google Sheets"""
        if not GOOGLE_SHEETS_AVAILABLE:
            return {
                'success': False,
                'error': 'Google Sheets API não disponível. Instale: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib',
                'data': None,
                'metadata': {}
            }
        
        try:
            # Extrai sheet ID da URL
            sheet_id = self._extract_google_sheet_id(url)
            
            # Tenta acessar via CSV export (público)
            csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
            
            response = requests.get(csv_url)
            if response.status_code == 200:
                # Processa como CSV
                csv_data = io.StringIO(response.text)
                df = self._read_csv_with_detection(csv_data)
                
                return {
                    'success': True,
                    'data': df,
                    'metadata': {
                        'source_type': 'google_sheets',
                        'sheet_id': sheet_id,
                        'url': url,
                        'access_method': 'public_csv',
                        'rows': len(df),
                        'columns': len(df.columns)
                    }
                }
            else:
                return {
                    'success': False,
                    'error': 'Planilha não é pública ou não existe',
                    'data': None,
                    'metadata': {'sheet_id': sheet_id}
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Erro ao processar Google Sheets: {str(e)}",
                'data': None,
                'metadata': {}
            }
    
    def _extract_google_sheet_id(self, url: str) -> str:
        """Extrai ID da planilha da URL do Google Sheets"""
        import re
        
        # Padrões para extrair sheet ID
        patterns = [
            r'/spreadsheets/d/([a-zA-Z0-9-_]+)',
            r'key=([a-zA-Z0-9-_]+)',
            r'id=([a-zA-Z0-9-_]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        raise ValueError("Não foi possível extrair ID da planilha da URL")
    
    def _process_url_file(self, url: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Processa arquivo de URL"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Detecta formato pelo header ou URL
            content_type = response.headers.get('content-type', '')
            filename = urlparse(url).path.split('/')[-1]
            
            # Lê conteúdo
            content = response.content
            
            return self._process_binary_data(content, filename, options)
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Erro ao baixar arquivo: {str(e)}",
                'data': None,
                'metadata': {}
            }
    
    def _process_local_file(self, filepath: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Processa arquivo local"""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
            
            filename = os.path.basename(filepath)
            file_ext = os.path.splitext(filename)[1].lower()
            
            # Lê arquivo baseado na extensão
            if file_ext in self.supported_formats['excel']:
                return self._process_excel_file(filepath, options)
            elif file_ext in self.supported_formats['csv']:
                return self._process_csv_file(filepath, options)
            elif file_ext in self.supported_formats['json']:
                return self._process_json_file(filepath, options)
            elif file_ext in self.supported_formats['parquet']:
                return self._process_parquet_file(filepath, options)
            elif file_ext in self.supported_formats['feather']:
                return self._process_feather_file(filepath, options)
            else:
                # Tenta detectar automaticamente
                return self._auto_detect_and_process(filepath, options)
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Erro ao processar arquivo local: {str(e)}",
                'data': None,
                'metadata': {}
            }
    
    def _process_binary_data(self, data: Union[bytes, io.BytesIO], 
                           filename: Optional[str] = None,
                           options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Processa dados binários"""
        try:
            # Converte para BytesIO se necessário
            if isinstance(data, bytes):
                data = io.BytesIO(data)
            
            # Detecta formato
            if filename:
                file_ext = os.path.splitext(filename)[1].lower()
            else:
                file_ext = self._detect_format_from_content(data)
            
            # Processa baseado no formato
            if file_ext in self.supported_formats['excel']:
                return self._process_excel_binary(data, options)
            elif file_ext in self.supported_formats['csv']:
                return self._process_csv_binary(data, options)
            elif file_ext in self.supported_formats['json']:
                return self._process_json_binary(data, options)
            else:
                # Tenta CSV como fallback
                return self._process_csv_binary(data, options)
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Erro ao processar dados binários: {str(e)}",
                'data': None,
                'metadata': {}
            }
    
    def _detect_format_from_content(self, data: io.BytesIO) -> str:
        """Detecta formato do arquivo pelo conteúdo"""
        data.seek(0)
        header = data.read(100)
        data.seek(0)
        
        # Assinaturas de arquivos
        if header.startswith(b'PK'):  # ZIP (Excel)
            return '.xlsx'
        elif b'<?xml' in header:
            return '.xml'
        elif header.startswith(b'{') or header.startswith(b'['):
            return '.json'
        else:
            return '.csv'  # Default para texto
    
    def _process_excel_file(self, filepath: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Processa arquivo Excel"""
        try:
            # Lê todas as abas
            excel_file = pd.ExcelFile(filepath)
            sheets_data = {}
            
            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(filepath, sheet_name=sheet_name)
                    if not df.empty:
                        sheets_data[sheet_name] = df
                except Exception as e:
                    print(f"Erro ao ler aba {sheet_name}: {str(e)}")
            
            # Se só tem uma aba, retorna diretamente
            if len(sheets_data) == 1:
                sheet_name = list(sheets_data.keys())[0]
                df = sheets_data[sheet_name]
            else:
                # Se tem múltiplas abas, pega a maior ou mais relevante
                df = max(sheets_data.values(), key=len)
                sheet_name = 'multiple_sheets'
            
            return {
                'success': True,
                'data': df,
                'metadata': {
                    'source_type': 'excel',
                    'filename': os.path.basename(filepath),
                    'sheets': list(sheets_data.keys()),
                    'selected_sheet': sheet_name,
                    'rows': len(df),
                    'columns': len(df.columns)
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Erro ao processar Excel: {str(e)}",
                'data': None,
                'metadata': {}
            }
    
    def _process_excel_binary(self, data: io.BytesIO, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Processa dados Excel binários"""
        try:
            data.seek(0)
            excel_file = pd.ExcelFile(data)
            
            # Pega a primeira aba ou a especificada
            sheet_name = excel_file.sheet_names[0]
            df = pd.read_excel(data, sheet_name=sheet_name)
            
            return {
                'success': True,
                'data': df,
                'metadata': {
                    'source_type': 'excel_binary',
                    'sheets': excel_file.sheet_names,
                    'selected_sheet': sheet_name,
                    'rows': len(df),
                    'columns': len(df.columns)
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Erro ao processar Excel binário: {str(e)}",
                'data': None,
                'metadata': {}
            }
    
    def _process_csv_file(self, filepath: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Processa arquivo CSV com detecção automática"""
        try:
            df = self._read_csv_with_detection(filepath)
            
            return {
                'success': True,
                'data': df,
                'metadata': {
                    'source_type': 'csv',
                    'filename': os.path.basename(filepath),
                    'rows': len(df),
                    'columns': len(df.columns)
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Erro ao processar CSV: {str(e)}",
                'data': None,
                'metadata': {}
            }
    
    def _process_csv_binary(self, data: io.BytesIO, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Processa dados CSV binários"""
        try:
            data.seek(0)
            df = self._read_csv_with_detection(data)
            
            return {
                'success': True,
                'data': df,
                'metadata': {
                    'source_type': 'csv_binary',
                    'rows': len(df),
                    'columns': len(df.columns)
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Erro ao processar CSV binário: {str(e)}",
                'data': None,
                'metadata': {}
            }
    
    def _read_csv_with_detection(self, file_input) -> pd.DataFrame:
        """Lê CSV com detecção automática de encoding e separador"""
        
        # Tenta diferentes combinações de encoding e separador
        for encoding in self.encoding_options:
            for separator in self.separator_options:
                try:
                    if hasattr(file_input, 'seek'):
                        file_input.seek(0)
                    
                    df = pd.read_csv(file_input, 
                                   sep=separator, 
                                   encoding=encoding,
                                   low_memory=False)
                    
                    # Verifica se a leitura foi bem-sucedida
                    if len(df.columns) > 1 and len(df) > 0:
                        return df
                        
                except Exception:
                    continue
        
        # Se nada funcionou, tenta com parâmetros padrão
        if hasattr(file_input, 'seek'):
            file_input.seek(0)
        return pd.read_csv(file_input, low_memory=False)
    
    def _process_json_file(self, filepath: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Processa arquivo JSON"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Converte para DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Tenta diferentes estratégias
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                elif 'values' in data:
                    df = pd.DataFrame(data['values'])
                else:
                    df = pd.DataFrame([data])
            else:
                df = pd.DataFrame([{'value': data}])
            
            return {
                'success': True,
                'data': df,
                'metadata': {
                    'source_type': 'json',
                    'filename': os.path.basename(filepath),
                    'rows': len(df),
                    'columns': len(df.columns)
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Erro ao processar JSON: {str(e)}",
                'data': None,
                'metadata': {}
            }
    
    def _process_json_binary(self, data: io.BytesIO, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Processa dados JSON binários"""
        try:
            data.seek(0)
            json_data = json.loads(data.read().decode('utf-8'))
            
            # Converte para DataFrame
            if isinstance(json_data, list):
                df = pd.DataFrame(json_data)
            elif isinstance(json_data, dict):
                df = pd.DataFrame([json_data])
            else:
                df = pd.DataFrame([{'value': json_data}])
            
            return {
                'success': True,
                'data': df,
                'metadata': {
                    'source_type': 'json_binary',
                    'rows': len(df),
                    'columns': len(df.columns)
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Erro ao processar JSON binário: {str(e)}",
                'data': None,
                'metadata': {}
            }
    
    def _process_parquet_file(self, filepath: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Processa arquivo Parquet"""
        try:
            df = pd.read_parquet(filepath)
            
            return {
                'success': True,
                'data': df,
                'metadata': {
                    'source_type': 'parquet',
                    'filename': os.path.basename(filepath),
                    'rows': len(df),
                    'columns': len(df.columns)
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Erro ao processar Parquet: {str(e)}",
                'data': None,
                'metadata': {}
            }
    
    def _process_feather_file(self, filepath: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Processa arquivo Feather"""
        try:
            df = pd.read_feather(filepath)
            
            return {
                'success': True,
                'data': df,
                'metadata': {
                    'source_type': 'feather',
                    'filename': os.path.basename(filepath),
                    'rows': len(df),
                    'columns': len(df.columns)
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Erro ao processar Feather: {str(e)}",
                'data': None,
                'metadata': {}
            }
    
    def _auto_detect_and_process(self, filepath: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Detecta formato automaticamente e processa"""
        
        # Tenta diferentes formatos
        formats_to_try = [
            ('csv', self._process_csv_file),
            ('excel', self._process_excel_file),
            ('json', self._process_json_file)
        ]
        
        for format_name, process_func in formats_to_try:
            try:
                result = process_func(filepath, options)
                if result['success']:
                    result['metadata']['detected_format'] = format_name
                    return result
            except Exception:
                continue
        
        return {
            'success': False,
            'error': 'Não foi possível detectar o formato do arquivo',
            'data': None,
            'metadata': {}
        }


class SmartDataCleaner:
    """Limpeza inteligente de dados"""
    
    def __init__(self):
        self.cleaning_strategies = {
            'remove_empty_rows': True,
            'remove_empty_columns': True,
            'standardize_headers': True,
            'detect_and_fix_types': True,
            'handle_duplicates': True
        }
    
    def clean_dataframe(self, df: pd.DataFrame, strategies: Optional[Dict[str, bool]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Limpa DataFrame usando estratégias inteligentes"""
        
        if strategies:
            self.cleaning_strategies.update(strategies)
        
        original_shape = df.shape
        cleaning_log = {
            'original_shape': original_shape,
            'operations': [],
            'final_shape': None,
            'warnings': []
        }
        
        df_clean = df.copy()
        
        # Remove linhas completamente vazias
        if self.cleaning_strategies.get('remove_empty_rows', True):
            before_rows = len(df_clean)
            df_clean = df_clean.dropna(how='all')
            removed_rows = before_rows - len(df_clean)
            if removed_rows > 0:
                cleaning_log['operations'].append({
                    'operation': 'remove_empty_rows',
                    'removed_count': removed_rows,
                    'description': f'Removidas {removed_rows} linhas completamente vazias'
                })
        
        # Remove colunas completamente vazias
        if self.cleaning_strategies.get('remove_empty_columns', True):
            before_cols = len(df_clean.columns)
            df_clean = df_clean.dropna(axis=1, how='all')
            removed_cols = before_cols - len(df_clean.columns)
            if removed_cols > 0:
                cleaning_log['operations'].append({
                    'operation': 'remove_empty_columns',
                    'removed_count': removed_cols,
                    'description': f'Removidas {removed_cols} colunas completamente vazias'
                })
        
        # Padroniza cabeçalhos
        if self.cleaning_strategies.get('standardize_headers', True):
            original_columns = df_clean.columns.tolist()
            df_clean.columns = self._standardize_column_names(df_clean.columns)
            if list(df_clean.columns) != original_columns:
                cleaning_log['operations'].append({
                    'operation': 'standardize_headers',
                    'description': 'Cabeçalhos padronizados (espaços, caracteres especiais)',
                    'example_changes': dict(zip(original_columns[:3], df_clean.columns[:3]))
                })
        
        # Detecta e corrige tipos
        if self.cleaning_strategies.get('detect_and_fix_types', True):
            type_changes = self._auto_convert_types(df_clean)
            if type_changes:
                cleaning_log['operations'].append({
                    'operation': 'detect_and_fix_types',
                    'changes': type_changes,
                    'description': f'Convertidos tipos de dados em {len(type_changes)} colunas'
                })
        
        # Trata duplicatas
        if self.cleaning_strategies.get('handle_duplicates', True):
            before_rows = len(df_clean)
            df_clean = df_clean.drop_duplicates()
            removed_duplicates = before_rows - len(df_clean)
            if removed_duplicates > 0:
                cleaning_log['operations'].append({
                    'operation': 'handle_duplicates',
                    'removed_count': removed_duplicates,
                    'description': f'Removidas {removed_duplicates} linhas duplicadas'
                })
        
        cleaning_log['final_shape'] = df_clean.shape
        
        return df_clean, cleaning_log
    
    def _standardize_column_names(self, columns: pd.Index) -> List[str]:
        """Padroniza nomes de colunas"""
        standardized = []
        
        for col in columns:
            # Converte para string
            col_str = str(col)
            
            # Remove espaços extras
            col_str = ' '.join(col_str.split())
            
            # Remove caracteres especiais no início/fim
            col_str = col_str.strip('_-. ')
            
            # Se ficou vazio, gera nome genérico
            if not col_str or col_str.lower() in ['unnamed', 'nan', 'null']:
                col_str = f'Column_{len(standardized) + 1}'
            
            # Garante unicidade
            original_col = col_str
            counter = 1
            while col_str in standardized:
                col_str = f"{original_col}_{counter}"
                counter += 1
            
            standardized.append(col_str)
        
        return standardized
    
    def _auto_convert_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Converte tipos automaticamente"""
        type_changes = {}
        
        for col in df.columns:
            try:
                # Tenta converter para numérico
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                non_null_numeric = numeric_series.count()
                
                if non_null_numeric > len(df[col]) * 0.8:  # 80% são numéricos
                    # Verifica se são todos inteiros
                    if all(numeric_series.dropna() == numeric_series.dropna().astype(int)):
                        df[col] = numeric_series.astype('Int64')  # Permite NaN
                        type_changes[col] = 'integer'
                    else:
                        df[col] = numeric_series
                        type_changes[col] = 'float'
                
                # Tenta converter para datetime
                elif self._could_be_date(df[col]):
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        type_changes[col] = 'datetime'
                    except:
                        pass
                
            except Exception:
                continue
        
        return type_changes
    
    def _could_be_date(self, series: pd.Series) -> bool:
        """Verifica se uma série poderia ser data"""
        # Amostra alguns valores para teste
        sample = series.dropna().astype(str).head(10)
        
        date_patterns = [
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',
            r'\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)',
        ]
        
        import re
        for pattern in date_patterns:
            matches = sample.str.contains(pattern, regex=True, case=False).sum()
            if matches > len(sample) * 0.5:  # 50% coincidem
                return True
        
        return False