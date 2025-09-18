"""
Database connector para Turso LibSQL
Sistema Regina - Smart Analytics
"""

import os
import json
import sqlite3
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

class TursoConnector:
    """Conector para banco de dados Turso LibSQL"""
    
    def __init__(self):
        self.database_url = os.getenv('TURSO_DATABASE_URL')
        self.auth_token = os.getenv('TURSO_AUTH_TOKEN')
        self.local_db_path = 'regina_smart_local.db'
        self._init_local_db()
    
    def _init_local_db(self):
        """Inicializa banco local SQLite para cache e desenvolvimento"""
        conn = sqlite3.connect(self.local_db_path)
        cursor = conn.cursor()
        
        # Tabela de análises realizadas
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                analysis_data TEXT,
                insights TEXT,
                charts_data TEXT,
                statistics TEXT
            )
        ''')
        
        # Tabela de métricas de performance
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER,
                processing_time REAL,
                data_rows INTEGER,
                data_columns INTEGER,
                charts_generated INTEGER,
                insights_generated INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (analysis_id) REFERENCES analyses (id)
            )
        ''')
        
        # Tabela de logs do sistema
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                module TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabela de configurações
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_analysis(self, session_id: str, filename: str, file_type: str, 
                     analysis_data: Dict, insights: List, charts_data: Dict, 
                     statistics: Dict) -> int:
        """Salva uma análise completa no banco de dados"""
        conn = sqlite3.connect(self.local_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analyses (session_id, filename, file_type, analysis_data, 
                                insights, charts_data, statistics)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            filename,
            file_type,
            json.dumps(analysis_data),
            json.dumps(insights),
            json.dumps(charts_data),
            json.dumps(statistics)
        ))
        
        analysis_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        self.log_event('INFO', f'Análise salva: {filename} (ID: {analysis_id})')
        return analysis_id
    
    def save_performance_metrics(self, analysis_id: int, processing_time: float,
                               data_rows: int, data_columns: int,
                               charts_generated: int, insights_generated: int):
        """Salva métricas de performance da análise"""
        conn = sqlite3.connect(self.local_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO performance_metrics 
            (analysis_id, processing_time, data_rows, data_columns, 
             charts_generated, insights_generated)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (analysis_id, processing_time, data_rows, data_columns,
              charts_generated, insights_generated))
        
        conn.commit()
        conn.close()
    
    def get_recent_analyses(self, limit: int = 10) -> List[Dict]:
        """Busca análises recentes"""
        conn = sqlite3.connect(self.local_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, session_id, filename, file_type, upload_timestamp
            FROM analyses 
            ORDER BY upload_timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        analyses = []
        for row in results:
            analyses.append({
                'id': row[0],
                'session_id': row[1],
                'filename': row[2],
                'file_type': row[3],
                'upload_timestamp': row[4]
            })
        
        return analyses
    
    def get_analysis_by_id(self, analysis_id: int) -> Optional[Dict]:
        """Busca análise específica por ID"""
        conn = sqlite3.connect(self.local_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM analyses WHERE id = ?
        ''', (analysis_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'id': row[0],
                'session_id': row[1],
                'filename': row[2],
                'file_type': row[3],
                'upload_timestamp': row[4],
                'analysis_data': json.loads(row[5]) if row[5] else {},
                'insights': json.loads(row[6]) if row[6] else [],
                'charts_data': json.loads(row[7]) if row[7] else {},
                'statistics': json.loads(row[8]) if row[8] else {}
            }
        return None
    
    def get_system_stats(self) -> Dict:
        """Busca estatísticas gerais do sistema"""
        conn = sqlite3.connect(self.local_db_path)
        cursor = conn.cursor()
        
        # Total de análises
        cursor.execute('SELECT COUNT(*) FROM analyses')
        total_analyses = cursor.fetchone()[0]
        
        # Análises hoje
        cursor.execute('''
            SELECT COUNT(*) FROM analyses 
            WHERE DATE(upload_timestamp) = DATE('now')
        ''')
        analyses_today = cursor.fetchone()[0]
        
        # Métricas médias
        cursor.execute('''
            SELECT AVG(processing_time), AVG(data_rows), AVG(charts_generated)
            FROM performance_metrics
        ''')
        avg_metrics = cursor.fetchone()
        
        # Tipos de arquivo mais comuns
        cursor.execute('''
            SELECT file_type, COUNT(*) as count
            FROM analyses
            GROUP BY file_type
            ORDER BY count DESC
            LIMIT 5
        ''')
        file_types = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_analyses': total_analyses,
            'analyses_today': analyses_today,
            'avg_processing_time': round(avg_metrics[0] or 0, 2),
            'avg_data_rows': int(avg_metrics[1] or 0),
            'avg_charts_generated': int(avg_metrics[2] or 0),
            'popular_file_types': [{'type': ft[0], 'count': ft[1]} for ft in file_types]
        }
    
    def log_event(self, level: str, message: str, module: str = 'system'):
        """Registra evento no log do sistema"""
        conn = sqlite3.connect(self.local_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO system_logs (level, message, module)
            VALUES (?, ?, ?)
        ''', (level, message, module))
        
        conn.commit()
        conn.close()
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Busca configuração do sistema"""
        conn = sqlite3.connect(self.local_db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT value FROM system_config WHERE key = ?', (key,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            try:
                return json.loads(result[0])
            except:
                return result[0]
        return default
    
    def set_config(self, key: str, value: Any):
        """Define configuração do sistema"""
        conn = sqlite3.connect(self.local_db_path)
        cursor = conn.cursor()
        
        value_str = json.dumps(value) if not isinstance(value, str) else value
        
        cursor.execute('''
            INSERT OR REPLACE INTO system_config (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        ''', (key, value_str))
        
        conn.commit()
        conn.close()
    
    def sync_to_turso(self):
        """Sincroniza dados locais com Turso (implementação futura)"""
        # TODO: Implementar sincronização com Turso quando libsql-client estiver disponível
        self.log_event('INFO', 'Sincronização com Turso ainda não implementada')
        pass

# Instância global do conector
db_connector = TursoConnector()

def init_database():
    """Inicializa conexão com banco de dados"""
    db_connector.log_event('INFO', 'Sistema de banco de dados inicializado')
    return db_connector