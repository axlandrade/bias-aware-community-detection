# src/memory_manager.py
import psutil
import gc
import os
import numpy as np
from typing import Optional

class MemoryManager:
    """Gerenciador de memória para sistemas com recursos limitados"""
    
    def __init__(self, max_memory_gb: float = 6.0):  # Deixar 2GB para sistema
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self.process = psutil.Process(os.getpid())
    
    def get_memory_usage(self) -> float:
        """Retorna uso de memória em GB"""
        return self.process.memory_info().rss / (1024 ** 3)
    
    def is_memory_critical(self, threshold_gb: Optional[float] = None) -> bool:
        """Verifica se a memória está crítica"""
        if threshold_gb is None:
            threshold_gb = self.max_memory_bytes / (1024 ** 3) * 0.9  # 90% do máximo
        return self.get_memory_usage() > threshold_gb
    
    def force_cleanup(self):
        """Força limpeza de memória"""
        gc.collect()
        gc.collect()  # Duas vezes para garantir
        
        # Liberar memória do numpy
        try:
            np._globals._NoValue = None
        except:
            pass
    
    def safe_chunk_processing(self, data, chunk_size: int, process_func):
        """
        Processa dados em chunks com controle de memória
        """
        results = []
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            
            # Verificar memória antes de processar
            if self.is_memory_critical():
                print("⚠️  Memória crítica, forçando limpeza...")
                self.force_cleanup()
            
            result = process_func(chunk)
            results.append(result)
            
            # Limpar referências
            del chunk
            self.force_cleanup()
        
        return results

def optimize_memory_settings():
    """Configurações otimizadas para seu hardware"""
    # Configurar numpy para usar menos memória
    os.environ['OMP_NUM_THREADS'] = '2'  # Limitar threads
    os.environ['MKL_NUM_THREADS'] = '2'
    
    # Configurar garbage collection
    gc.set_threshold(700, 10, 10)  # Coleta mais agressiva
    
    print("✅ Configurações de memória otimizadas para GTX 1050 + 8GB RAM")

# Uso no notebook:
# from src.memory_manager import MemoryManager, optimize_memory_settings
# optimize_memory_settings()
# memory_manager = MemoryManager()