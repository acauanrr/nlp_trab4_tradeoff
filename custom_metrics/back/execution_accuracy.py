# project_root/custom_metrics/execution_accuracy.py

import sqlite3
from typing import List, Tuple, Any, Dict
import os
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

class ExecutionAccuracy(BaseMetric):
    """
    Métrica que avalia a acurácia de execução de uma query SQL.

    Esta métrica executa a query gerada (actual_output) e a query de referência
    (expected_output) em um banco de dados SQLite e compara se os conjuntos de
    resultados são idênticos. A ordem das linhas é ignorada na comparação.
    """
    def __init__(
        self,
        db_root_path: str = "data/spider/database",
        threshold: float = 1.0
    ):
        self.threshold = threshold
        self.db_root_path = db_root_path

    def measure(self, test_case: LLMTestCase) -> float:
        """
        Mede a acurácia de execução para um dado test case.
        """
        if not all([isinstance(test_case.actual_output, str), isinstance(test_case.expected_output, str)]):
            raise ValueError("actual_output e expected_output devem ser strings contendo queries SQL.")
            
        # --- ALTERAÇÃO: Lendo o db_id de 'context' ---
        if not test_case.context or not isinstance(test_case.context, list) or len(test_case.context) == 0:
            raise ValueError("O parâmetro 'context' do test case deve conter o 'db_id' como seu primeiro elemento.")
        
        db_id = test_case.context[0]
        db_path = os.path.join(self.db_root_path, db_id, f"{db_id}.sqlite")

        if not os.path.exists(db_path):
            self.reason = f"Falha: Banco de dados não encontrado em '{db_path}'"
            self.score = 0.0
            return self.score

        # Executa a query gerada
        predicted_results, pred_error = self._execute_sql(test_case.actual_output, db_path)
        if pred_error:
            self.reason = f"Erro ao executar a query gerada: {pred_error}"
            self.score = 0.0
            return self.score

        # Executa a query de referência
        expected_results, gt_error = self._execute_sql(test_case.expected_output, db_path)
        if gt_error:
            self.reason = f"AVISO: Erro ao executar a query de referência: {gt_error}. A comparação pode não ser válida."
            self.score = 0.0
            return self.score
            
        # Compara os resultados (ordem-agnóstico)
        if predicted_results == expected_results:
            self.score = 1.0
            self.reason = "Sucesso: Os resultados da query gerada e da referência são idênticos."
        else:
            self.score = 0.0
            self.reason = f"Falha: Os resultados diferem. Gerado: {predicted_results}, Esperado: {expected_results}"
            
        return self.score

    def _execute_sql(self, query: str, db_path: str) -> Tuple[Any, Any]:
        """
        Executa uma query SQL no banco de dados especificado e retorna os resultados.
        """
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                results = cursor.fetchall()
                # Converte para um conjunto de tuplas para comparação agnóstica à ordem
                return set(results), None
        except sqlite3.Error as e:
            return None, str(e)

    def is_successful(self) -> bool:
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Execution Accuracy"
