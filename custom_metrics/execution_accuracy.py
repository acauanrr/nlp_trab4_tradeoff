# project_root/custom_metrics/execution_accuracy.py

import sqlite3
import os
from typing import Tuple, Any
import asyncio
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

class ExecutionAccuracy(BaseMetric):
    """
    Métrica que avalia a acurácia de execução de uma query SQL.

    Executa a query gerada (actual_output) e a query de referência
    (expected_output) num banco SQLite, compara se os resultados coincidem,
    ignorando a ordem. Implementa tanto a interface síncrona (measure) quanto
    a assíncrona (a_measure) exigida pelo DeepEval para paralelismo.
    """

    def __init__(self, db_root_path: str = "data/spider/database", threshold: float = 1.0) -> None:
        # --- CORREÇÃO: Inicializa o pai sem argumentos ---
        super().__init__()
        # Atribui o threshold e outros atributos depois da inicialização
        self.threshold = threshold
        self.db_root_path = db_root_path
        self.async_mode = True # Sinaliza ao DeepEval que podemos rodar em modo assíncrono

    async def a_measure(self, test_case: LLMTestCase, **kwargs) -> float:
        """
        Versão assíncrona exigida pelo DeepEval, que executa 'measure' em um thread.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.measure, test_case)

    def measure(self, test_case: LLMTestCase) -> float:
        """Mede a acurácia de execução para um dado test case."""
        if not isinstance(test_case.actual_output, str) or not isinstance(test_case.expected_output, str):
            self.reason = "actual_output ou expected_output não é uma string"
            self.score = 0.0
            return self.score

        if not test_case.context or not isinstance(test_case.context, list):
            self.reason = "O 'context' deve ser uma lista contendo o db_id no índice 0."
            self.score = 0.0
            return self.score

        db_id = test_case.context[0]
        db_path = os.path.join(self.db_root_path, db_id, f"{db_id}.sqlite")

        if not os.path.exists(db_path):
            self.reason = f"Banco de dados não encontrado em '{db_path}'"
            self.score = 0.0
            return self.score

        predicted_results, pred_error = self._execute_sql(test_case.actual_output, db_path)
        if pred_error:
            self.reason = f"Erro ao executar a query gerada: {pred_error}"
            self.score = 0.0
            return self.score

        expected_results, gt_error = self._execute_sql(test_case.expected_output, db_path)
        if gt_error:
            self.reason = f"Erro na query de referência: {gt_error}"
            self.score = 0.0
            return self.score

        if predicted_results == expected_results:
            self.score = 1.0
            self.reason = "Resultados idênticos."
        else:
            self.score = 0.0
            # Adiciona mais detalhes no motivo da falha para facilitar a depuração
            self.reason = f"Resultados divergem. Gerado: {predicted_results}, Esperado: {expected_results}"

        return self.score

    def _execute_sql(self, query: str, db_path: str) -> Tuple[Any, Any]:
        """Executa uma query SQL e devolve (result_set, error_message | None)."""
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                return set(cursor.fetchall()), None
        except sqlite3.Error as exc:
            return None, str(exc)

    def is_successful(self) -> bool:
        """Convenience para DeepEval dashboards."""
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Execution Accuracy"
