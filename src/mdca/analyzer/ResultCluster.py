import math

from mdca.analyzer.Index import IndexLocations
from mdca.analyzer.ResultPath import CalculatedResult


RESULT_CLUSTERING_MAX_DISTANCE: float = 0.9


class ResultCluster:
    def __init__(self, first_result: CalculatedResult):
        self.best_result: CalculatedResult = first_result
        self.results: list[CalculatedResult] = [first_result]

    def add_result(self, result: CalculatedResult):
        self.results.append(result)
        if result.weight > self.best_result.weight:
            self.best_result = result


class ResultClusterSet:
    def __init__(self):
        self.clusters: list[ResultCluster] = []

    def _distance(self, a: IndexLocations, b: IndexLocations) -> float:
        common_count: int = (a & b).count
        distance: float = math.sqrt(((a.count - common_count) / a.count) * ((b.count - common_count) / b.count))
        return distance

    def cluster_result(self, result: CalculatedResult) -> None:
        for cluster in self.clusters:
            distance: float = self._distance(cluster.best_result.locations, result.locations)
            if distance <= RESULT_CLUSTERING_MAX_DISTANCE:
                cluster.add_result(result)
                return
        new_cluster: ResultCluster = ResultCluster(result)
        self.clusters.append(new_cluster)

    def get_results(self) -> list[CalculatedResult]:
        results: list[CalculatedResult] = []
        for cluster in self.clusters:
            results.append(cluster.best_result)
        return results

    def __len__(self) -> int:
        return len(self.clusters)
