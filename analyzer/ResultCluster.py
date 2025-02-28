from analyzer.ResultPath import CalculatedResult


RESULT_CLUSTER_MAX_DISTANCE: float = 0.35


class ResultCluster:
    def __init__(self, centroid: CalculatedResult):
        self.centroid: CalculatedResult = centroid
        self.results: list[CalculatedResult] = []

    def add_result(self, result: CalculatedResult):
        self.results.append(result)

    def get_best_result(self) -> CalculatedResult:
        max_weight_result: CalculatedResult = self.centroid
        for result in self.results:
            if result.weight > max_weight_result.weight:
                max_weight_result = result
        return max_weight_result

    def distance(self, result: CalculatedResult) -> float:
        centroid_item_map: set[str] = set()
        for item in self.centroid.items:
            centroid_item_map.add(str(item))
        equal_items: int = 0
        for item in result.items:
            if str(item) in centroid_item_map:
                equal_items += 1
        distance: float = ((len(self.centroid.items) + len(result.items) - 2 * equal_items) /
                           (len(self.centroid.items) + len(result.items)))
        return distance


class ResultClusterSet:
    def __init__(self):
        self.clusters: list[ResultCluster] = []

    def cluster_result(self, result: CalculatedResult) -> None:
        for cluster in self.clusters:
            distance: float = cluster.distance(result)
            if distance < RESULT_CLUSTER_MAX_DISTANCE:
                cluster.add_result(result)
                return
        new_cluster: ResultCluster = ResultCluster(result)
        self.clusters.append(new_cluster)

    def get_results(self) -> list[CalculatedResult]:
        results: list[CalculatedResult] = []
        for cluster in self.clusters:
            results.append(cluster.get_best_result())
        return results

    def __len__(self) -> int:
        return len(self.clusters)
