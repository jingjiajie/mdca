from mdca.analyzer.Index import IndexLocations
from mdca.analyzer.ResultPath import CalculatedResult, BinResultValue
from mdca.analyzer.commons import ColumnInfo

RESULT_CLUSTERING_DISTANCE: float = 0.2


class ResultCluster:
    def __init__(self, column_info: dict[str, ColumnInfo], centroid: CalculatedResult):
        self.centroid: CalculatedResult = centroid
        self.best_result: CalculatedResult = centroid
        self.results: list[CalculatedResult] = []
        self.column_info: dict[str, ColumnInfo] = column_info

    def __str__(self):
        return str(self.centroid)

    def add_result(self, result: CalculatedResult):
        self.results.append(result)

        if result.weight > self.best_result.weight:
            self.best_result = result
        elif result.weight == self.best_result.weight:
            if len(result.items) > len(self.best_result.items):
                self.best_result = result
            elif len(result.items) == len(self.best_result.items):
                result_cut_range_cov: float = self._get_total_cut_range_coverage(result)
                # todo 缓存
                best_result_cut_range_cov: float = self._get_total_cut_range_coverage(self.best_result)
                if result_cut_range_cov < best_result_cut_range_cov:
                    self.best_result = result

    def _get_total_cut_range_coverage(self, result: CalculatedResult):
        total_cov: float = 0
        for item in result.items:
            if isinstance(item.value, BinResultValue):
                col_info: ColumnInfo = self.column_info[item.column]
                bin_val: BinResultValue = item.value
                bin_cov: float | None = None
                if bin_val.lower_cut_point_value is not None and bin_val.upper_cut_point_value is not None:
                    bin_cov = (bin_val.upper_cut_point_value - bin_val.lower_cut_point_value) / (
                                col_info.max - col_info.min)
                elif bin_val.lower_cut_point_value is not None:
                    bin_cov = (col_info.max - bin_val.lower_cut_point_value) / (col_info.max - col_info.min)
                elif bin_val.upper_cut_point_value is not None:
                    bin_cov = (bin_val.upper_cut_point_value - col_info.min) / (col_info.max - col_info.min)
                assert bin_cov is not None and bin_cov >= 0
                total_cov += bin_cov
        return total_cov


class ResultClusterSet:
    def __init__(self, column_info: dict[str, ColumnInfo]):
        self.clusters: list[ResultCluster] = []
        self.column_info: dict[str, ColumnInfo] = column_info

    def _distance(self, a: IndexLocations, b: IndexLocations) -> float:
        # common_count: int = (a & b).count
        # distance: float = math.sqrt(((a.count - common_count) / a.count) * ((b.count - common_count) / b.count))
        distance: float = (a.count + b.count - 2 * (a & b).count) / (a.count + b.count)
        return distance

    def cluster_result(self, result: CalculatedResult) -> None:
        min_distance: float | None = None
        min_distance_cluster: ResultCluster | None = None
        for cluster in self.clusters:
            distance: float = self._distance(cluster.centroid.locations, result.locations)
            if min_distance is None or distance < min_distance:
                min_distance = distance
                min_distance_cluster = cluster
        if min_distance is not None and min_distance <= RESULT_CLUSTERING_DISTANCE:
            min_distance_cluster.add_result(result)
        else:
            new_cluster: ResultCluster = ResultCluster(self.column_info, result)
            self.clusters.append(new_cluster)

    def get_results(self) -> list[CalculatedResult]:
        results: list[CalculatedResult] = []
        for cluster in self.clusters:
            results.append(cluster.best_result)
        return results

    def __len__(self) -> int:
        return len(self.clusters)
