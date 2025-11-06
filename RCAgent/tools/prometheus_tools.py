from prometheus_api_client import PrometheusConnect
import datetime, json, os


class PrometheusTool:
    def __init__(self, url="http://localhost:9090"):
        self.client = PrometheusConnect(url=url, disable_ssl=True)

    def query_range(self, metric: str, minutes: int = 5):
        end_time = datetime.datetime.now()
        start_time = end_time - datetime.timedelta(minutes=minutes)
        return self.client.get_metric_range_data(
            metric, start_time=start_time, end_time=end_time
        )

    def snapshot(self, metric: str, minutes: int = 5, path="snapshots/prometheus.json"):
        data = self.query_range(metric, minutes)
        os.makedirs("snapshots", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path


if __name__ == "__main__":
    prom = PrometheusTool()
    print("Prometheus snapshot saved:", prom.snapshot("up"))
