import logging, datetime, requests, json, os
from logging_loki import LokiHandler


class LokiTool:
    def __init__(self, url="http://localhost:3100", labels={"job": "rcagent"}):
        self.url = url
        self.labels = labels
        self.logger = logging.getLogger("loki")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(LokiHandler(url=f"{url}/loki/api/v1/push", tags=labels))

    def push_log(self, message: str):
        self.logger.info(message)
        return f"Log pushed: {message}"

    def snapshot(self, query='{job="rcagent"}', minutes=5, path="snapshots/loki.json"):
        end = int(datetime.datetime.now().timestamp() * 1e9)
        start = end - minutes * 60 * 1e9
        resp = requests.get(
            f"{self.url}/loki/api/v1/query_range",
            params={"query": query, "start": start, "end": end, "limit": 500},
        )
        data = resp.json()
        os.makedirs("snapshots", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path


if __name__ == "__main__":
    loki = LokiTool()
    loki.push_log("RCAgent log event")
    print("Loki snapshot saved:", loki.snapshot())
