from jaeger_client import Config
import logging, os


class JaegerTool:
    def __init__(self, service_name="rcagent"):
        logging.getLogger("").handlers = []
        config = Config(
            config={
                "sampler": {"type": "const", "param": 1},
                "logging": True,
            },
            service_name=service_name,
        )
        self.tracer = config.initialize_tracer()

    def record_span(self, name="test_span"):
        with self.tracer.start_span(name) as span:
            span.set_tag("component", "RCAgent")
        return f"Span '{name}' recorded in Jaeger."

    def snapshot(self, path="snapshots/jaeger.txt"):
        os.makedirs("snapshots", exist_ok=True)
        with open(path, "w") as f:
            f.write("Jaeger snapshot placeholder. View traces in Jaeger UI.")
        return path


if __name__ == "__main__":
    jaeger = JaegerTool()
    print("Jaeger snapshot saved:", jaeger.snapshot())
