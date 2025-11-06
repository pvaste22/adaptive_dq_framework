FROM --platform=$TARGETPLATFORM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates tini curl gnupg && rm -rf /var/lib/apt/lists/*
# Use prebuilt wheels for speed
RUN python -m pip install -U pip wheel setuptools && \
    pip install --no-cache-dir \
      confluent-kafka==2.3.0 \
      numpy==1.26.4 \
      pandas==2.2.2 \
      scikit-learn==1.4.2 \
      xgboost==1.7.6 \
      ujson==5.9.0 \
      PyYAML==6.0.1
      
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && pip install -r /app/requirements.txt
COPY src/ /app/src/
COPY data/artifacts/models/latest/ /app/data/artifacts/models/latest/
ENV KAFKA_BOOTSTRAP_SERVERS="my-kafka-controller-0.my-kafka-controller-headless.default.svc.cluster.local:9092" \
    TOPIC_INPUT="e2-data" \
    TOPIC_OUTPUT="dq-scores" \
    KAFKA_GROUP_ID="dq-xapp-300" \
    WINDOW_SIZE_SEC="300" \
    TICK_INTERVAL_SEC="60" \
    DQ_MODEL_DIR="/app/data/artifacts/models/latest"
ENTRYPOINT ["/usr/bin/tini","-g","--"]
CMD ["python","-m","src.deployment.kafka_consumer"]
