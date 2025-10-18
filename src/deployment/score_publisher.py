"""
Score Publisher - Publishes DQ scores to Kafka
"""
import json
import time
from typing import Dict, Any
from confluent_kafka import Producer

from common.logger import get_phase4_logger
from common.constants import MAIN_CONFIG

logger = get_phase4_logger('score_publisher')

class ScorePublisher:
    """Publishes DQ scores to Kafka topic"""
    
    def __init__(self, bootstrap_servers: str = None, topic: str = None):
        """
        Initialize Kafka producer
        
        Args:
            bootstrap_servers: Kafka brokers (defaults to config)
            topic: Output topic (defaults to config)
        """
        kafka_config = MAIN_CONFIG.get('kafka', {})
        
        self.bootstrap_servers = bootstrap_servers or kafka_config.get('brokers', 'localhost:9092')
        self.topic = topic or kafka_config.get('out_topic', 'dq-scores')
        
        self.producer = Producer({
            'bootstrap.servers': self.bootstrap_servers,
            'acks': kafka_config.get('acks', 'all'),
            'enable.idempotence': kafka_config.get('idempotence', True),
            'compression.type': kafka_config.get('compression', 'lz4'),
            'linger.ms': 10,
            'batch.size': 16384,
        })
        
        logger.info(f"Score publisher initialized: topic={self.topic}")
    
    def publish_score(self, 
                     window_id: str,
                     start_time: str,
                     end_time: str,
                     dq_score: float,
                     metadata: Dict[str, Any] = None):
        """
        Publish DQ score for a window
        
        Args:
            window_id: Unique window identifier
            start_time: Window start timestamp (ISO)
            end_time: Window end timestamp (ISO)
            dq_score: Predicted quality score (0.0-1.0)
            metadata: Optional additional metadata
        """
        try:
            message = {
                'window_id': window_id,
                'start_time': start_time,
                'end_time': end_time,
                'dq_score': float(dq_score),
                'timestamp': time.time(),
            }
            message['publish_ts_ms'] = int(time.time() * 1000)
            # Add optional metadata
            if metadata:
                message['metadata'] = metadata
            
            # Serialize to JSON
            msg_bytes = json.dumps(message).encode('utf-8')
            
            # Publish
            self.producer.produce(
                self.topic,
                key=window_id.encode('utf-8'),
                value=msg_bytes,
                callback=self._delivery_callback
            )
            
            self.producer.poll(0)  # Trigger callbacks
            
            logger.debug(f"Published score: window={window_id}, score={dq_score:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to publish score: {e}")
            raise
    
    def _delivery_callback(self, err, msg):
        """Callback for delivery confirmation"""
        if err:
            logger.error(f"Message delivery failed: {err}")
        else:
            logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}]")
    
    def flush(self, timeout: float = 5.0):
        """Wait for all messages to be delivered"""
        remaining = self.producer.flush(timeout)
        if remaining > 0:
            logger.warning(f"{remaining} messages not delivered")
        else:
            logger.info("All messages delivered")
    
    def close(self):
        """Close producer"""
        self.flush()
        logger.info("Score publisher closed")