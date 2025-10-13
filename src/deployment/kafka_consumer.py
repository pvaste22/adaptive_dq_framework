"""
Complete Kafka Consumer with Windowing, Feature Extraction, and Prediction
DEPLOY THIS FILE AS XAPP
"""
import json
import signal
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import time
import pandas as pd
import numpy as np

from confluent_kafka import Consumer

# Your existing modules
from common.constants import KPM_TO_CANON, COLUMN_NAMES, MEAS_INTERVAL_SEC
from common.logger import get_phase4_logger
from data_processing.unit_converter import UnitConverter
from data_processing.feature_extractor import make_feature_row

# New modules
from deployment.dq_predictor import DQScorePredictor
from deployment.score_publisher import ScorePublisher

logger = get_phase4_logger('kafka_consumer')

# ============ CONFIGURATION ============
import os
BOOTSTRAP = os.getenv("KAFKA_BROKERS", "localhost:9092")
IN_TOPIC = "e2-data"
OUT_TOPIC = "dq-scores"
GROUP_ID = "dqscore-xapp"

WINDOW_SIZE_SEC = 300  # 5 minutes
TICK_INTERVAL_SEC = 60  # 1 minute step

_running = True

# ============ WINDOWED BUFFER ============
class TimeWindowBuffer:
    """Maintains time-windowed data per entity"""
    
    def __init__(self, window_size_sec: int = 300):
        self.window_size = timedelta(seconds=window_size_sec)
        # {entity_id: deque of (timestamp, row_dict)}
        self.buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=2000))
        
    def add_rows(self, entity_type: str, rows: List[Dict], converter: UnitConverter):
        """Add rows to buffer with unit conversion"""
        for row in rows:
            ts_str = row.get("timestamp")
            if not ts_str:
                continue
                
            ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            
            # Get entity ID
            if entity_type == "cell":
                entity_id = row.get("Viavi.Cell.Name")
            else:  # ue
                entity_id = row.get("Viavi.UE.Name")
                
            if entity_id:
                self.buffers[entity_id].append((ts, row))
    
    def get_window_data(self, entity_id: str, end_time: datetime) -> List[Dict]:
        """Get all rows in window [end_time - window_size, end_time]"""
        start_time = end_time - self.window_size
        
        window_data = []
        for ts, row in self.buffers[entity_id]:
            if start_time <= ts <= end_time:
                window_data.append(row)
        
        return window_data
    
    def get_all_entities(self) -> List[str]:
        """Get list of all entity IDs being tracked"""
        return list(self.buffers.keys())
    
    def cleanup_old_data(self, cutoff_time: datetime):
        """Remove data older than cutoff"""
        for entity_id in list(self.buffers.keys()):
            while self.buffers[entity_id]:
                ts, _ = self.buffers[entity_id][0]
                if ts < cutoff_time:
                    self.buffers[entity_id].popleft()
                else:
                    break
            
            if not self.buffers[entity_id]:
                del self.buffers[entity_id]

# ============ KAFKA HELPERS ============
def _stop(*_):
    global _running
    _running = False

def _iso_from_ms(ms: int) -> str:
    return datetime.fromtimestamp(ms/1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _decode_value(v):
    """Decode Kafka message value"""
    try:
        if isinstance(v, (bytes, bytearray)):
            b = bytes(v)
            # Strip 4-byte length prefix if present
            if len(b) >= 4 and b[0] != ord('{'):
                b = b[4:]
            
            decoded_str = b.decode("utf-8", errors='replace')
            return json.loads(decoded_str)
            
        elif isinstance(v, str):
            return json.loads(v)
        else:
            raise TypeError(f"Unsupported value type: {type(v)}")
            
    except Exception as e:
        logger.error(f"Decode error: {e}")
        raise

def normalize_from_e2sm(msg: dict) -> Tuple[Optional[str], List[Dict]]:
    """Parse E2SM message to normalized rows"""
    try:
        meta = msg.get("metadata", {})
        if "event_ts_ms" not in meta:
            return None, []
            
        ts_iso = _iso_from_ms(int(meta["event_ts_ms"]))
        rows: List[Dict] = []

        # Cell (Format1)
        fmt1 = msg.get("indicationMessage-Format1")
        if fmt1:
            names: List[str] = []
            for mi in fmt1.get("measInfoList", []):
                mt = mi.get("measType", {})
                name = mt.get("measName") or mt.get("measTypeName")
                if name:
                    names.append(name)

            for md in fmt1.get("measData", []):
                rec_meta = md.get("metadata", {}) or {}
                rec = {
                    "timestamp": ts_iso,
                    "Viavi.Cell.Name": rec_meta.get("cell_name"),
                }
                
                if "band" in rec_meta:
                    rec["band"] = rec_meta["band"]

                vals: List[float] = []
                for x in md.get("measRecord", []):
                    if isinstance(x, dict):
                        val = x.get("real") or x.get("integer") or x.get("measValue")
                        vals.append(val if val is not None else 0.0)
                    else:
                        vals.append(float(x) if x is not None else 0.0)

                for n, v in zip(names, vals):
                    canon = KPM_TO_CANON.get(n)
                    if canon:
                        rec[canon] = v
                        
                rows.append(rec)
            return "cell", rows

        # UE (Format3)
        fmt3 = msg.get("indicationMessage-Format3")
        if fmt3:
            names: List[str] = []
            for mi in fmt3.get("measInfoList", []):
                mt = mi.get("measType", {})
                name = mt.get("measName") or mt.get("measTypeName")
                if name:
                    names.append(name)

            for ue_data in fmt3.get("ueMeasData", []):
                ue_id_obj = ue_data.get("ueID", {})
                gnb_ue = ue_id_obj.get("gNB-UEID", {})
                ue_name = gnb_ue.get("amf-UE-NGAP-ID")

                vals: List[float] = []
                meas_report = ue_data.get("measReport", {}) or {}
                for report_item in meas_report.get("measReportList", []):
                    for x in report_item.get("measRecord", []):
                        if isinstance(x, dict):
                            val = x.get("real") or x.get("integer") or x.get("measValue")
                            vals.append(val if val is not None else 0.0)
                        else:
                            vals.append(float(x) if x is not None else 0.0)

                rec = {"timestamp": ts_iso, "Viavi.UE.Name": ue_name}
                
                for n, v in zip(names, vals):
                    canon = KPM_TO_CANON.get(n)
                    if canon:
                        rec[canon] = v
                        
                rows.append(rec)
            return "ue", rows

        return None, []
        
    except Exception as e:
        logger.error(f"Normalization error: {e}")
        return None, []

def build_consumer():
    return Consumer({
        "bootstrap.servers": BOOTSTRAP,
        "group.id": GROUP_ID,
        "enable.auto.commit": False,
        "auto.offset.reset": "latest",  # Only new messages
        "session.timeout.ms": 45000,
        "max.poll.interval.ms": 300000,
    })

# ============ WINDOW PROCESSOR ============
def process_windows(cell_buffer: TimeWindowBuffer, 
                   ue_buffer: TimeWindowBuffer,
                   converter: UnitConverter,
                   predictor: DQScorePredictor,
                   publisher: ScorePublisher):
    """Process all windows and compute DQ scores"""
    
    current_time = datetime.now(timezone.utc)
    window_id = current_time.strftime("%Y%m%d_%H%M%S")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"[TICK] Processing window: {window_id}")
    logger.info(f"{'='*60}")
    
    # Collect all data from both buffers for the window
    cell_entities = cell_buffer.get_all_entities()
    ue_entities = ue_buffer.get_all_entities()
    
    # Get all cell data
    all_cell_rows = []
    for entity_id in cell_entities:
        rows = cell_buffer.get_window_data(entity_id, current_time)
        all_cell_rows.extend(rows)
    
    # Get all UE data
    all_ue_rows = []
    for entity_id in ue_entities:
        rows = ue_buffer.get_window_data(entity_id, current_time)
        all_ue_rows.extend(rows)
    
    logger.info(f"Window data: {len(all_cell_rows)} cell rows, {len(all_ue_rows)} UE rows")
    
    # Check if we have enough data
    #if len(all_cell_rows) < 10 or len(all_ue_rows) < 10:
        #logger.warning(f"Insufficient data for window {window_id} - skipping")
        #return
    logger.info(f"Window data: {len(all_cell_rows)} cell rows, {len(all_ue_rows)} UE rows")
    
    try:
        # Convert to DataFrames
        cell_df = pd.DataFrame(all_cell_rows)
        ue_df = pd.DataFrame(all_ue_rows)
        
        # Apply unit conversion
        cell_df, ue_df = converter.standardize_units_comprehensive(cell_df, ue_df)
        logger.info(" Unit conversion applied")
        
        # Prepare window data
        window_start = current_time - timedelta(seconds=WINDOW_SIZE_SEC)
        window_data = {
            'cell_data': cell_df,
            'ue_data': ue_df,
            'metadata': {
                'window_id': window_id,
                'window_start_time': window_start.timestamp(),
                'window_end_time': current_time.timestamp(),
                'start_time': window_start.isoformat(),
                'end_time': current_time.isoformat(),
            }
        }
        
        # Extract features
        features = make_feature_row(window_data)
        logger.info(f" Extracted {len(features)} features")
        
        # Predict DQ score
        dq_score = predictor.predict(features)
        logger.info(f" Predicted DQ Score: {dq_score:.4f}")
        
        # Publish score
        publisher.publish_score(
            window_id=window_id,
            start_time=window_start.isoformat(),
            end_time=current_time.isoformat(),
            dq_score=dq_score,
            metadata={
                'cell_entities': len(cell_entities),
                'ue_entities': len(ue_entities),
                'cell_rows': len(all_cell_rows),
                'ue_rows': len(all_ue_rows),
            }
        )
        logger.info(f" Published score to '{OUT_TOPIC}' topic")
        
    except Exception as e:
        logger.error(f"Window processing failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup old data
    cutoff = current_time - timedelta(seconds=WINDOW_SIZE_SEC * 2)
    cell_buffer.cleanup_old_data(cutoff)
    ue_buffer.cleanup_old_data(cutoff)
    
    logger.info(f"{'='*60}\n")

# ============ MAIN CONSUMER LOOP ============
def consume_and_predict():
    """Main consumer loop with prediction pipeline"""
    global _running
    
    # Initialize components
    logger.info("Initializing DQ Score xApp...")
    
    consumer = build_consumer()
    consumer.subscribe([IN_TOPIC])
    
    converter = UnitConverter()
    predictor = DQScorePredictor()  # Auto-loads latest model
    publisher = ScorePublisher(bootstrap_servers=BOOTSTRAP, topic=OUT_TOPIC)
    
    cell_buffer = TimeWindowBuffer(WINDOW_SIZE_SEC)
    ue_buffer = TimeWindowBuffer(WINDOW_SIZE_SEC)
    
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    
    # Log model info
    model_info = predictor.get_model_info()
    logger.info(f"Loaded model: {model_info}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"DQ Score xApp STARTED")
    logger.info(f"  Input topic: {IN_TOPIC}")
    logger.info(f"  Output topic: {OUT_TOPIC}")
    logger.info(f"  Bootstrap: {BOOTSTRAP}")
    logger.info(f"  Window: {WINDOW_SIZE_SEC}s, Tick: {TICK_INTERVAL_SEC}s")
    logger.info(f"{'='*60}\n")
    
    msg_count = 0
    last_tick_time = time.time()
    
    try:
        while _running:
            msg = consumer.poll(0.5)
            
            if msg is None:
                # Check if time to process window
                current_time = time.time()
                if current_time - last_tick_time >= TICK_INTERVAL_SEC:
                    process_windows(cell_buffer, ue_buffer, converter, predictor, publisher)
                    last_tick_time = current_time
                continue
                
            if msg.error():
                logger.error(f"Consumer error: {msg.error()}")
                continue
            
            try:
                msg_count += 1
                val = msg.value()
                obj = _decode_value(val)
                
                # Parse E2SM message
                entity, rows = normalize_from_e2sm(obj)
                
                if entity and rows:
                    logger.debug(f"[msg #{msg_count}] {entity.upper()}: {len(rows)} rows")
                    
                    # Add to appropriate buffer (conversion happens here)
                    if entity == "cell":
                        cell_buffer.add_rows("cell", rows, converter)
                    elif entity == "ue":
                        ue_buffer.add_rows("ue", rows, converter)
                    
                consumer.commit(msg, asynchronous=True)
                
            except Exception as e:
                logger.error(f"Message processing error: {e}")
                import traceback
                traceback.print_exc()
                consumer.commit(msg, asynchronous=True)
                
    finally:
        logger.info("\nShutting down DQ Score xApp...")
        logger.info(f"Processed {msg_count} messages")
        
        try:
            publisher.close()
            consumer.close()
        except:
            pass
        
        logger.info("DQ Score xApp stopped.")

if __name__ == "__main__":
    consume_and_predict()