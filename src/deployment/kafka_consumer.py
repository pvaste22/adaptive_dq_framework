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

#WINDOW_SIZE_SEC = 300  # 5 minutes
#TICK_INTERVAL_SEC = 60  # 1 minute step
WINDOW_SIZE_SEC   = int(os.getenv("WINDOW_SIZE_SEC", "300"))
TICK_INTERVAL_SEC = int(os.getenv("TICK_INTERVAL_SEC", "60"))
COALESCE_MS = int(os.getenv("COALESCE_MS", "700"))  # 0.5–1.0s is good

# coalesce state
_COALESCE_TARGET_END_MS = None
_COALESCE_UNTIL_MS = 0
_COALESCE_SEEN_CELL = False
_COALESCE_SEEN_UE = False

_running = True
_LAST_WINDOW_END_MS = None


# ============ WINDOWED BUFFER ============
class TimeWindowBuffer:
    """Maintains time-windowed data per entity"""
    
    def __init__(self, window_size_sec: int = 300):
        self.window_size = timedelta(seconds=window_size_sec)
        # {entity_id: deque of (timestamp, row_dict)}
        self.buffers: Dict[str, deque] = defaultdict(lambda: deque())
        
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
                #self.buffers[entity_id].append((ts, row))
                kafka_ts_ms = row.get("kafka_ts_ms")
                self.buffers[entity_id].append((ts, kafka_ts_ms, row))
    
    def get_window_data(self, entity_id: str, end_time) -> List[Dict]:
        """Get rows in [end-W, end] by *ingestion time* (kafka_ts_ms), right-edge inclusive"""
        # accept datetime or ms
        if isinstance(end_time, int):
            end_ms = int(end_time)
        else:
            # datetime -> ms
            end_ms = int(end_time.timestamp() * 1000)

        start_ms = end_ms - int(self.window_size.total_seconds()*1000) + 1
        out = []
        for ts, kafka_ts_ms, row in self.buffers[entity_id]:
            if kafka_ts_ms is None:
                continue
            k = int(kafka_ts_ms)
            if start_ms <= k <= end_ms:         # RIGHT-EDGE inclusive
                out.append(row)
        return out
    
    def get_all_entities(self) -> List[str]:
        """Get list of all entity IDs being tracked"""
        return list(self.buffers.keys())
    
    def cleanup_by_kafka_cutoff(self, cutoff_ms: int):
        """Drop all buffered rows with kafka_ts_ms <= cutoff_ms (by entity)"""
        for entity_id in list(self.buffers.keys()):
            dq = self.buffers[entity_id]
            while dq and dq[0][1] is not None and int(dq[0][1]) <= cutoff_ms:
                dq.popleft()
            if not dq:
                del self.buffers[entity_id]


    def cleanup_old_data(self, cutoff_time: datetime):
        """Remove data older than cutoff"""
        for entity_id in list(self.buffers.keys()):
            while self.buffers[entity_id]:
                ts, _, _ = self.buffers[entity_id][0]
                if ts < cutoff_time:
                    self.buffers[entity_id].popleft()
                else:
                    break
            
            if not self.buffers[entity_id]:
                del self.buffers[entity_id]

    def max_ingest_ms(self) -> Optional[int]:
        m = None
        for dq in self.buffers.values():
            for _, kafka_ts_ms, _ in dq:
                if kafka_ts_ms is not None:
                    m = kafka_ts_ms if m is None or kafka_ts_ms > m else m
        return m

# ============ KAFKA HELPERS ============
def _stop(*_):
    global _running
    _running = False

def _iso_from_ms(ms: int) -> str:
    return datetime.fromtimestamp(ms/1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _decode_value(v):
    """Decode Kafka message value into JSON dict (or None on failure)."""
    if not v:
        return None

    # bytes normalize
    try:
        b = v if isinstance(v, (bytes, bytearray)) else v.encode("utf-8")
    except Exception:
        return None

    # common 4-byte length prefix case
    if len(b) >= 4 and b[0] not in (ord('{'), ord('[')):
        b = b[4:]

    # find first JSON start in bytes
    lb, ls = b.find(b'{'), b.find(b'[')
    starts = [x for x in (lb, ls) if x != -1]
    if not starts:
        return None
    b = b[min(starts):]

    # strict UTF-8; if fails, skip
    try:
        s = b.decode("utf-8")
    except UnicodeDecodeError:
        return None

    s = s.strip()
    if not s or s[0] not in ('{', '['):
        return None

    # trim trailing garbage after last closing brace/bracket
    closer = '}' if s[0] == '{' else ']'
    end = s.rfind(closer)
    if end != -1:
        s = s[:end+1]

    # parse JSON; fallback to line-by-line if needed
    try:
        return json.loads(s)
    except Exception:
        for line in s.splitlines():
            line = line.strip()
            if line and line[0] in ('{','['):
                try:
                    return json.loads(line)
                except Exception:
                    pass
        return None

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

def _floor_to_minute_ms(ms: int) -> int:
    return ms - (ms % 60000)

def validate_features(features: Dict[str, float], predictor: DQScorePredictor):
    """Diagnostic: Check feature alignment"""
    expected_features = set(predictor.feature_names or [])
    actual_features = set(features.keys())
    
    missing = expected_features - actual_features
    extra = actual_features - expected_features
    
    if missing:
        logger.warning(f"Missing {len(missing)} features: {list(missing)[:5]}...")
    if extra:
        logger.info(f"Extra {len(extra)} features (will be ignored): {list(extra)[:5]}...")
    numeric_features = {
        k: v for k, v in features.items() 
        if isinstance(v, (int, float, np.integer, np.floating))
    }
    # Check for NaN/inf values
    bad_values = {k: v for k, v in numeric_features.items() if not np.isfinite(v)}
    if bad_values:
        logger.warning(f"Non-finite values in features: {bad_values}  features: {list(bad_values.keys())[:5]}...")

def _reset_coalesce():
    global _COALESCE_TARGET_END_MS, _COALESCE_UNTIL_MS, _COALESCE_SEEN_CELL, _COALESCE_SEEN_UE
    _COALESCE_TARGET_END_MS = None
    _COALESCE_UNTIL_MS = 0
    _COALESCE_SEEN_CELL = False
    _COALESCE_SEEN_UE = False

# ============ WINDOW PROCESSOR ============
def process_windows(cell_buffer: TimeWindowBuffer, 
                   ue_buffer: TimeWindowBuffer,
                   converter: UnitConverter,
                   predictor: DQScorePredictor,
                   publisher: ScorePublisher):
    """Process all windows and compute DQ scores"""
    
    global _LAST_WINDOW_END_MS

    # 1) latest ingestion ts across streams (publish-ASAP → use MAX watermark)
    cell_max = cell_buffer.max_ingest_ms()
    ue_max   = ue_buffer.max_ingest_ms()
    ingest_max_ms = max([x for x in (cell_max, ue_max) if x is not None], default=None)
    if ingest_max_ms is None:
        return  # no data yet

    # 2) close window at the last fully-ingested minute (RIGHT-EDGE inclusive)
    window_end_ms = _floor_to_minute_ms(ingest_max_ms) + 60000 - 1

    # 3) enforce 1-min hop (avoid duplicate emits)
    if _LAST_WINDOW_END_MS is not None and window_end_ms <= _LAST_WINDOW_END_MS:
        return
    if _LAST_WINDOW_END_MS is not None and window_end_ms - _LAST_WINDOW_END_MS < 60000:
        return

    _LAST_WINDOW_END_MS = window_end_ms
    window_end   = datetime.fromtimestamp(window_end_ms/1000, tz=timezone.utc)
    window_start = window_end - timedelta(seconds=WINDOW_SIZE_SEC)
    window_id    = window_end.strftime("%Y%m%d_%H%M%S")

    logger.info(f"\n{'='*60}")
    logger.info(f"[TICK] Processing window: {window_id}")
    logger.info(f"{'='*60}")

    # Collect rows by *ingestion-time* in [end-W, end] (expects get_window_data to use kafka_ts_ms)
    cell_entities = cell_buffer.get_all_entities()
    ue_entities   = ue_buffer.get_all_entities()

    all_cell_rows = []
    for entity_id in cell_entities:
        all_cell_rows.extend(cell_buffer.get_window_data(entity_id, window_end_ms))  # pass ms

    all_ue_rows = []
    for entity_id in ue_entities:
        all_ue_rows.extend(ue_buffer.get_window_data(entity_id, window_end_ms))      # pass ms

    logger.info(f"Window data: {len(all_cell_rows)} cell rows, {len(all_ue_rows)} UE rows")

    # Derive diagnostics (event-time only for metadata; selection is ingestion-time)
    all_rows = (all_cell_rows or []) + (all_ue_rows or [])
    if all_rows:
        last_event_ts = max(datetime.fromisoformat(r["timestamp"].replace('Z','+00:00'))
                            for r in all_rows).astimezone(timezone.utc)
        kafka_ts_vals = [int(r["kafka_ts_ms"]) for r in all_rows if "kafka_ts_ms" in r]
        last_kafka_ts_ms = max(kafka_ts_vals) if kafka_ts_vals else None
    else:
        last_event_ts = None
        last_kafka_ts_ms = None

    try:
        # DataFrames + unit conversion
        cell_df = pd.DataFrame(all_cell_rows)
        ue_df   = pd.DataFrame(all_ue_rows)
        cell_df, ue_df = converter.standardize_units_comprehensive(cell_df, ue_df)
        logger.info(" Unit conversion applied")

        # Window metadata payload
        window_data = {
            'cell_data': cell_df,
            'ue_data': ue_df,
            'metadata': {
                'window_id': window_id,
                'window_start_time': window_start.timestamp(),
                'window_end_time': window_end.timestamp(),
                'start_time': window_start.isoformat(),
                'end_time': window_end.isoformat(),
                'last_event_ts': last_event_ts.isoformat() if last_event_ts else None,
                'last_kafka_ts_ms': last_kafka_ts_ms,
            }
        }

        # Feature → predict → timings
        t0 = time.time()
        features = make_feature_row(window_data)
        validate_features(features, predictor)
        dq_score = predictor.predict(features)
        processing_time_ms = int((time.time() - t0) * 1000)

        # Publish
        publish_ts_ms = int(time.time() * 1000)
        latency_ms = (max(0, publish_ts_ms - int(last_kafka_ts_ms))
                      if last_kafka_ts_ms is not None else None)
        publisher.publish_score(
            window_id=window_id,
            start_time=window_start.isoformat(),
            end_time=window_end.isoformat(),
            dq_score=dq_score,
            metadata={
                'cell_entities': len(cell_entities),
                'ue_entities': len(ue_entities),
                'cell_rows': len(all_cell_rows),
                'ue_rows': len(all_ue_rows),
                'last_event_ts': last_event_ts.isoformat() if last_event_ts else None,
                'last_kafka_ts_ms': last_kafka_ts_ms,
                'publish_ts_ms': publish_ts_ms,
                'latency_ms': latency_ms,
                'processing_time_ms': processing_time_ms,
            }
        )
        logger.info(f" Predicted DQ Score: {dq_score:.4f}")
        logger.info(f" Published score to '{OUT_TOPIC}' topic")

    except Exception as e:
        logger.error(f"Window processing failed: {e}")
        import traceback; traceback.print_exc()

    # 4) Cleanup by *ingestion cutoff* so late arrivals can't re-enter past windows
    try:
        _hist_retain_ms = (WINDOW_SIZE_SEC - 60) * 1000  # keep last W-60s for next window
        cutoff_ms = window_end_ms - _hist_retain_ms
        cell_buffer.cleanup_by_kafka_cutoff(cutoff_ms)   # drop k_ts < cutoff_ms
        ue_buffer.cleanup_by_kafka_cutoff(cutoff_ms)
    except AttributeError:
        # fallback: time-based old-data cleanup if cutoff method not present
        cutoff = window_end - timedelta(seconds=WINDOW_SIZE_SEC * 2)
        cell_buffer.cleanup_old_data(cutoff)
        ue_buffer.cleanup_old_data(cutoff)

    logger.info(f"{'='*60}\n")


# ============ MAIN CONSUMER LOOP ============
def consume_and_predict():
    """Main consumer loop with prediction pipeline"""
    global _running
    global _COALESCE_TARGET_END_MS, _COALESCE_UNTIL_MS, _COALESCE_SEEN_CELL, _COALESCE_SEEN_UE

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
    
    now = time.time()
    align = 60 - (int(now) % 60)
    if 0 < align < 60:
        time.sleep(align)
    msg_count = 0
    last_tick_time = time.time()

    
    try:
        while _running:

            now_ms = int(time.time() * 1000)
            if _COALESCE_TARGET_END_MS is not None and now_ms >= _COALESCE_UNTIL_MS:
                process_windows(cell_buffer, ue_buffer, converter, predictor, publisher)
                _reset_coalesce()
                last_tick_time = time.time()

            # Check if time to process window
            current_time = time.time()
            if current_time - last_tick_time >= TICK_INTERVAL_SEC:
                process_windows(cell_buffer, ue_buffer, converter, predictor, publisher)
                last_tick_time = current_time
            msg = consumer.poll(0.5)    
            if msg is None:
                continue
                
            if msg.error():
                logger.error(f"Consumer error: {msg.error()}")
                continue
            
            try:
                _, kafka_ts_ms = msg.timestamp()
                msg_count += 1
                val = msg.value()
                obj = _decode_value(val)

                if obj is None:
                    consumer.commit(msg, asynchronous=True)
                    continue
                
                # Parse E2SM message
                entity, rows = normalize_from_e2sm(obj)
                
                if entity and rows:
                    logger.debug(f"[msg #{msg_count}] {entity.upper()}: {len(rows)} rows")
                    for r in rows:
                        r["kafka_ts_ms"] = int(kafka_ts_ms)
                    # Add to appropriate buffer (conversion happens here)
                    if entity == "cell":
                        cell_buffer.add_rows("cell", rows, converter)
                        _COALESCE_SEEN_CELL = True
                    elif entity == "ue":
                        ue_buffer.add_rows("ue", rows, converter)
                        _COALESCE_SEEN_UE = True
                cell_max = cell_buffer.max_ingest_ms()
                ue_max   = ue_buffer.max_ingest_ms()
                if cell_max is not None or ue_max is not None:
                    #new_end_ms = _floor_to_minute_ms(max([x for x in (cell_max, ue_max) if x is not None])) + 60000 - 1 #min(cell_max, ue_max))
                    cand_end_ms = _floor_to_minute_ms(max([x for x in (cell_max, ue_max) if x is not None])) + 60000 - 1
                    if _LAST_WINDOW_END_MS is None or cand_end_ms - _LAST_WINDOW_END_MS >= 60000:
                        now_ms = int(time.time() * 1000)

                        # new target minute? start/refresh coalesce window
                        if _COALESCE_TARGET_END_MS != cand_end_ms:
                            _COALESCE_TARGET_END_MS = cand_end_ms
                            _COALESCE_UNTIL_MS = now_ms + COALESCE_MS

                    
                        if (_COALESCE_SEEN_CELL and _COALESCE_SEEN_UE) or now_ms >= _COALESCE_UNTIL_MS:
                            process_windows(cell_buffer, ue_buffer, converter, predictor, publisher)
                            _reset_coalesce()
                            last_tick_time = time.time()                    
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