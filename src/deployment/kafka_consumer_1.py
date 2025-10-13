import json
import signal
from datetime import datetime, timezone
from typing import Tuple, List, Dict, Optional
import os
from confluent_kafka import Consumer

# Import your mapping
from common.constants import KPM_TO_CANON

BOOTSTRAP = os.getenv("KAFKA_BROKERS", "my-kafka-controller-0.my-kafka-controller-headless.default.svc.cluster.local:9092")
IN_TOPIC = "e2-data"
GROUP_ID = "dqscore-xapp"

_running = True

def _stop(*_):
    global _running
    _running = False

def _iso_from_ms(ms: int) -> str:
    """Convert milliseconds timestamp to ISO format"""
    return datetime.fromtimestamp(ms/1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _decode_value(v):
    """Decode Kafka message value with proper error handling"""
    try:
        if isinstance(v, (bytes, bytearray)):
            b = bytes(v)
            # Check for 4-byte length prefix (common in some protocols)
            if len(b) >= 4 and b[0] != ord('{') and b[0] != ord('['):
                b = b[4:]  # strip 4-byte length prefix
            
            # FIX: errors parameter goes in decode(), not json.loads()
            decoded_str = b.decode("utf-8", errors='replace')  # or 'ignore' or 'strict'
            return json.loads(decoded_str)
            
        elif isinstance(v, str):
            return json.loads(v)
        else:
            raise TypeError(f"Unsupported value type: {type(v)}")
            
    except json.JSONDecodeError as e:
        print(f"[kafka] JSON decode error: {e}")
        print(f"[kafka] First 200 chars: {str(v[:200])}")
        raise
    except UnicodeDecodeError as e:
        print(f"[kafka] Unicode decode error: {e}")
        raise

def normalize_message(msg: dict) -> Tuple[Optional[str], List[Dict]]:
    """
    Normalize E2SM-KPM messages to canonical format
    Returns (entity, rows) where entity is "cell" or "ue"
    """
    try:
        meta = msg.get("metadata", {})
        if "event_ts_ms" not in meta:
            print("[normalize] No event_ts_ms in metadata")
            return None, []
            
        ts_iso = _iso_from_ms(int(meta["event_ts_ms"]))
        rows: List[Dict] = []

        # -------- CELL (Format1) --------
        fmt1 = msg.get("indicationMessage-Format1")
        if fmt1:
            # Get ordered metric names from measInfoList
            names: List[str] = []
            for mi in fmt1.get("measInfoList", []):
                mt = mi.get("measType", {})
                name = mt.get("measName") or mt.get("measTypeName")
                if name:
                    names.append(name)

            # Process each cell's measurements
            for md in fmt1.get("measData", []):
                rec_meta = md.get("metadata", {}) or {}
                rec = {
                    "timestamp": ts_iso,
                    "Viavi.Cell.Name": rec_meta.get("cell_name"),
                }
                
                # Add band if available
                if "band" in rec_meta:
                    rec["band"] = rec_meta["band"]

                # Extract measurement values
                vals: List[float] = []
                for x in md.get("measRecord", []):
                    if isinstance(x, dict):
                        val = x.get("real") or x.get("integer") or x.get("measValue")
                        vals.append(val if val is not None else 0.0)
                    else:
                        vals.append(float(x) if x is not None else 0.0)

                # Map to canonical names
                for n, v in zip(names, vals):
                    canon = KPM_TO_CANON.get(n)
                    if canon:
                        rec[canon] = v
                        
                rows.append(rec)
            return "cell", rows

        # -------- UE (Format3) --------
        fmt3 = msg.get("indicationMessage-Format3")
        if fmt3:
            # Get ordered metric names
            names: List[str] = []
            for mi in fmt3.get("measInfoList", []):
                mt = mi.get("measType", {})
                name = mt.get("measName") or mt.get("measTypeName")
                if name:
                    names.append(name)

            # Process each UE's measurements
            for ue_data in fmt3.get("ueMeasData", []):
                # Extract UE ID
                ue_id_obj = ue_data.get("ueID", {})
                gnb_ue = ue_id_obj.get("gNB-UEID", {})
                ue_name = gnb_ue.get("amf-UE-NGAP-ID")

                # Extract measurement values
                vals: List[float] = []
                meas_report = ue_data.get("measReport", {}) or {}
                for report_item in meas_report.get("measReportList", []):
                    for x in report_item.get("measRecord", []):
                        if isinstance(x, dict):
                            val = x.get("real") or x.get("integer") or x.get("measValue")
                            vals.append(val if val is not None else 0.0)
                        else:
                            vals.append(float(x) if x is not None else 0.0)

                rec = {
                    "timestamp": ts_iso,
                    "Viavi.UE.Name": ue_name
                }
                
                # Map to canonical names
                for n, v in zip(names, vals):
                    canon = KPM_TO_CANON.get(n)
                    if canon:
                        rec[canon] = v
                        
                rows.append(rec)
            return "ue", rows

        print("[normalize] No recognized format in message")
        return None, []
        
    except Exception as e:
        print(f"[normalize] Error normalizing message: {e}")
        import traceback
        traceback.print_exc()
        return None, []

def build_consumer():
    """Build Kafka consumer with optimized settings"""
    return Consumer({
        "bootstrap.servers": BOOTSTRAP,
        "group.id": GROUP_ID,
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
        "session.timeout.ms": 45000,
        "max.poll.interval.ms": 300000,
        "fetch.min.bytes": 1,
        "fetch.wait.max.ms": 100,
    })

def consume_and_normalize():
    """Main consumer loop with proper error handling"""
    global _running
    c = build_consumer()
    c.subscribe([IN_TOPIC])
    
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    print(f"[kafka] Consuming topic={IN_TOPIC}")
    print(f"[kafka] Bootstrap={BOOTSTRAP}")
    print(f"[kafka] Group={GROUP_ID}")
    print(f"[kafka] Press Ctrl+C to stop\n")

    msg_count = 0
    error_count = 0

    try:
        while _running:
            msg = c.poll(0.5)
            
            if msg is None:
                continue
                
            if msg.error():
                print(f"[kafka] Consumer error: {msg.error()}")
                error_count += 1
                continue

            try:
                msg_count += 1
                val = msg.value()
                
                # Decode the message
                obj = _decode_value(val)
                
                # Check if already normalized (has normalized_rows field)
                if isinstance(obj, dict) and "normalized_rows" in obj:
                    entity = obj.get("metadata", {}).get("entity")
                    rows = obj.get("normalized_rows") or []
                    
                    if rows:
                        print(f"[msg #{msg_count}] {entity.upper()}: {len(rows)} rows (pre-normalized)")
                        print(f"  Sample keys: {list(rows[0].keys())[:8]}")
                        
                        # HERE: Add your processing logic
                        # For example, append to buffers, write to database, etc.
                        # process_rows(entity, rows)
                        
                    c.commit(msg, asynchronous=True)
                    continue

                # Otherwise, normalize it ourselves
                entity, rows = normalize_message(obj)
                
                if entity and rows:
                    print(f"[msg #{msg_count}] {entity.upper()}: {len(rows)} rows (normalized)")
                    print(f"  Sample keys: {list(rows[0].keys())[:8]}")
                    
                    # HERE: Add your processing logic
                    # process_rows(entity, rows)
                    
                    c.commit(msg, asynchronous=True)
                else:
                    print(f"[msg #{msg_count}] Could not normalize message")
                    
            except Exception as e:
                error_count += 1
                print(f"[kafka] Processing error (msg #{msg_count}): {e}")
                import traceback
                traceback.print_exc()
                # Still commit to avoid reprocessing bad messages repeatedly
                c.commit(msg, asynchronous=True)
                
    finally:
        print(f"\n[kafka] Shutting down...")
        print(f"[kafka] Processed: {msg_count} messages, Errors: {error_count}")
        try:
            c.close()
        except Exception as e:
            print(f"[kafka] Error closing consumer: {e}")

if __name__ == "__main__":
    consume_and_normalize()