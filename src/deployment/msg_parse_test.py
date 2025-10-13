#!/usr/bin/env python3
"""
Test Kafka Message Parsing
"""
import json
from confluent_kafka import Consumer
from datetime import datetime, timezone

BOOTSTRAP = "my-kafka-controller-0.my-kafka-controller-headless.default.svc.cluster.local:9092"
TOPIC = "e2-data"

def decode_value(v):
    """Decode Kafka message"""
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
            return None
    except Exception as e:
        print(f"❌ Decode error: {e}")
        return None

def test_parsing():
    """Test message parsing"""
    print("=" * 60)
    print("KAFKA MESSAGE PARSING TEST")
    print("=" * 60)
    
    # Create consumer
    consumer = Consumer({
        "bootstrap.servers": BOOTSTRAP,
        "group.id": "test-parser",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    })
    
    consumer.subscribe([TOPIC])
    
    print(f"\n📡 Consuming from topic: {TOPIC}")
    print(f"🔗 Bootstrap: {BOOTSTRAP}\n")
    
    msg_count = 0
    max_messages = 5
    
    try:
        while msg_count < max_messages:
            msg = consumer.poll(2.0)
            
            if msg is None:
                print("⏳ Waiting for messages...")
                continue
                
            if msg.error():
                print(f"❌ Consumer error: {msg.error()}")
                continue
            
            msg_count += 1
            print(f"\n{'─'*60}")
            print(f"📨 MESSAGE #{msg_count}")
            print(f"{'─'*60}")
            
            # Decode
            val = msg.value()
            print(f"Raw value type: {type(val)}")
            print(f"Raw value length: {len(val) if val else 0} bytes")
            
            obj = decode_value(val)
            
            if obj is None:
                print("❌ Failed to decode JSON")
                continue
            
            print(f"✅ Decoded JSON successfully")
            
            # Check structure
            print(f"\n🔍 TOP-LEVEL KEYS:")
            for key in obj.keys():
                print(f"  - {key}")
            
            # Check metadata
            meta = obj.get("metadata", {})
            print(f"\n📋 METADATA:")
            print(f"  - event_ts_ms: {meta.get('event_ts_ms')}")
            print(f"  - entity: {meta.get('entity')}")
            
            # Check Format1 (cell)
            fmt1 = obj.get("indicationMessage-Format1")
            if fmt1:
                print(f"\n📊 FORMAT1 (CELL):")
                meas_data = fmt1.get("measData", [])
                print(f"  - measData entries: {len(meas_data)}")
                
                if meas_data:
                    first = meas_data[0]
                    print(f"  - First entry keys: {list(first.keys())}")
                    print(f"  - measRecord length: {len(first.get('measRecord', []))}")
                    
                    # Check metadata in measData
                    md_meta = first.get("metadata", {})
                    print(f"  - metadata keys: {list(md_meta.keys())}")
                    print(f"  - cell_name: {md_meta.get('cell_name')}")
            
            # Check Format3 (ue)
            fmt3 = obj.get("indicationMessage-Format3")
            if fmt3:
                print(f"\n📊 FORMAT3 (UE):")
                ue_data = fmt3.get("ueMeasData", [])
                print(f"  - ueMeasData entries: {len(ue_data)}")
                
                if ue_data:
                    first = ue_data[0]
                    print(f"  - First entry keys: {list(first.keys())}")
                    ue_id = first.get("ueID", {}).get("gNB-UEID", {})
                    print(f"  - amf-UE-NGAP-ID: {ue_id.get('amf-UE-NGAP-ID')}")
            
            # Check normalized_rows (NEW FORMAT)
            norm_rows = obj.get("normalized_rows", [])
            if norm_rows:
                print(f"\n✨ NORMALIZED_ROWS:")
                print(f"  - Count: {len(norm_rows)}")
                if norm_rows:
                    first = norm_rows[0]
                    print(f"  - First row keys: {list(first.keys())}")
                    print(f"  - Sample row: {json.dumps(first, indent=4)[:200]}...")
            
            print(f"\n{'='*60}\n")
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped by user")
    
    finally:
        consumer.close()
        print(f"\n✅ Tested {msg_count} messages")

if __name__ == "__main__":
    test_parsing()