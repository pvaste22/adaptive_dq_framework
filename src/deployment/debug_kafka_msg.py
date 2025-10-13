# debug_kafka_messages.py
"""Debug script to check what's actually in Kafka"""

import json
from confluent_kafka import Consumer
import os

BOOTSTRAP = os.getenv("KAFKA_BROKERS", "localhost:9092")
TOPIC = "e2-data"

def debug_messages(max_messages=5):
    """Read and print first few messages for debugging"""
    
    c = Consumer({
        "bootstrap.servers": BOOTSTRAP,
        "group.id": "debug-group",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    })
    
    c.subscribe([TOPIC])
    print(f"Reading from {TOPIC} on {BOOTSTRAP}\n")
    
    count = 0
    try:
        while count < max_messages:
            msg = c.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print(f"Error: {msg.error()}")
                continue
                
            count += 1
            print(f"\n{'='*60}")
            print(f"Message #{count}")
            print(f"{'='*60}")
            
            val = msg.value()
            print(f"Type: {type(val)}")
            print(f"Length: {len(val)} bytes")
            
            # Try to decode
            try:
                if isinstance(val, bytes):
                    # Check for length prefix
                    if len(val) >= 4:
                        prefix = val[:4]
                        print(f"First 4 bytes: {prefix.hex()}")
                        
                        # Try with and without prefix
                        try:
                            obj = json.loads(val[4:].decode('utf-8'))
                            print("✓ Decoded WITH 4-byte prefix")
                        except:
                            obj = json.loads(val.decode('utf-8'))
                            print("✓ Decoded WITHOUT prefix")
                    else:
                        obj = json.loads(val.decode('utf-8'))
                        print("✓ Decoded directly")
                else:
                    obj = json.loads(val)
                    print("✓ Decoded from string")
                
                # Print structure
                print(f"\nTop-level keys: {list(obj.keys())}")
                
                if "metadata" in obj:
                    print(f"Metadata: {obj['metadata']}")
                
                if "normalized_rows" in obj:
                    rows = obj["normalized_rows"]
                    print(f"Normalized rows: {len(rows)}")
                    if rows:
                        print(f"Sample row keys: {list(rows[0].keys())}")
                        print(f"Sample row: {rows[0]}")
                
                # Print first 500 chars of JSON
                print(f"\nJSON preview:")
                print(json.dumps(obj, indent=2)[:500])
                print("...")
                
            except Exception as e:
                print(f"✗ Decode failed: {e}")
                print(f"Raw preview: {str(val[:200])}")
                
    finally:
        c.close()

if __name__ == "__main__":
    debug_messages(5)