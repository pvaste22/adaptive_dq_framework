
import json
import signal
from datetime import datetime, timezone
from typing import Tuple, List, Dict

from confluent_kafka import Consumer
from common.constants import KPM_TO_CANON   # <-- aapke repo ka mapping (RRU.*, DRB.*, PEE.*, TB.*)

BOOTSTRAP = "localhost:9092"   # aapke setup ke hisaab se (kubectl exec ke through same pod)
IN_TOPIC  = "e2-data"
GROUP_ID  = "dqscore-xapp"

_running = True
def _stop(*_):  # graceful shutdown
    global _running
    _running = False

def _iso_from_ms(ms: int) -> str:
    return datetime.fromtimestamp(ms/1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def normalize_message(msg: dict) -> Tuple[str, List[Dict]]:
    """
    Returns (entity, rows)
      entity: "cell" | "ue"
      rows: list of dicts with training-canonical columns:
        - "timestamp" (ISO from metadata.event_ts_ms)
        - "Viavi.Cell.Name" / "Viavi.UE.Name"
        - (cell) optional "band"
        - canonical metric keys via KPM_TO_CANON
    """
    meta = msg.get("metadata", {})
    if "event_ts_ms" not in meta:
        return None, []
    ts_iso = _iso_from_ms(int(meta["event_ts_ms"]))
    rows: List[Dict] = []

    # -------- CELL (Format1) --------
    fmt1 = msg.get("indicationMessage-Format1")
    if fmt1:
        # ordered metric names
        names: List[str] = []
        for mi in fmt1.get("measInfoList", []):
            for mt in mi.get("measTypeList", []):
                names.append(mt.get("measName") or mt.get("measTypeName"))

        # per-record values + per-row identity (from measData[].metadata)
        for mi in fmt1.get("measInfoList", []):
            for md in mi.get("measData", []):
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
                        vals.append(x.get("measValue") or x.get("real") or x.get("integer"))
                    else:
                        vals.append(x)

                for n, v in zip(names, vals):
                    canon = KPM_TO_CANON.get(n)
                    if canon is not None:
                        rec[canon] = v
                rows.append(rec)
        return "cell", rows

    # -------- UE (Format3) --------
    fmt3 = msg.get("indicationMessage-Format3")
    if fmt3:
        # UE identity (latest structure): ueMeasData[].ueID.gNB-UEID.amf-UE-NGAP-ID
        ue_name = None
        # prefer per-UE id if present; else top-level ueID
        if fmt3.get("ueMeasData"):
            ue_name = (
                fmt3["ueMeasData"][0]
                .get("ueID", {})
                .get("gNB-UEID", {})
                .get("amf-UE-NGAP-ID")
            )
        if ue_name is None:
            ue_name = (
                fmt3.get("ueID", {})
                .get("gNB-UEID", {})
                .get("amf-UE-NGAP-ID")
            )

        # names in measInfoList[].measTypeList[]
        names: List[str] = []
        for mi in fmt3.get("measInfoList", []):
            for mt in mi.get("measTypeList", []):
                names.append(mt.get("measName") or mt.get("measTypeName"))

        # values: ueMeasData[].measReport.measReportList[].measRecord[]
        for ue in fmt3.get("ueMeasData", []):
            vals: List[float] = []
            rep = ue.get("measReport", {}) or {}
            for block in rep.get("measReportList", []):
                for x in block.get("measRecord", []):
                    if isinstance(x, dict):
                        vals.append(x.get("measValue") or x.get("real") or x.get("integer"))
                    else:
                        vals.append(x)

            rec = {"timestamp": ts_iso, "Viavi.UE.Name": ue_name}
            for n, v in zip(names, vals):
                canon = KPM_TO_CANON.get(n)
                if canon is not None:
                    rec[canon] = v
            rows.append(rec)
        return "ue", rows

    return None, []

def build_consumer():
    return Consumer({
        "bootstrap.servers": BOOTSTRAP,
        "group.id": GROUP_ID,
        "enable.auto.commit": False,      # commit after successful handle
        "auto.offset.reset": "earliest",
        "session.timeout.ms": 45000,
        "max.poll.interval.ms": 300000,
    })

def consume_and_normalize():
    global _running
    c = build_consumer()
    c.subscribe([IN_TOPIC])
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    print(f"[kafka] consuming topic={IN_TOPIC} on {BOOTSTRAP} as group={GROUP_ID}")

    try:
        while _running:
            msg = c.poll(0.25)
            if msg is None:
                continue
            if msg.error():
                print(f"[kafka] error: {msg.error()}")
                continue

            try:
                obj = json.loads(msg.value())
                entity, rows = normalize_message(obj)
                if entity and rows:
                    # ⬇️ yahin pe next step me window buffers me add karenge
                    print(f"[normalize] {entity}: rows={len(rows)} keys={list(rows[0].keys())[:6]}")
                c.commit(msg, asynchronous=True)
            except Exception as e:
                print(f"[kafka] processing error: {e}")
    finally:
        try:
            c.close()
        except Exception:
            pass

if __name__ == "__main__":
    consume_and_normalize()