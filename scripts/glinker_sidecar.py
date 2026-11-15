"""GLinker sidecar for lemmalog's GlinkerHttpLinker (feature `glinker`).

GLinker is Python, so lemmalog reaches it over HTTP. The spans come from
gline-rs. GLinker runs in external-entities mode (L2 -> L3 -> L0) and only
picks the entity for each span.

    pip install glinker
    python scripts/glinker_sidecar.py --entities data/entities.jsonl --port 8765

POST /link
  {"texts": ["Bob works at Acme."],
   "entities": [[{"text": "Bob", "start": 0, "end": 3, "label": "person"}, ...]]}
  -> {"links": [[{"entity_id": "person:bob", "confidence": 0.97}, ...]]}

Offsets are in characters. A span GLinker cannot link comes back as
entity_id "unknown", and lemmalog then emits no fact for it.
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from glinker import ProcessorFactory


def build(entities, model, threshold):
    ex = ProcessorFactory.create_simple(model_name=model, threshold=threshold,
                                        entities=entities, external_entities=True)
    return ex


def link(executor, texts, entities):
    result = executor.execute({"texts": texts, "entities": entities})
    l0 = result.get("l0_result")
    out = []
    for spans, found in zip(entities, l0.entities):
        by_pos = {(e.mention_start, e.mention_end): e for e in found}
        row = []
        for s in spans:
            e = by_pos.get((s["start"], s["end"]))
            le = getattr(e, "linked_entity", None) if e else None
            row.append({"entity_id": le.entity_id if le else "unknown",
                        "confidence": float(le.confidence) if le else 0.0})
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", required=True, help="JSONL: entity_id, label, aliases, description")
    ap.add_argument("--model", default="knowledgator/gliner-linker-base-v1.0")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    executor = build(a.entities, a.model, a.threshold)

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/link":
                self.send_error(404)
                return
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            body = json.dumps({"links": link(executor, req["texts"], req["entities"])}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
