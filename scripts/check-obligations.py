#!/usr/bin/env python3
"""Check (or rebuild) obligations.lock against the tree.

obligations.lock pins the obligations in spec/extraction.md and every test
tagged `// obligation: <id>` under tests/, so a change to either is visible
and needs the approver to re-lock.

  scripts/check-obligations.py [verify]          exit 1 if the lock is stale
  scripts/check-obligations.py build <commit> <approver> > obligations.lock

Scheme (matches every lock since 15ad842):
  entry.hash         sha256 of the whitespace-collapsed text from the
                     `// obligation:` line to the closing brace of the
                     tagged fn
  obligations_digest sha256(json.dumps(<spec yaml>["obligations"], sort_keys=True))
  entries            sorted by (obligation, path, name)

Requires PyYAML. Run from the repository root.
"""
import hashlib
import json
import os
import re
import subprocess
import sys

try:
    import yaml
except ImportError:
    sys.exit("check-obligations: needs PyYAML (pip install pyyaml)")

SPEC = "spec/extraction.md"
LOCK = "obligations.lock"
TAG = re.compile(r"^// obligation: (SPEC-[\w.-]+)\n(?:#\[[^\n]*\]\n)*fn (\w+)", re.M)


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def fn_end(src, start):
    """Index just past the brace that closes the first block after `start`."""
    depth = 0
    for j in range(src.index("{", start), len(src)):
        depth += {"{": 1, "}": -1}.get(src[j], 0)
        if depth == 0:
            return j + 1
    raise ValueError("unbalanced braces")


def tagged_checks():
    out = []
    for f in sorted(os.listdir("tests")):
        if f.endswith(".rs"):
            src = open(os.path.join("tests", f), encoding="utf-8").read()
            for m in TAG.finditer(src):
                text = src[m.start():fn_end(src, m.end())]
                out.append({"obligation": m.group(1), "path": f"tests/{f}", "name": m.group(2), "lang": "rust",
                            "hash": sha(" ".join(text.split()))})
    return sorted(out, key=lambda e: (e["obligation"], e["path"], e["name"]))


def obligations():
    spec = open(SPEC, encoding="utf-8").read()
    m = re.search(r"```yaml\n(.*?)```", spec, re.S)
    if not m:
        sys.exit(f"check-obligations: no ```yaml block in {SPEC}")
    return yaml.safe_load(m.group(1))["obligations"]


def key(e):
    return (e["obligation"], e["path"], e["name"])


def verify():
    lock = json.load(open(LOCK, encoding="utf-8"))
    problems = []
    obs = obligations()
    if sha(json.dumps(obs, sort_keys=True)) != lock["obligations_digest"]:
        problems.append(f"{SPEC}: the obligations changed since the lock")
    ids = {o["id"] for o in obs}
    locked = {key(e): e["hash"] for e in lock["entries"]}
    now = {key(e): e["hash"] for e in tagged_checks()}
    for k in sorted(locked.keys() | now.keys()):
        where = f"{k[1]}::{k[2]} ({k[0]})"
        if k not in now:
            problems.append(f"{where}: locked check is missing or untagged")
        elif k not in locked:
            problems.append(f"{where}: tagged check is not in the lock")
        elif locked[k] != now[k]:
            problems.append(f"{where}: locked check was modified")
    for k in now:
        if k[0] not in ids:
            problems.append(f"{k[1]}::{k[2]}: tagged with unknown obligation {k[0]}")
    for o in sorted(ids - {k[0] for k in now}):
        problems.append(f"{o}: obligation has no tagged check")
    crit = lock.get("criteria_commit", "")
    r = subprocess.run(["git", "merge-base", "--is-ancestor", crit, "HEAD"], capture_output=True)
    if r.returncode not in (0, 1):
        print(f"check-obligations: note: could not check criteria_commit {crit[:12]} (shallow clone?)")
    elif r.returncode == 1:
        problems.append(f"criteria_commit {crit[:12]} is not an ancestor of HEAD")
    if problems:
        print(f"check-obligations: {LOCK} is stale ({len(problems)} problem(s)):")
        for p in problems:
            print(f"  - {p}")
        print("The approver must review these changes and re-lock:\n"
              "  scripts/check-obligations.py build <criteria commit> <approver> > obligations.lock")
        return 1
    print(f"check-obligations: ok ({len(now)} checks, {len(ids)} obligations, approver {lock['approver']})")
    return 0


def build(commit, approver):
    lock = {"version": 2, "approver": approver, "criteria_commit": commit,
            "obligations_digest": sha(json.dumps(obligations(), sort_keys=True)), "entries": tagged_checks()}
    print(json.dumps(lock, indent=2))
    return 0


if __name__ == "__main__":
    args = sys.argv[1:] or ["verify"]
    if args[0] == "verify" and len(args) == 1:
        sys.exit(verify())
    if args[0] == "build" and len(args) == 3:
        sys.exit(build(args[1], args[2]))
    sys.exit(__doc__)
