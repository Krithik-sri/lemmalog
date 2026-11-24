#!/usr/bin/env python3
"""Check (or rebuild) obligations.lock against the tree.

obligations.lock pins the obligations in spec/extraction.md, every test
tagged `// obligation: <id>` under tests/, the helpers those tests use (top-
level fns, structs, enums, impls, consts in the same file, followed
transitively), and the data files the obligations list under `fixtures:`.
A change to any of them is visible and needs the approver to re-lock. A
tagged test or helper that names a data file no obligation declares as a
fixture is also reported.

  scripts/check-obligations.py [verify]          exit 1 if the lock is stale
  scripts/check-obligations.py build <commit> <approver> > obligations.lock

Scheme (entries and digest unchanged since 15ad842; helpers and fixtures
since lock version 3):
  entry.hash         sha256 of the whitespace-collapsed text from the
                     `// obligation:` line to the closing brace of the
                     tagged fn
  helper.hash        sha256 of the whitespace-collapsed item text
  fixture.hash       sha256 of the file's bytes
  obligations_digest sha256(json.dumps(<spec yaml>["obligations"], sort_keys=True))
  entries            sorted by (obligation, path, name); helpers by
                     (path, item); fixtures by (obligation, path)

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
# a top-level item: optional attributes, then fn/struct/enum/impl/const/static/type/trait at column 0
ITEM = re.compile(r"^((?:#\[[^\n]*\]\n)*)(?:pub(?:\([^)]*\))? )?(fn|struct|enum|impl|const|static|type|trait)\b([^\n{;]*)", re.M)
DATA = re.compile(r'"(?:\{\}/)?([\w./-]+\.(?:tsv|csv|jsonl?|txt|ya?ml|toml))"')


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


def item_end(src, start):
    """End of a top-level item: its closing brace, or its `;` if it has none."""
    brace, semi = src.find("{", start), src.find(";", start)
    if semi != -1 and (brace == -1 or semi < brace):
        return semi + 1
    return fn_end(src, start)


def ws(text):
    return " ".join(text.split())


def items(src):
    """Top-level items as (label, names it defines, text). An impl is
    labelled by its header and defines its type's name, so a test using
    `Score` pulls in `struct Score` and every `impl ... Score`."""
    out = []
    for m in ITEM.finditer(src):
        kind, head = m.group(2), m.group(3).strip()
        if kind == "fn" and "#[test]" in m.group(1):
            continue
        text = src[m.start():item_end(src, m.start(2))]
        if kind == "impl":
            names = set(re.findall(r"\b([A-Z]\w*)\b", head.split(" for ")[-1]))
            label = "impl " + ws(head)
        else:
            name = re.match(r"(\w+)", head).group(1)
            names, label = {name}, f"{kind} {name}"
        out.append((label, names, text))
    return out


def used_helpers(src, texts):
    """Items transitively referenced by `texts` (the tagged tests)."""
    pool = items(src)
    seen, frontier = {}, list(texts)
    while frontier:
        t = frontier.pop()
        idents = set(re.findall(r"\b\w+\b", t))
        for label, names, text in pool:
            if label not in seen and names & idents:
                seen[label] = text
                frontier.append(text)
    return seen


def scan():
    """(tagged checks, helpers, data files named by either)."""
    checks, helpers, named = [], [], set()
    for f in sorted(os.listdir("tests")):
        if not f.endswith(".rs"):
            continue
        path = f"tests/{f}"
        src = open(path, encoding="utf-8").read()
        texts = []
        for m in TAG.finditer(src):
            text = src[m.start():fn_end(src, m.end())]
            texts.append(text)
            checks.append({"obligation": m.group(1), "path": path, "name": m.group(2), "lang": "rust",
                           "hash": sha(ws(text))})
        for label, text in used_helpers(src, texts).items():
            helpers.append({"path": path, "item": label, "hash": sha(ws(text))})
            texts.append(text)
        for t in texts:
            for lit in DATA.findall(t):
                for cand in (lit, f"tests/{lit}"):
                    if os.path.isfile(cand):
                        named.add((cand, path))
    checks.sort(key=lambda e: (e["obligation"], e["path"], e["name"]))
    helpers.sort(key=lambda e: (e["path"], e["item"]))
    return checks, helpers, named


def tagged_checks():
    return scan()[0]


def fixtures(obs):
    out = []
    for o in obs:
        for p in o.get("fixtures", []) or []:
            h = sha_file(p) if os.path.isfile(p) else None
            out.append({"obligation": o["id"], "path": p, "hash": h})
    return sorted(out, key=lambda e: (e["obligation"], e["path"]))


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


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
    checks, helpers, named = scan()
    locked = {key(e): e["hash"] for e in lock["entries"]}
    now = {key(e): e["hash"] for e in checks}
    hkey = lambda e: (e["path"], e["item"])
    lh = {hkey(e): e["hash"] for e in lock.get("helpers", [])}
    nh = {hkey(e): e["hash"] for e in helpers}
    for k in sorted(lh.keys() | nh.keys()):
        if k not in nh:
            problems.append(f"{k[0]}: locked helper `{k[1]}` is gone or no longer used by a tagged check")
        elif k not in lh:
            problems.append(f"{k[0]}: helper `{k[1]}` used by a tagged check is not in the lock")
        elif lh[k] != nh[k]:
            problems.append(f"{k[0]}: locked helper `{k[1]}` was modified")
    fkey = lambda e: (e["obligation"], e["path"])
    fx = fixtures(obs)
    lf = {fkey(e): e["hash"] for e in lock.get("fixtures", [])}
    nf = {fkey(e): e["hash"] for e in fx}
    for k in sorted(lf.keys() | nf.keys()):
        if k in nf and nf[k] is None:
            problems.append(f"{k[1]} ({k[0]}): declared fixture does not exist")
        elif k not in nf:
            problems.append(f"{k[1]} ({k[0]}): locked fixture is no longer declared")
        elif k not in lf:
            problems.append(f"{k[1]} ({k[0]}): fixture is not in the lock")
        elif lf[k] != nf[k]:
            problems.append(f"{k[1]} ({k[0]}): locked fixture was modified")
    declared = {e["path"] for e in fx}
    for f, by in sorted(named):
        if f not in declared:
            problems.append(f"{by}: a tagged check or its helper reads {f}, which no obligation declares under fixtures:")
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
    print(f"check-obligations: ok ({len(now)} checks, {len(nh)} helpers, {len(nf)} fixtures, "
          f"{len(ids)} obligations, approver {lock['approver']})")
    return 0


def build(commit, approver):
    obs = obligations()
    checks, helpers, _ = scan()
    fx = fixtures(obs)
    missing = [e["path"] for e in fx if e["hash"] is None]
    if missing:
        sys.exit(f"check-obligations: declared fixtures do not exist: {', '.join(missing)}")
    lock = {"version": 3, "approver": approver, "criteria_commit": commit,
            "obligations_digest": sha(json.dumps(obs, sort_keys=True)), "entries": checks,
            "helpers": helpers, "fixtures": fx}
    print(json.dumps(lock, indent=2))
    return 0


if __name__ == "__main__":
    args = sys.argv[1:] or ["verify"]
    if args[0] == "verify" and len(args) == 1:
        sys.exit(verify())
    if args[0] == "build" and len(args) == 3:
        sys.exit(build(args[1], args[2]))
    sys.exit(__doc__)
