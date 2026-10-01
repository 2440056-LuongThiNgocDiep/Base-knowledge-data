import ast, re
import argparse, collections, inspect, itertools, json, random, sys
from pathlib import Path

import generate
import program
import annotate_graphs

HERE = Path(__file__).parent
ENUM_LIMIT = 200_000
MC_DRAWS = 4_000
MIN_SUPPORT = 128


def graph_signature(spec, kernel_ids):
    nodes = spec["nodes"]
    by_id = {n["node_id"]: n for n in nodes}
    resolved = {}

    def sources(nid):
        if nid in resolved:
            return resolved[nid]
        out = []
        for name, v in sorted((by_id[nid].get("args") or {}).items()):
            for item in (v if isinstance(v, list) else [v]):
                if not isinstance(item, dict) or "ref" not in item:
                    continue
                parent = item["ref"]
                if by_id[parent]["atom_id"] in kernel_ids:
                    out.extend(sources(parent))
                else:
                    out.append((name, parent))
        resolved[nid] = out
        return out

    edges = []
    for nid, node in by_id.items():
        if node["atom_id"] in kernel_ids:
            continue
        for name, parent in sources(nid):
            edges.append((by_id[parent]["atom_id"], node["atom_id"], name))
    tail = program.returned(spec)
    while by_id[tail]["atom_id"] in kernel_ids:
        ups = [p for _, p in sources(tail)]
        if not ups:
            break
        tail = ups[0]
    labels = sorted(n["atom_id"] for n in nodes
                    if n["atom_id"] not in kernel_ids)
    return json.dumps({"atoms": labels, "edges": sorted(edges),
                       "returns": by_id[tail]["atom_id"]},
                      sort_keys=True, separators=(",", ":"))


def domains(t):
    out = []
    for k, r in (t.get("vars") or {}).items():
        if r["type"] == "choice":
            out.append((k, list(r["values"])))
        else:
            drop = {x for x in r.get("exclude", []) if not isinstance(x, str)}
            out.append((k, [v for v in range(r["min"], r["max"] + 1) if v not in drop]))
    return out


def support(t):
    doms = domains(t)
    cases = t.get("cases") or [None]
    raw = len(cases)
    for _, vs in doms:
        raw *= len(vs)
    if raw <= ENUM_LIMIT:
        seen = set()
        for case in cases:
            for combo in itertools.product(*[vs for _, vs in doms]):
                vals = dict(case) if case else {}
                ok = True
                for (k, _), v in zip(doms, combo):
                    rule = t["vars"][k]
                    bad = {vals[x] for x in rule.get("exclude", [])
                           if isinstance(x, str) and x in vals}
                    if v in bad:
                        ok = False
                        break
                    vals[k] = v
                if not ok:
                    continue
                try:
                    for name, expr in (t.get("derive") or {}).items():
                        vals[name] = generate.evaluate(expr, vals)
                    if not all(generate.evaluate(c, vals)
                               for c in (t.get("constraints") or [])):
                        continue
                    seen.add(t["template"].format(**generate.enrich(dict(vals))))
                except Exception:
                    continue
        return len(seen), True


    seen = set()
    for _ in range(MC_DRAWS):
        try:
            seen.add(t["template"].format(**generate.sample(t)))
        except Exception:
            pass
        if len(seen) >= MIN_SUPPORT:
            return len(seen), True
    return len(seen), False


FLOOR_COVERED = 0.70
FLOOR_REUSE = 2
CEILING_SHARE = 0.25


def spread_report(tem, comp):
    templated = {t["atom"] for t in tem}
    use = collections.Counter()
    for c in comp:
        use.update(set(c["atoms"]))
    used = set(use)

    covered = len(used & templated) / len(templated)
    thin = sorted(a for a, k in use.items() if k < FLOOR_REUSE)
    hot = [(a, k) for a, k in use.most_common() if k / len(comp) > CEILING_SHARE]
    missing = sorted(templated - used)

    print(f"composites {len(comp)}   atoms with a template {len(templated)}   "
          f"used by a composite {len(used & templated)}")
    print()
    ok = True
    def check(name, good, detail=""):
        nonlocal ok
        ok = ok and good
        print(f"  {'pass' if good else 'FAIL'}  {name}{('   ' + detail) if detail else ''}")

    check(f"coverage >= {FLOOR_COVERED:.0%}", covered >= FLOOR_COVERED, f"{covered:.0%}")
    check(f"every used atom in >= {FLOOR_REUSE} composites", not thin,
          f"{len(thin)} below: {', '.join(thin[:4])}{' ...' if len(thin) > 4 else ''}")
    check(f"no atom in > {CEILING_SHARE:.0%} of composites", not hot,
          "; ".join(f"{a} {k}/{len(comp)}" for a, k in hot[:3]))

    print(f"\n  atoms with a template but no composite: {len(missing)}")
    for a in missing[:12]:
        print(f"     {a}")
    if len(missing) > 12:
        print(f"     ... and {len(missing) - 12} more")
    return 0 if ok else 1


def canonical(v):
    if isinstance(v, bool) or isinstance(v, float):
        return False
    if isinstance(v, tuple):
        return generate.round_trips(v)
    if isinstance(v, (dict, list, set)):
        return False
    return True


def graph_edges(g):
    ids = [n["node_id"] for n in g["nodes"]]
    edges = set()
    for n in g["nodes"]:
        for spec in n["inputs"].values():
            if spec["source"] in ids:
                edges.add((spec["source"], n["node_id"]))
    return ids, edges


def check_graph(g):
    problems = []
    ids, edges = graph_edges(g)
    if len(set(ids)) != len(ids):
        problems.append("duplicate node_id")
    final = g["final"].split(".")[0]
    if final not in ids:
        problems.append(f"final {final!r} is not a node")
        return problems
    adj = collections.defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
    pos = {n: i for i, n in enumerate(ids)}
    if any(pos[a] >= pos[b] for a, b in edges):
        problems.append("cycle or forward reference")
    reaches = {final}
    for n in reversed(ids):
        if adj[n] & reaches:
            reaches.add(n)
    orphans = [n for n in ids if n not in reaches]
    if orphans:
        problems.append(f"nodes not on a path to final: {orphans}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=25, help="samples per template")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    atoms = generate.load(HERE / "atoms.jsonl")
    tem = generate.load(HERE / "templates.jsonl")
    comp = generate.load(HERE / "composite.jsonl")
    import kernel as _k
    atom_ids = {a["id"] for a in atoms} | set(_k.REGISTRY)
    fail = []

    def bad(kind, tid, msg):
        fail.append(f"[{kind}] {tid}: {msg}")

    declared = {a["id"] for a in atoms}
    for aid, kind in _k.KIND.items():
        if kind == "knowledge" and aid not in declared:
            bad("library", aid, "registered in code but absent from atoms.jsonl")
    for row in atoms:
        if row["id"] not in _k.REGISTRY:
            bad("library", row["id"], "declared but not registered in code")


    for t in tem + comp:
        for case in (t.get("cases") or []):
            ov = sorted(set(case) & set(t.get("vars") or {}))
            if ov:
                bad("cases", t["id"], f"also sampled in vars: {', '.join(ov)}")
                break


    for t in tem + comp:
        seen_vars = set(t.get("cases") and t["cases"][0] or {})
        for k, rule in (t.get("vars") or {}).items():
            for e in rule.get("exclude", []):
                if isinstance(e, str) and e not in seen_vars:
                    bad("order", t["id"],
                        f"{k} excludes {e!r}, which is not sampled before it")
            seen_vars.add(k)
        known = seen_vars | set(generate.SAFE)
        try:
            for name in generate.derive_order(t.get("derive") or {}):
                expr = t["derive"][name]
                for n in ast.walk(ast.parse(expr, mode="eval")):
                    if isinstance(n, ast.Name) and n.id not in known:
                        bad("order", t["id"],
                            f"derive {name} reads {n.id!r}, which is not defined")
                known.add(name)
        except ValueError as e:
            bad("order", t["id"], str(e))


    for kind, rows in (("atoms", atoms), ("atomic", tem), ("composite", comp)):
        seen = set()
        for r in rows:
            if not r.get("unit"):
                bad("unit", r["id"], "no unit field")
            if r["id"] in seen:
                bad("unit", r["id"], f"duplicate {kind} id")
            seen.add(r["id"])
    dup = ({t["id"] for t in tem} & {c["id"] for c in comp}) | \
          ({a["id"] for a in atoms} & ({t["id"] for t in tem} | {c["id"] for c in comp}))
    for d in sorted(dup):
        bad("unit", d, "id used by more than one record kind")
    units = {r.get("unit") for r in atoms + tem + comp if r.get("unit")}
    if len(units) > 1:
        bad("unit", "pack", f"one pack is one unit, found {sorted(units)}")


    specs = {g["id"]: g["nodes"] for g in generate.load(HERE / "graphs.jsonl")}
    for gid in sorted(set(specs) - {c["id"] for c in comp}):
        bad("graphs", gid, "graph spec has no composite in composite.jsonl")
    program_form = {gid for gid, nodes in specs.items() if any("args" in n for n in nodes)}
    kernel_only = {a for a, kind in _k.KIND.items() if kind == "kernel"}
    full = {g["id"]: g for g in generate.load(HERE / "graphs.jsonl")}
    shapes = {}
    for gid in sorted(program_form):
        try:
            sig = graph_signature(full[gid], kernel_only)
        except Exception as e:
            bad("graphs", gid, f"cannot key the graph: {type(e).__name__}: {e}")
            continue
        prev = shapes.setdefault(sig, gid)
        if prev != gid:
            bad("graphs", gid,
                f"same atom-labelled dependency graph as {prev}; two composites "
                f"may not differ only in wording or sampled values")
    fresh, ungraphed = annotate_graphs.expand(comp, specs)
    for cid in ungraphed:
        bad("graphs", cid, "composite has no graph spec in graphs.jsonl")
    for old, new in zip(comp, fresh):
        if old["id"] in program_form:
            continue
        if old.get("graph") != new.get("graph") or old.get("atoms") != new.get("atoms"):
            bad("graphs", old["id"],
                "stale against graphs.jsonl -- run: python3 annotate_graphs.py")

    for t in tem:
        if t["atom"] not in atom_ids:
            bad("references", t["id"], f"unknown atom {t['atom']}")
        if not t.get("args"):
            bad("references", t["id"], "atomic template has no args")
    for c in comp:
        for a in c["atoms"]:
            if a not in atom_ids:
                bad("references", c["id"], f"unknown atom {a}")
        if not c.get("example"):
            bad("references", c["id"], "composite has no worked example")

    for t in tem + comp:
        seen = {}
        for _ in range(args.draws):
            try:
                q, a = generate.make(t)
            except Exception as e:
                bad("generation", t["id"], f"{type(e).__name__}: {e}")
                break
            if not canonical(a):
                bad("answer", t["id"],
                    f"answer is {type(a).__name__} {a!r}; an answer must be one "
                    f"canonical value (no dict, list, set, bool or float)")
                break
            key = json.dumps(a, sort_keys=True, default=str)
            if seen.setdefault(q, key) != key:
                bad("determinism", t["id"], "same question, two different answers")
                break


    origin = {}
    for t in tem + comp:
        for _ in range(args.draws):
            try:
                q = t["template"].format(**generate.sample(t))
            except Exception:
                break
            prev = origin.setdefault(q, t["id"])
            if prev != t["id"]:
                bad("duplicate", t["id"], f"renders a question {prev} also renders: {q!r}")
                break

    for c in comp:
        if c["id"] in program_form:
            continue
        g = c.get("graph")
        if not g:
            bad("graphs", c["id"], "no graph annotation")
            continue
        for p in check_graph(g):
            bad("graphs", c["id"], p)
        for n in g["nodes"]:
            if not n.get("atom_id"):
                bad("graphs", c["id"],
                    f"node {n['node_id']} has no atom_id; name the atom it applies, "
                    f"or \"arithmetic\" if the step is only arithmetic")
        used = {n["atom_id"] for n in g["nodes"]
                if n.get("atom_id") and n["atom_id"] != "arithmetic"}
        if used - set(c["atoms"]):
            bad("graphs", c["id"], f"graph uses atoms not declared: {used - set(c['atoms'])}")
        if set(c["atoms"]) - used:
            bad("graphs", c["id"], f"declared atoms unused by graph: {set(c['atoms']) - used}")
        ex = c.get("example") or {}
        for _ in range(args.draws):
            try:
                vals = generate.sample(c)
                got, _ = generate.run_graph(c, vals)
                gold = got
            except Exception as e:
                bad("crosscheck", c["id"], f"{type(e).__name__}: {e}")
                break
            _, outs = generate.run_graph(c, vals)
            final = c["graph"]["final"].split(".")[0]
            for nid, o in outs.items():


                if not canonical(o):
                    bad("graphs", c["id"],
                        f"node {nid} emits {type(o).__name__} {o!r}, which is not one "
                        f"canonical exact value")
                    break
            question = c["template"].format(**vals)


            shown_vars = set(re.findall(r"\{(\w+)\}", c["template"]))
            for _ in range(len(c.get("derive") or {})):
                for nm, dx in (c.get("derive") or {}).items():
                    if nm in shown_vars:
                        shown_vars |= {x.id for x in ast.walk(ast.parse(dx, mode="eval"))
                                       if isinstance(x, ast.Name)}
            for node in c["graph"]["nodes"]:
                for name, src in node["inputs"].items():
                    if src["source"] != "question":
                        continue
                    val = str(vals.get(src["field"], ""))
                    shown = question.replace("−", "-").replace("+ ", "+")
                    shown = re.sub(r"- (?=\d)", "-", shown)
                    if val and val not in shown and src["field"] not in shown_vars:
                        bad("graphs", c["id"],
                            f"node {node['node_id']} binds {src['field']}={val!r}, "
                            f"which the question never shows")

    covered = {t["atom"] for t in tem}
    for c in comp:
        for a in set(c["atoms"]) - covered:
            bad("coverage", c["id"], f"atom {a} has no working atomic template")

    print(f"atoms {len(atoms)}   atomic {len(tem)}   composite {len(comp)}")
    if fail:
        print(f"\nFAILED ({len(fail)}):")
        for f in fail:
            print("  " + f)
    else:
        print("all checks passed")


    small, stale = [], []
    for t in tem + comp:
        n, exact = support(t)
        if n < MIN_SUPPORT and not t.get("finite_support"):
            small.append((t["id"], n, "exact" if exact else "approx"))
        elif n >= MIN_SUPPORT and t.get("finite_support"):
            stale.append((t["id"], n))
    declared = sum(1 for t in tem + comp if t.get("finite_support"))
    print("\natom spread across composites (section 1 targets, not a correctness gate):")
    spread_report(tem, comp)

    print(f"\nfinite_support declared: {declared}   "
          f"(excluded from the 8.4 target; lead must approve each)")
    if small:
        print(f"\nFAILED: {len(small)} templates make under {MIN_SUPPORT} distinct "
              f"questions and are not marked finite_support")
        for i, n, e in sorted(small, key=lambda x: x[1]):
            print(f"   {i:<46} {n:>6}  ({e})")
    if stale:
        print(f"\nFAILED: {len(stale)} marked finite_support but reach "
              f"{MIN_SUPPORT}; drop the flag")
        for i, n in stale:
            print(f"   {i:<46} {n:>6}")
    return 1 if (fail or small or stale) else 0


if __name__ == "__main__":
    sys.exit(main())
