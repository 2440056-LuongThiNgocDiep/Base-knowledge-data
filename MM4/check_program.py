import argparse, collections, inspect, json, random, re, sys
from pathlib import Path

import kernel, atoms, generate, program

HERE = Path(__file__).parent
DRAWS = 200


ASK_FORM = re.compile(r"in the form\s+[^.?]*", re.I)
TUPLEY = re.compile(r"\(\s*-?[\d/]+\s*(,\s*-?[\d/]+\s*)+\)")


def knowledge_depth(nodes, final=None):
    deps = program.dependencies(nodes)
    kind = {n["node_id"]: kernel.KIND.get(n["atom_id"]) for n in nodes}
    memo = {}

    def d(nid):
        if nid not in memo:
            step = 1 if kind.get(nid) == "knowledge" else 0
            memo[nid] = step + max([d(p) for p in deps[nid]], default=0)
        return memo[nid]

    return d(final or nodes[-1]["node_id"])


def check(comp, spec):
    nodes = spec["nodes"]
    final = program.returned(spec) or nodes[-1]["node_id"]
    out = []

    def rule(name, ok, detail=""):
        out.append((name, ok, detail))

    qvars = set(comp.get("vars") or {}) | set(comp.get("derive") or {})
    for case in (comp.get("cases") or []):
        qvars |= set(case)


    ids = [n["node_id"] for n in nodes]
    rule("node ids unique", len(set(ids)) == len(ids))
    rule("returned node exists", final in ids, final)

    bad_atom = [n["node_id"] for n in nodes if n["atom_id"] not in kernel.REGISTRY]
    rule("every atom is registered", not bad_atom, ", ".join(bad_atom))

    bad_arg, bad_ref, forward = [], [], []
    seen = []
    for n in nodes:
        fn = kernel.REGISTRY.get(n["atom_id"])
        args = n.get("args") or {}
        if fn is not None:
            want = set(inspect.signature(fn).parameters)
            if set(args) != want:
                bad_arg.append(f"{n['node_id']} wants {sorted(want)}, got {sorted(args)}")
        for v in args.values():
            for ref in (v if isinstance(v, (list, tuple)) else [v]):
                if not isinstance(ref, str):
                    continue
                if ref.startswith("question."):
                    if ref[len("question."):] not in qvars:
                        bad_ref.append(f"{n['node_id']}: {ref}")
                elif ref in ids:
                    if ref not in seen:
                        forward.append(f"{n['node_id']} reads {ref}")
                elif ref not in program.CONSTANTS:
                    bad_ref.append(f"{n['node_id']}: {ref}")
        seen.append(n["node_id"])
    rule("arguments match the atom", not bad_arg, "; ".join(bad_arg))
    rule("references resolve", not bad_ref, "; ".join(bad_ref))
    rule("acyclic (no forward reads)", not forward, "; ".join(forward))

    anc = program.ancestors(nodes, final) | {final}
    orphan = [i for i in ids if i not in anc]
    rule("every node reaches the answer", not orphan, ", ".join(orphan))


    kind = {n["node_id"]: kernel.KIND.get(n["atom_id"]) for n in nodes}
    atom_of = {n["node_id"]: n["atom_id"] for n in nodes}
    deps = program.dependencies(nodes)
    best = set()

    def chain(nid, acc):
        nonlocal best
        acc = acc | ({atom_of[nid]} if kind.get(nid) == "knowledge" else set())
        if len(acc) > len(best):
            best = acc
        for p in deps[nid]:
            chain(p, acc)

    chain(final, set())
    rule("two knowledge atoms on one path", len(best) >= 2,
         f"{len(best)} chained: {sorted(a.split('.')[-1] for a in best)}")


    answers, novar, crash = collections.Counter(), collections.Counter(), ""
    shown_answers = collections.Counter()
    per_node = collections.defaultdict(set)
    for _ in range(DRAWS):
        try:
            v = generate.sample(comp)
            ans, vals = program.run(nodes, v, final)
        except Exception as e:
            crash = f"{type(e).__name__}: {e}"
            break
        answers[str(ans)] += 1
        shown_answers[str(kernel.display(ans, comp.get("display")))] += 1
        for k, x in vals.items():
            per_node[k].add(str(x))
        for n in nodes:
            src = [vals.get(r) for r in (n.get("args") or {}).values()
                   if isinstance(r, str) and r in vals]
            if src and str(vals[n["node_id"]]) in {str(s) for s in src}:
                novar[n["node_id"]] += 1
    rule("executes on every draw", not crash, crash)


    ex = comp.get("example")
    if ex and not crash:
        mismatch = ""
        try:
            v = dict(ex["vars"])
            for name in generate.derive_order(comp.get("derive") or {}):
                v[name] = generate.evaluate(comp["derive"][name], v)
            got, _ = program.run(nodes, v, final)
            shown = str(kernel.display(got, comp.get("display"))).strip()
            if shown != str(ex["answer"]).strip():
                mismatch = f"program gave {shown!r}, worked example says {ex['answer']!r}"
        except Exception as e:
            mismatch = f"{type(e).__name__}: {e}"
        rule("reproduces the worked example", not mismatch, mismatch)

    asked = ASK_FORM.search(comp.get("template") or "")
    if asked and shown_answers:
        odd = [a for a in shown_answers if TUPLEY.fullmatch(str(a).strip())]
        rule("answer matches the form the question asks for", not odd,
             f"question asks for {asked.group(0).strip()!r} but the answer renders as "
             f"{odd[0]!r}; set a \"display\" such as \"polynomial\"" if odd else "")
    if answers:
        top, k = answers.most_common(1)[0]
        rule("answers vary", k / DRAWS <= 0.6, f"{top!r} on {k*100//DRAWS}%, {len(answers)} distinct")
        const = [i for i in ids if len(per_node[i]) == 1
                 and kernel.KIND.get(atom_of.get(i)) == "knowledge"]
        rule("no node is constant", not const, ", ".join(const))
        noop = [f"{i} on {c*100//DRAWS}%" for i, c in novar.items() if c > DRAWS * 0.5]
        rule("no step is a no-op", not noop, ", ".join(noop))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("id", nargs="?")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    comps = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "composite.jsonl")}
    specs = [json.loads(l) for l in open(HERE / "graphs.jsonl")]
    if a.id:
        specs = [s for s in specs if s["id"] == a.id] or sys.exit(f"no composite {a.id!r}")

    legacy = [s["id"] for s in specs if not any("args" in n for n in s["nodes"])]
    specs = [s for s in specs if any("args" in n for n in s["nodes"])]
    worst = 0
    for s in specs:
        res = check(comps[s["id"]], s)
        bad = [r for r in res if not r[1]]
        worst = max(worst, len(bad))
        print(f"\n{s['id']}   {'OK' if not bad else str(len(bad)) + ' FAILED'}"
              f"   knowledge depth {knowledge_depth(s["nodes"], program.returned(s))}")
        for name, ok, detail in res:
            print(f"  {'pass' if ok else 'FAIL'}  {name:<32} {detail}")
    if legacy:
        print(f"\n{len(legacy)} composites still in the legacy expr form, not checked:")
        print("   " + ", ".join(legacy[:8]) + (" ..." if len(legacy) > 8 else ""))
    return 1 if worst else 0


if __name__ == "__main__":
    raise SystemExit(main())
