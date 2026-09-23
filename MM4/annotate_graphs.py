import ast, json
from pathlib import Path

HERE = Path(__file__).parent
ORDER = ["id", "unit", "atoms", "template", "cases", "vars", "derive", "constraints",
         "args", "display", "example", "graph"]


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def wiring(expr, prior, question_vars):
    names = {n.id for n in ast.walk(ast.parse(expr, mode="eval"))
             if isinstance(n, ast.Name)}
    ins = {}
    for nm in sorted(names):
        if nm in prior:
            ins[nm] = {"source": nm, "field": "value", "type": "expression"}
        elif nm in question_vars:
            ins[nm] = {"source": "question", "field": nm, "type": "expression"}
    return ins


def expand(composites, graphs):
    out, missing = [], []
    for c in composites:
        c = dict(c)
        spec = graphs.get(c["id"])
        if spec is None:
            missing.append(c["id"])
            out.append(c)
            continue
        if any("args" in n for n in spec):
            live = []
            for n in spec:
                a = n["atom_id"]
                if a and not a.startswith("kernel.") and a not in live:
                    live.append(a)
            c["atoms"] = live
            c.pop("graph", None)
            out.append({k: c[k] for k in ORDER if k in c})
            continue
        qvars = set(c.get("vars") or {}) | set(c.get("derive") or {})
        for case in (c.get("cases") or []):
            qvars |= set(case)
        nodes, prior = [], set()
        for n in spec:
            nid, expr = n["node_id"], n["expr"]
            nodes.append({
                "node_id": nid,
                "atom_id": n.get("atom_id"),
                "inputs": wiring(expr, prior, qvars),
                "expr": expr,
                "outputs": {"value": {"type": "expression"}},
                "executor": f"expr:{c['id']}.{nid}",
                "verifier": f"exact:{c['id']}.{nid}",
            })
            prior.add(nid)
        c["graph"] = {"nodes": nodes, "final": f"{spec[-1]['node_id']}.value"}
        seen = []
        for nd in nodes:
            if nd["atom_id"] and nd["atom_id"] != "arithmetic" and nd["atom_id"] not in seen:
                seen.append(nd["atom_id"])
        c["atoms"] = seen
        out.append({k: c[k] for k in ORDER if k in c})
    return out, missing


def build(composite_path=None, graphs_path=None):
    composite_path = composite_path or HERE / "composite.jsonl"
    graphs_path = graphs_path or HERE / "graphs.jsonl"
    graphs = {g["id"]: g["nodes"] for g in load(graphs_path)}
    rows, missing = expand(load(composite_path), graphs)
    open(composite_path, "w", encoding="utf-8").writelines(
        json.dumps(r, separators=(",", ":"), ensure_ascii=False) + "\n" for r in rows)
    print(f"annotated {len(rows) - len(missing)} composites; "
          f"unannotated: {len(missing)} {missing}")


if __name__ == "__main__":
    build()
