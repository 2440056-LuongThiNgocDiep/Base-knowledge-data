import sys
import ast
import argparse, collections, inspect, json, math, random
from fractions import Fraction
from pathlib import Path
import sympy

import kernel, atoms

def term(coef, suffix=""):
    if coef == 0:
        return ""
    body = suffix if abs(coef) == 1 and suffix else f"{abs(coef)}{suffix}"
    return f" {'−' if coef < 0 else '+'} {body}"


def coef(a, body=""):
    if a == 1:
        return body
    if a == -1:
        return "−" + body
    return f"{a}{body}"


def linfac(a, v="x"):
    if a == 0:
        return v
    return f"({v} − {a})" if a > 0 else f"({v} + {-a})"


def shift(v, pos, neg):
    return f"{abs(v)} unit{'' if abs(v) == 1 else 's'} {pos if v > 0 else neg}"


SAFE = {
    "abs": abs, "min": min, "max": max, "int": int, "str": str, "len": len,
    "sum": sum, "sorted": sorted, "set": set, "range": range, "bool": bool,
    "all": all, "any": any,
    "comb": math.comb, "gcd": math.gcd,
    "Fraction": Fraction, "sqrt": sympy.sqrt,
    "pi": sympy.pi, "ceiling": sympy.ceiling,
    "coef": coef,
    "poly": kernel.poly,
    "bracket": kernel.bracket,
    "quad": kernel.quad,
    "cubic": kernel.cubic,
    "lineq": kernel.lineq,
    "pf": kernel.pf,
    "term": term, "linfac": linfac, "shift": shift,
}

MAX_TRIES = 500


def canon(v):
    if isinstance(v, tuple):
        if len(v) == 1:
            return f"({canon(v[0])},)"
        return "(" + ", ".join(canon(x) for x in v) + ")"
    if isinstance(v, Fraction):
        return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"
    if isinstance(v, float):
        raise TypeError("floats are not exact; use Fraction")
    if isinstance(v, bool):
        raise TypeError("a bool cannot round-trip; emit the value it decides")
    return str(v)


def parse_struct(text):
    def walk(n):
        if isinstance(n, ast.Tuple):
            return tuple(walk(e) for e in n.elts)
        if isinstance(n, ast.Constant) and type(n.value) is int:
            return n.value
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -walk(n.operand)
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div):
            return Fraction(walk(n.left), walk(n.right))
        raise ValueError(f"not a structured value: {ast.dump(n)[:40]}")
    return walk(ast.parse(text.strip(), mode="eval").body)


def round_trips(v):
    try:
        return parse_struct(canon(v)) == v
    except Exception:
        return False


def run_graph(t, vals):
    g = t.get("graph")
    if not g:
        return None, None
    env, outs = dict(vals), {}
    for node in g["nodes"]:
        v = evaluate(node["expr"], env)


        outs[node["node_id"]] = canon(v) if isinstance(v, tuple) else v
        env[node["node_id"]] = v
    return outs[g["final"].split(".")[0]], outs


def _load_graphs():
    import json as _j
    from pathlib import Path as _P
    f = _P(__file__).parent / "graphs.jsonl"
    if not f.exists():
        return {}
    out = {}
    for line in open(f, encoding="utf-8"):
        if line.strip():
            g = _j.loads(line)
            if "return" in g:
                out[g["id"]] = g
    return out


GRAPHS = _load_graphs()


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def evaluate(expr, vals):
    return eval(expr, {"__builtins__": {}}, {**SAFE, **vals})


def sample_var(rule, ctx):
    if rule["type"] == "choice":
        return random.choice(rule["values"])
    if rule["type"] == "int":
        drop = {ctx[x] if isinstance(x, str) else x for x in rule.get("exclude", [])
                if not isinstance(x, str) or x in ctx}
        vals = [v for v in range(rule["min"], rule["max"] + 1) if v not in drop]
        if not vals:
            raise ValueError("empty domain")
        return random.choice(vals)
    raise ValueError(rule["type"])


def derive_order(derive):
    deps = {}
    for name, expr in derive.items():
        reads = {n.id for n in ast.walk(ast.parse(expr, mode="eval"))
                 if isinstance(n, ast.Name)}
        deps[name] = reads & set(derive) - {name}
    out, seen = [], set()

    def visit(n, stack):
        if n in seen:
            return
        if n in stack:
            raise ValueError(f"derive cycle: {' -> '.join(list(stack) + [n])}")
        for d in sorted(deps[n]):
            visit(d, stack | {n})
        seen.add(n)
        out.append(n)

    for n in derive:
        visit(n, set())
    return out


def sample(t):
    order = derive_order(t.get("derive") or {})
    for _ in range(MAX_TRIES):
        vals = {}
        if t.get("cases"):
            vals.update(random.choice(t["cases"]))
        for k, r in (t.get("vars") or {}).items():
            vals[k] = sample_var(r, vals)
        for name in order:
            vals[name] = evaluate(t["derive"][name], vals)
        if all(evaluate(c, vals) for c in (t.get("constraints") or [])):
            return enrich(vals)
    raise RuntimeError(f"{t['id']}: no assignment satisfied constraints in {MAX_TRIES} tries")


def pi_frac(n, d=1):
    from fractions import Fraction
    f = Fraction(n, d)
    if f == 0:
        return "0"
    if f == 1:
        return "pi"
    if f == -1:
        return "-pi"
    if f.denominator == 1:
        return f"{f.numerator}*pi"
    if f.numerator == 1:
        return f"pi/{f.denominator}"
    if f.numerator == -1:
        return f"-pi/{f.denominator}"
    return f"{f.numerator}*pi/{f.denominator}"


def lin_x(b, c):
    if b == 1:
        s = "x"
    elif b == -1:
        s = "-x"
    else:
        s = f"{b}x"
    if c > 0:
        return f"{s}+{c}"
    if c < 0:
        return f"{s}{c}"
    return s


def shift_x(c):
    return lin_x(1, c)


def term_x(coef, c):
    if coef == "1":
        s = "x"
    elif coef == "-1":
        s = "-x"
    else:
        s = f"{coef}*x"
    if c > 0:
        return f"{s}+{c}"
    if c < 0:
        return f"{s}{c}"
    return s


def sin_model(A, arg, d0):
    if A == 1:
        s = f"sin({arg})"
    elif A == -1:
        s = f"-sin({arg})"
    else:
        s = f"{A}sin({arg})"
    if d0 > 0:
        return f"{s}+{d0}"
    if d0 < 0:
        return f"{s}{d0}"
    return s


def enrich(vals):
    if "b" in vals and "c" in vals:
        vals["lin"] = lin_x(int(vals["b"]), int(vals["c"]))
    if "c" in vals:
        vals["xshift"] = shift_x(int(vals["c"]))
    if {"bn", "bd", "c"}.issubset(vals):
        vals["omega"] = pi_frac(int(vals["bn"]), int(vals["bd"]))
        vals["arg"] = term_x(vals["omega"], int(vals["c"]))
    if {"A", "arg", "d0"}.issubset(vals):
        vals["model"] = sin_model(int(vals["A"]), vals["arg"], int(vals["d0"]))
    return vals


def answer_of(t, vals):
    import program, kernel
    spec = GRAPHS.get(t["id"])
    if spec is not None:
        got, _ = program.run(spec["nodes"], vals,
                             (spec.get("return") or {}).get("ref"))
        return kernel.display(got, t.get("display"))
    node = [{"node_id": "n1", "atom_id": t["atom"], "args": t["args"]}]
    got, _ = program.run(node, vals, "n1")
    return kernel.display(got, t.get("display"))


def make(t):
    vals = sample(t)
    return t["template"].format(**vals), answer_of(t, vals)


def instances(t, want, tries_per_hit=40):
    seen, rows = set(), []
    for _ in range(want * tries_per_hit):
        if len(rows) >= want:
            break
        vals = sample(t)
        q = t["template"].format(**vals)
        if q in seen:
            continue
        seen.add(q)
        answer = answer_of(t, vals)
        _, node_outputs = run_graph(t, vals)
        rows.append((q, answer, node_outputs, vals))
    return rows


ATOMIC_PARTITIONS = [("atomic_train", 64), ("atomic_dev", 32), ("atomic_test", 32)]
COMPOSITE_PARTITIONS = {
    "train": [("composite_train_prompts", 64), ("composite_train_unseen_value", 64)],
    "dev": [("composite_dev", 128)],
    "structural_test": [("composite_structural_test", 128)],
    "depth_stress": [("composite_depth_stress", 128)],
}


def allocate(pool, plan):
    want = [w for _, w in plan]
    total = sum(want)
    if pool >= total:
        return want
    exact = [pool * w / total for w in want]
    got = [int(e) for e in exact]
    for i in sorted(range(len(want)), key=lambda i: exact[i] - got[i], reverse=True):
        if sum(got) >= pool:
            break
        got[i] += 1
    return got


def write_partitioned(rows, t, plan, pub, prv, counts):
    sizes = allocate(len(rows), plan)
    i = 0
    for (name, want), size in zip(plan, sizes):
        take = rows[i:i + size]
        i += size
        for j, (q, answer, node_outputs, vals) in enumerate(take, 1):
            iid = f"{t['id']}__{name}__{j:04d}"
            public = {"instance_id": iid, "template_id": t["id"],
                      "family_id": t.get("family_id"), "partition": name,
                      "question": q}
            if t.get("graph"):
                public["step_ids"] = [x["node_id"] for x in t["graph"]["nodes"]]
            pub[name].write(json.dumps(public, separators=(",", ":"),
                                       ensure_ascii=False) + "\n")


            prv[name].write(json.dumps({"instance_id": iid, "answer": answer,
                                        "node_outputs": node_outputs,
                                        "question_vars": vals},
                                       separators=(",", ":"), ensure_ascii=False,
                                       default=str) + "\n")
            counts[name] += 1
        if len(take) < want:
            counts.setdefault("_short", [])
            counts["_short"].append((t["id"], name, len(take), want))


def all_templates(d):
    rows = []
    for name in ("templates.jsonl", "composite.jsonl"):
        f = d / name
        if f.exists():
            rows += load(f)
    return rows


def main():
    p = argparse.ArgumentParser(
        description="Generate sample questions from the templates.")
    p.add_argument("--id", help="only this template id")
    p.add_argument("-n", "--num", type=int, default=3,
                   help="samples per template (default 3)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--data", default=str(Path(__file__).parent))
    p.add_argument("--out", help="write JSONL here instead of printing")
    p.add_argument("--steps", action="store_true",
                   help="also show each graph step's output")
    args = p.parse_args()

    random.seed(args.seed)
    d = Path(args.data)
    templates = all_templates(d)
    if args.id:
        templates = [t for t in templates if t["id"] == args.id]
        if not templates:
            raise SystemExit(f"no template with id {args.id!r}")

    rows, failed = [], []
    for t in templates:
        try:
            got = instances(t, args.num)
        except Exception as e:
            failed.append((t["id"], f"{type(e).__name__}: {e}"))
            continue
        if len(got) < args.num:
            failed.append((t["id"], f"only {len(got)}/{args.num} distinct"))
        for q, answer, node_outputs, vals in got:
            rows.append({"template_id": t["id"], "question": q,
                         "answer": answer, "node_outputs": node_outputs,
                         "question_vars": {k: str(v) for k, v in vals.items()}})

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        print(f"{len(rows)} samples from {len(templates)} templates -> {args.out}")
    else:
        current = None
        for r in rows:
            if r["template_id"] != current:
                current = r["template_id"]
                print(f"\n=== {current}")
            print(f"  Q: {r['question']}")
            print(f"  A: {r['answer']}")
            if args.steps and r["node_outputs"]:
                for nid, v in r["node_outputs"].items():
                    print(f"       {nid} = {v}")

    if failed:
        print(f"\n{len(failed)} template(s) with problems:")
        for tid, why in failed[:20]:
            print(f"   {tid:<46} {why}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
