import kernel, atoms

CONSTANTS = {"pi", "e"}


class ProgramError(ValueError):
    pass


def returned(spec):
    r = spec.get("return")
    if isinstance(r, dict) and "ref" in r:
        return r["ref"]
    if isinstance(r, str):
        return r
    return spec["nodes"][-1]["node_id"]


def resolve(ref, qvars, seen):
    if isinstance(ref, (list, tuple)):
        return tuple(resolve(x, qvars, seen) for x in ref)
    if not isinstance(ref, dict) or len(ref) != 1:
        raise ProgramError(f"argument {ref!r} must be exactly one of "
                           f"{{question}}, {{ref}}, {{literal}}")
    tag, val = next(iter(ref.items()))
    if tag == "question":
        if val not in qvars:
            raise ProgramError(f"question field {val!r} does not exist")
        return qvars[val]
    if tag == "ref":
        if val not in seen:
            raise ProgramError(f"node {val!r} is not an earlier node")
        return seen[val]
    if tag == "literal":
        return val
    raise ProgramError(f"unknown argument tag {tag!r}")


def run(nodes, qvars, final=None):
    seen = {}
    for node in nodes:
        nid, aid = node["node_id"], node["atom_id"]
        if nid in seen:
            raise ProgramError(f"duplicate node {nid}")
        fn = kernel.REGISTRY.get(aid)
        if fn is None:
            raise ProgramError(f"unknown atom {aid!r}")
        args = {k: resolve(v, qvars, seen) for k, v in (node.get("args") or {}).items()}
        try:
            seen[nid] = fn(**args)
        except TypeError as e:
            raise ProgramError(f"{nid}: {aid} called with {sorted(args)}: {e}")
    out = final or nodes[-1]["node_id"]
    if out not in seen:
        raise ProgramError(f"returned node {out} was never computed")
    return seen[out], seen


def dependencies(nodes):
    ids = [n["node_id"] for n in nodes]

    def refs(v):
        if isinstance(v, (list, tuple)):
            return {r for x in v for r in refs(x)}
        if isinstance(v, dict) and "ref" in v:
            return {v["ref"]}
        return set()

    deps = {}
    for n in nodes:
        seen = refs(list((n.get("args") or {}).values()))
        deps[n["node_id"]] = {r for r in seen if r in ids}
    return deps


def ancestors(nodes, target):
    deps = dependencies(nodes)
    out, stack = set(), [target]
    while stack:
        cur = stack.pop()
        for p in deps.get(cur, ()):
            if p not in out:
                out.add(p); stack.append(p)
    return out


def depth(nodes, final=None):
    deps = dependencies(nodes)
    memo = {}

    def d(nid):
        if nid not in memo:
            memo[nid] = 1 + max([d(p) for p in deps[nid]], default=0)
        return memo[nid]

    return d(final or nodes[-1]["node_id"])


def interfaces(nodes):
    by_id = {n["node_id"]: n["atom_id"] for n in nodes}
    deps = dependencies(nodes)
    return {(by_id[p], n["atom_id"]) for n in nodes for p in deps[n["node_id"]]}
