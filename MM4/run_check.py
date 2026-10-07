#!/usr/bin/env python3
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REQUIRED_FILES = [
    "atoms.jsonl",
    "templates.jsonl",
    "composite.jsonl",
    "graphs.jsonl",
    "coverage.tsv",
    "dataset.jsonl",
]
EXPECTED_ATOMIC = 40
EXPECTED_COMPOSITE = 34
EXPECTED_TEMPLATES = EXPECTED_ATOMIC + EXPECTED_COMPOSITE
SAMPLES_PER_TEMPLATE = 128
EXPECTED_DATASET = EXPECTED_TEMPLATES * SAMPLES_PER_TEMPLATE


def load_jsonl(path):
    rows = []
    errors = []
    if not path.exists():
        return rows, [f"missing file: {path.name}"]
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                errors.append(f"{path.name}:{lineno}: invalid JSON: {e}")
    return rows, errors


def check_required_files(errors):
    for name in REQUIRED_FILES:
        if not (ROOT / name).exists():
            errors.append(f"missing required file: {name}")


def check_jsonl_ids(name, rows, errors):
    seen = set()
    for i, row in enumerate(rows, 1):
        ident = row.get("id")
        if not ident:
            errors.append(f"{name}:{i}: missing required field 'id'")
            continue
        if ident in seen:
            errors.append(f"{name}:{i}: duplicate id '{ident}'")
        seen.add(ident)
    return seen


def check_atoms(errors):
    rows, errs = load_jsonl(ROOT / "atoms.jsonl")
    errors.extend(errs)
    ids = check_jsonl_ids("atoms.jsonl", rows, errors)
    return rows, ids


def check_templates(errors, atom_ids):
    rows, errs = load_jsonl(ROOT / "templates.jsonl")
    errors.extend(errs)
    ids = check_jsonl_ids("templates.jsonl", rows, errors)
    for i, row in enumerate(rows, 1):
        atoms = row.get("atoms", [])
        if not isinstance(atoms, list):
            errors.append(f"templates.jsonl:{i}: 'atoms' must be a list")
            continue
        for atom in atoms:
            if atom not in atom_ids:
                errors.append(f"templates.jsonl:{i}: unknown atom '{atom}'")
    return rows, ids


def check_composite(errors, atom_ids):
    rows, errs = load_jsonl(ROOT / "composite.jsonl")
    errors.extend(errs)
    ids = check_jsonl_ids("composite.jsonl", rows, errors)
    for i, row in enumerate(rows, 1):
        atoms = row.get("atoms", [])
        if not isinstance(atoms, list):
            errors.append(f"composite.jsonl:{i}: 'atoms' must be a list")
            continue
        if len(atoms) < 2:
            errors.append(f"composite.jsonl:{i}: composite must contain at least 2 atoms")
        for atom in atoms:
            if atom not in atom_ids:
                errors.append(f"composite.jsonl:{i}: unknown atom '{atom}'")
    return rows, ids


def check_graphs(errors, template_ids, atom_ids):
    rows, errs = load_jsonl(ROOT / "graphs.jsonl")
    errors.extend(errs)
    graph_ids = check_jsonl_ids("graphs.jsonl", rows, errors)

    for i, row in enumerate(rows, 1):
        tid = row.get("template_id") or row.get("id")
        if tid not in template_ids:
            errors.append(f"graphs.jsonl:{i}: unknown template id '{tid}'")

        nodes = row.get("nodes", [])
        edges = row.get("edges", [])
        if not isinstance(nodes, list):
            errors.append(f"graphs.jsonl:{i}: 'nodes' must be a list")
            continue
        if not isinstance(edges, list):
            errors.append(f"graphs.jsonl:{i}: 'edges' must be a list")
            continue

        node_ids = set()
        for node in nodes:
            if isinstance(node, dict):
                nid = node.get("id") or node.get("atom") or node.get("name")
                if nid:
                    node_ids.add(nid)
                    if nid in atom_ids:
                        continue
            elif isinstance(node, str):
                node_ids.add(node)

        # Check that graph edges reference existing node ids where possible.
        for eidx, edge in enumerate(edges, 1):
            if isinstance(edge, dict):
                src = edge.get("source") or edge.get("from")
                dst = edge.get("target") or edge.get("to")
            elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                src, dst = edge[0], edge[1]
            else:
                continue
            if src is not None and node_ids and src not in node_ids:
                errors.append(f"graphs.jsonl:{i}: edge {eidx} references unknown source '{src}'")
            if dst is not None and node_ids and dst not in node_ids:
                errors.append(f"graphs.jsonl:{i}: edge {eidx} references unknown target '{dst}'")

    return rows, graph_ids


def check_coverage(errors, template_ids):
    path = ROOT / "coverage.tsv"

    if not path.exists():
        errors.append("coverage.tsv: file not found")
        return

    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            lines = [line.rstrip("\r\n") for line in f if line.strip()]
    except Exception as e:
        errors.append(f"coverage.tsv: cannot read file: {e}")
        return

    if not lines:
        errors.append("coverage.tsv: empty file")
        return

    # Detect delimiter
    header = lines[0]
    delimiter = "\t" if "\t" in header else ","

    fields = [x.strip().lower() for x in header.split(delimiter)]

    # Accept id / template_id / code
    id_candidates = ["id", "template_id", "code"]

    id_column = None
    for candidate in id_candidates:
        if candidate in fields:
            id_column = fields.index(candidate)
            break

    if id_column is None:
        errors.append(
            "coverage.tsv: missing required column "
            "'id', 'template_id', or 'code'; "
            f"found columns: {fields}"
        )
        return

    # decision column is required
    if "decision" not in fields:
        errors.append(
            "coverage.tsv: missing required column 'decision'; "
            f"found columns: {fields}"
        )
        return

    decision_column = fields.index("decision")

    seen = set()
    valid_decisions = {"include", "defer", "exclude"}

    for line_no, line in enumerate(lines[1:], 2):
        parts = [x.strip() for x in line.split(delimiter)]

        if len(parts) < len(fields):
            errors.append(
                f"coverage.tsv:{line_no}: "
                f"expected {len(fields)} columns, found {len(parts)}"
            )
            continue

        template_id = parts[id_column]
        decision = parts[decision_column].lower()

        if not template_id:
            errors.append(
                f"coverage.tsv:{line_no}: empty template/code id"
            )
            continue

        if template_id in seen:
            errors.append(
                f"coverage.tsv:{line_no}: duplicate template/code "
                f"'{template_id}'"
            )
        seen.add(template_id)

        if decision not in valid_decisions:
            errors.append(
                f"coverage.tsv:{line_no}: invalid decision "
                f"'{decision}'; expected one of "
                f"{sorted(valid_decisions)}"
            )

    # Check that every template has a coverage row
    missing = sorted(template_ids - seen)

    if missing:
        errors.append(
            "coverage.tsv: missing coverage entries for "
            f"{len(missing)} templates: {missing[:20]}"
            + (" ..." if len(missing) > 20 else "")
        )

    # Warn about unknown codes
    unknown = sorted(seen - template_ids)

    if unknown:
        errors.append(
            "coverage.tsv: unknown template/code ids: "
            f"{unknown[:20]}"
            + (" ..." if len(unknown) > 20 else "")
        )

def check_dataset(errors, template_ids):
    path = ROOT / "dataset.jsonl"
    rows, errs = load_jsonl(path)
    errors.extend(errs)

    if not rows:
        return

    # Dataset records generated by generate.py use template_id, not id.
    bad_schema = 0
    counts = Counter()
    for i, row in enumerate(rows, 1):
        tid = row.get("template_id")
        if not tid:
            bad_schema += 1
            errors.append(f"dataset.jsonl:{i}: missing required field 'template_id'")
            continue
        if tid not in template_ids:
            errors.append(f"dataset.jsonl:{i}: unknown template_id '{tid}'")
        counts[tid] += 1

    if bad_schema:
        return

    expected_ids = set(template_ids)
    for tid in sorted(expected_ids):
        n = counts.get(tid, 0)
        if n != SAMPLES_PER_TEMPLATE:
            errors.append(
                f"dataset.jsonl: template '{tid}' has {n} samples; "
                f"expected {SAMPLES_PER_TEMPLATE}"
            )

    unexpected = set(counts) - expected_ids
    for tid in sorted(unexpected):
        errors.append(f"dataset.jsonl: unexpected template_id '{tid}'")


def check_dataset_schema(errors):
    rows, _ = load_jsonl(ROOT / "dataset.jsonl")
    if not rows:
        return
    required = {"template_id", "question", "answer", "node_outputs", "question_vars"}
    for i, row in enumerate(rows[:20], 1):
        missing = sorted(required - set(row))
        if missing:
            errors.append(f"dataset.jsonl:{i}: missing required fields {missing}")


def check_wording(errors, rows, name):
    # Lightweight heuristic only: flag explicit graph/solution-step leakage.
    forbidden_patterns = [
        r"\blập\s+.*khoảng",
        r"\bchuẩn\s+hóa\s+cận",
        r"\btính\s+.*dựa\s+trên\s+Var",
        r"\bstandardize\s+the\s+(lower|upper)\s+endpoint",
        r"\bform\s+the\s+interval.*standardize",
    ]
    for i, row in enumerate(rows, 1):
        q = str(row.get("template", row.get("question", "")))
        for pat in forbidden_patterns:
            if re.search(pat, q, re.IGNORECASE):
                errors.append(f"{name}:{i}: possible solution-step leakage: {q}")
                break


def main():
    errors = []
    check_required_files(errors)

    atoms, atom_ids = check_atoms(errors)
    templates, template_ids = check_templates(errors, atom_ids)
    composites, composite_ids = check_composite(errors, atom_ids)
    all_template_ids = template_ids | composite_ids
    graphs, _ = check_graphs(errors, all_template_ids, atom_ids)

    # Both atomic and composite templates are part of the dataset.
    check_coverage(errors, all_template_ids)
    check_dataset(errors, all_template_ids)
    check_dataset_schema(errors)

    check_wording(errors, templates, "templates.jsonl")
    check_wording(errors, composites, "composite.jsonl")

    print(f"ATOMS: {len(atom_ids)} (expected {EXPECTED_ATOMIC})")
    print(f"ATOMIC TEMPLATES: {len(template_ids)} (expected {EXPECTED_ATOMIC})")
    print(f"COMPOSITE TEMPLATES: {len(composite_ids)} (expected {EXPECTED_COMPOSITE})")
    print(f"TOTAL TEMPLATES: {len(all_template_ids)} (expected {EXPECTED_TEMPLATES})")

    dataset_rows, _ = load_jsonl(ROOT / "dataset.jsonl")
    print(f"DATASET: {len(dataset_rows)} records (expected {EXPECTED_DATASET})")

    if errors:
        print(f"\nERRORS: {len(errors)}")
        for e in errors:
            print("  -", e)
        print("RESULT: FAIL")
        return 1

    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
