import json
import subprocess
import sys
from pathlib import Path
from collections import Counter


HERE = Path(__file__).parent


def run_command(cmd, title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print("CMD:", " ".join(cmd))

    result = subprocess.run(
        cmd,
        cwd=HERE,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    if result.returncode != 0:
        print(f"\n[FAIL] {title}")
        return False

    print(f"\n[PASS] {title}")
    return True


def check_jsonl(path, required_fields=None):
    path = HERE / path

    print("\n" + "=" * 70)
    print(f"CHECK FILE: {path.name}")
    print("=" * 70)

    if not path.exists():
        print(f"[FAIL] File not found: {path}")
        return False

    required_fields = required_fields or []

    count = 0
    errors = 0
    partitions = Counter()
    templates = Counter()
    instances = set()

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue

            count += 1

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[FAIL] Invalid JSON at line {line_no}: {e}")
                errors += 1
                continue

            for field in required_fields:
                if field not in row:
                    print(
                        f"[FAIL] line {line_no}: "
                        f"missing field '{field}'"
                    )
                    errors += 1

            if "instance_id" in row:
                iid = row["instance_id"]

                if iid in instances:
                    print(
                        f"[FAIL] duplicate instance_id "
                        f"at line {line_no}: {iid}"
                    )
                    errors += 1

                instances.add(iid)

            if "partition" in row:
                partitions[row["partition"]] += 1

            if "template_id" in row:
                templates[row["template_id"]] += 1

    print(f"Records: {count}")

    if partitions:
        print("\nPartitions:")
        for p, n in sorted(partitions.items()):
            print(f"  {p:<35} {n}")

    print(f"\nUnique instance_id: {len(instances)}")
    print(f"Unique templates:   {len(templates)}")

    if templates:
        print("\nTop templates:")
        for tid, n in templates.most_common(10):
            print(f"  {tid:<45} {n}")

    if errors:
        print(f"\n[FAIL] {errors} problem(s)")
        return False

    print("\n[PASS] JSONL structure is valid")
    return True


def check_partition_balance(path):
    path = HERE / path

    print("\n" + "=" * 70)
    print("CHECK PARTITION BALANCE")
    print("=" * 70)

    if not path.exists():
        print(f"[FAIL] File not found: {path}")
        return False

    counts = Counter()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            if "partition" in row:
                counts[row["partition"]] += 1

    if not counts:
        print("[WARN] No partition field found")
        return False

    total = sum(counts.values())

    print(f"Total records: {total}\n")

    for name, count in sorted(counts.items()):
        percentage = 100 * count / total
        print(
            f"{name:<35} "
            f"{count:>6} "
            f"({percentage:6.2f}%)"
        )

    print("\n[PASS] Partition counts inspected")
    return True


def check_generated(path):
    path = HERE / path

    print("\n" + "=" * 70)
    print("CHECK GENERATED.JSONL")
    print("=" * 70)

    if not path.exists():
        print(f"[FAIL] File not found: {path}")
        return False

    count = 0
    templates = Counter()
    bad = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"[FAIL] Invalid JSON line {line_no}")
                bad += 1
                continue

            count += 1

            if "template_id" not in row:
                print(
                    f"[FAIL] line {line_no}: "
                    "missing template_id"
                )
                bad += 1

            if "question" not in row:
                print(
                    f"[FAIL] line {line_no}: "
                    "missing question"
                )
                bad += 1

            if "answer" not in row:
                print(
                    f"[FAIL] line {line_no}: "
                    "missing answer"
                )
                bad += 1

            if "template_id" in row:
                templates[row["template_id"]] += 1

    print(f"Total samples: {count}")
    print(f"Templates:     {len(templates)}")

    if templates:
        values = list(templates.values())
        print(f"Min/template:  {min(values)}")
        print(f"Max/template:  {max(values)}")
        print(f"Avg/template:  {sum(values) / len(values):.2f}")

    if bad:
        print(f"\n[FAIL] {bad} problems found")
        return False

    print("\n[PASS] generated.jsonl looks valid")
    return True


def main():
    print("=" * 70)
    print("MM4 DATASET PIPELINE CHECK")
    print("=" * 70)

    results = []

    # 1. Validate atom/template/composite/graph definitions
    results.append(
        run_command(
            [
                sys.executable,
                "validate.py",
                "--draws",
                "100"
            ],
            "1. VALIDATE PROJECT DEFINITIONS"
        )
    )

    # 2. Check generated.jsonl
    results.append(
        check_generated("generated.jsonl")
    )

    # 3. Check generated.jsonl structure
    results.append(
        check_jsonl(
            "generated.jsonl",
            required_fields=[
                "template_id",
                "question",
                "answer"
            ]
        )
    )

    # 4. Check dataset partitions
    partition_candidates = [
        "dataset_public.jsonl",
        "dataset.jsonl",
        "public.jsonl",
        "atomic_train.jsonl",
    ]

    found_partition_file = None

    for candidate in partition_candidates:
        if (HERE / candidate).exists():
            found_partition_file = candidate
            break

    if found_partition_file:
        results.append(
            check_jsonl(
                found_partition_file,
                required_fields=[
                    "instance_id",
                    "template_id",
                    "partition",
                    "question"
                ]
            )
        )

        results.append(
            check_partition_balance(found_partition_file)
        )
    else:
        print("\n" + "=" * 70)
        print("CHECK PARTITION FILE")
        print("=" * 70)
        print(
            "[INFO] No standard partition file found yet. "
            "This is OK if build_dataset.py has not been run."
        )

    # 5. Final report
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    passed = sum(results)
    total = len(results)

    print(f"Passed: {passed}/{total}")

    if all(results):
        print("\nALL CHECKS PASSED")
        return 0

    print("\nSOME CHECKS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())