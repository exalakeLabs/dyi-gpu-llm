#!/usr/bin/env python
import json

from project_config import PREPARED_DIR, TRAIN_FILE


def chunks(text, max_chars=2500):
    paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 80]
    buf, size = [], 0

    for p in paras:
        if size + len(p) > max_chars and buf:
            yield "\n\n".join(buf)
            buf, size = [], 0
        buf.append(p)
        size += len(p)

    if buf:
        yield "\n\n".join(buf)


def build_dataset(in_dir=PREPARED_DIR, out_file=TRAIN_FILE) -> int:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    rows = 0

    with out_file.open("w", encoding="utf-8") as out:
        for file in sorted(in_dir.glob("*.txt")):
            text = file.read_text(encoding="utf-8", errors="ignore")
            for chunk in chunks(text):
                out.write(json.dumps({"text": chunk}, ensure_ascii=False) + "\n")
                rows += 1

    print(f"Wrote {rows} rows to {out_file}")
    return rows


def main() -> int:
    build_dataset()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
