#!/usr/bin/env zsh
set -euo pipefail

ROOT="${0:A:h}"
cd "$ROOT"

RAW_TEXT_OUTPUT_DIR="${RAWTEXT_DIR:-}"
RUN_GUTENBERG=1
RUN_WIKIPEDIA=1
RUN_HTML=0

usage() {
  cat <<'EOF'
Usage: ./get_raw_text.zsh [options]

Options:
  --output-dir DIR       Write raw text files to DIR instead of RAWTEXT_DIR.
  --skip-gutenberg      Do not download Project Gutenberg books.
  --skip-wikipedia      Do not download Wikipedia plaintext pages.
  --run-html            Download and convert URLs listed in HTML_URLS.
  -h, --help            Show this help.

Environment:
  RAWTEXT_DIR           Default output directory, usually set by .env/.runtime.
  PYTHON                Python executable to use after .runtime is loaded.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      RAW_TEXT_OUTPUT_DIR="${2:?missing directory for --output-dir}"
      shift 2
      ;;
    --skip-gutenberg)
      RUN_GUTENBERG=0
      shift
      ;;
    --skip-wikipedia)
      RUN_WIKIPEDIA=0
      shift
      ;;
    --run-html)
      RUN_HTML=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      print -u2 "error: unknown option: $1"
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -f "$ROOT/.runtime" ]]; then
  source "$ROOT/.runtime" >/dev/null
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
fi

PYTHON="${PYTHON:-python}"
RAW_TEXT_OUTPUT_DIR="${RAW_TEXT_OUTPUT_DIR:-${RAWTEXT_DIR:-text}}"

if ! "$PYTHON" - <<'PY'
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd() / "src"))
import http_client
PY
then
  print -u2 "error: selected Python cannot import the raw-text downloader helpers: $PYTHON"
  print -u2 "Run ./install.zsh or set PYTHON=/path/to/.venv/bin/python and retry."
  exit 1
fi

GUTENBERG_TASKS=(
  "science||400"
  "mathematics||250"
  "statistics||120"
  "economics||180"
  "finance||100"
  "banking||100"
  "commerce||120"
  "law||150"
  "government||100"
  "engineering||180"
  "logic||100"
  "philosophy||150"
  "history|Modern|120"
  "biography||150"
)

WIKIPEDIA_TITLES=(
  "Artificial intelligence"
  "Machine learning"
  "Natural language processing"
  "Information retrieval"
  "Statistics"
  "Economics"
  "Computer science"
  "Software engineering"
)

HTML_URLS=(
  "https://en.wikipedia.org/wiki/Artificial_intelligence"
  "https://en.wikipedia.org/wiki/Natural_language_processing"
)

run_cmd() {
  print
  print "==> $*"
  "$@"
}

download_gutenberg() {
  local spec query rest topic max_books
  local -a cmd

  for spec in "${GUTENBERG_TASKS[@]}"; do
    query="${spec%%|*}"
    rest="${spec#*|}"
    topic="${rest%%|*}"
    max_books="${rest##*|}"

    cmd=(
      "$PYTHON"
      "$ROOT/src/download_gutenberg.py"
      --query "$query"
      --max-books "$max_books"
      --output-dir "$RAW_TEXT_OUTPUT_DIR"
    )

    if [[ -n "$topic" ]]; then
      cmd+=(--topic "$topic")
    fi

    run_cmd "${cmd[@]}"
  done
}

download_wikipedia() {
  local title

  for title in "${WIKIPEDIA_TITLES[@]}"; do
    run_cmd \
      "$PYTHON" \
      "$ROOT/src/download_web_text.py" \
      --wikipedia-title "$title" \
      --output-dir "$RAW_TEXT_OUTPUT_DIR"
  done
}

download_html() {
  local url

  for url in "${HTML_URLS[@]}"; do
    run_cmd \
      "$PYTHON" \
      "$ROOT/src/download_web_text.py" \
      --url "$url" \
      --output-dir "$RAW_TEXT_OUTPUT_DIR"
  done
}

mkdir -p "$RAW_TEXT_OUTPUT_DIR"

print "Raw text output: $RAW_TEXT_OUTPUT_DIR"
print "Python: $PYTHON"

if (( RUN_GUTENBERG )); then
  print
  print "Project Gutenberg"
  download_gutenberg
fi

if (( RUN_WIKIPEDIA )); then
  print
  print "Wikipedia plaintext"
  download_wikipedia
fi

if (( RUN_HTML )); then
  print
  print "HTML to plaintext"
  download_html
fi
