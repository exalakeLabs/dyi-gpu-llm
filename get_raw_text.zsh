#!/usr/bin/env zsh
set -euo pipefail

ROOT="${0:A:h}"
cd "$ROOT"

RAW_TEXT_OUTPUT_DIR="${RAWTEXT_DIR:-}"
RUN_GUTENBERG=1
RUN_WIKIPEDIA=1
RUN_HTML=0
WIKI_CRAWL_DEPTH="${WIKI_CRAWL_DEPTH:-1}"
WIKI_CRAWL_MAX_PAGES="${WIKI_CRAWL_MAX_PAGES:-240}"
WIKI_CRAWL_LINKS_PER_PAGE="${WIKI_CRAWL_LINKS_PER_PAGE:-35}"

usage() {
  cat <<'EOF'
Usage: ./get_raw_text.zsh [options]

Options:
  --output-dir DIR       Write raw text files to DIR instead of RAWTEXT_DIR.
  --skip-gutenberg      Do not download Project Gutenberg books.
  --skip-wikipedia      Do not download Wikipedia plaintext pages.
  --run-html            Download and convert URLs listed in HTML_URLS.
  --wiki-crawl-depth N  Wikipedia link depth from each seed page (default: 1).
  --wiki-crawl-max N    Max crawled Wikipedia pages per topic crawl (default: 240).
  --wiki-crawl-links N  Max linked pages enqueued per crawled page (default: 35).
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
    --wiki-crawl-depth)
      WIKI_CRAWL_DEPTH="${2:?missing value for --wiki-crawl-depth}"
      shift 2
      ;;
    --wiki-crawl-max)
      WIKI_CRAWL_MAX_PAGES="${2:?missing value for --wiki-crawl-max}"
      shift 2
      ;;
    --wiki-crawl-links)
      WIKI_CRAWL_LINKS_PER_PAGE="${2:?missing value for --wiki-crawl-links}"
      shift 2
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
  "Data science"
  "Probability theory"
  "Linear algebra"
  "Optimization"
  "Algorithms"
  "Database"
  "Distributed computing"
  "Cybersecurity"
  "Chemical compound"
  "Chemistry"
  "Organic chemistry"
  "Physical chemistry"
  "Biochemistry"
  "Polymer chemistry"
  "Medicinal chemistry"
  "Quantum mechanics"
  "Quantum field theory"
  "Particle physics"
  "Condensed matter physics"
  "Atomic physics"
  "Optics"
  "Automotive engineering"
  "Automobile"
  "Internal combustion engine"
  "Electric vehicle"
  "Hybrid vehicle"
  "Vehicle dynamics"
  "Automotive industry"
)

WIKIPEDIA_CRAWL_SEEDS=(
  "organic|Organic chemistry|Functional group|Organic reaction|Stereochemistry|Aromaticity|Polymer chemistry|Medicinal chemistry"
  "quantum|Quantum mechanics|Quantum physics|Quantum field theory|Particle physics|Condensed matter physics|Atomic physics|Quantum information"
  "automotive|Automotive engineering|Automobile|Internal combustion engine|Electric vehicle|Hybrid vehicle|Vehicle dynamics|Automotive industry"
)

ORGANIC_CRAWL_TERMS=(
  "organic"
  "chem"
  "reaction"
  "compound"
  "carbon"
  "hydrocarbon"
  "functional"
  "synthesis"
  "polymer"
  "aromatic"
  "stereo"
)

QUANTUM_CRAWL_TERMS=(
  "quantum"
  "particle"
  "field"
  "atom"
  "atomic"
  "photon"
  "electron"
  "wave"
  "physics"
  "mechanics"
)

AUTOMOTIVE_CRAWL_TERMS=(
  "automotive"
  "automobile"
  "vehicle"
  "engine"
  "motor"
  "car"
  "transmission"
  "brake"
  "tire"
  "electric"
  "hybrid"
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
  local group label rest seed term
  local -a cmd seeds terms

  for title in "${WIKIPEDIA_TITLES[@]}"; do
    run_cmd \
      "$PYTHON" \
      "$ROOT/src/download_web_text.py" \
      --wikipedia-title "$title" \
      --output-dir "$RAW_TEXT_OUTPUT_DIR"
  done

  for group in "${WIKIPEDIA_CRAWL_SEEDS[@]}"; do
    label="${group%%|*}"
    rest="${group#*|}"
    seeds=("${(@ps:|:)rest}")

    case "$label" in
      organic)
        terms=("${ORGANIC_CRAWL_TERMS[@]}")
        ;;
      quantum)
        terms=("${QUANTUM_CRAWL_TERMS[@]}")
        ;;
      automotive)
        terms=("${AUTOMOTIVE_CRAWL_TERMS[@]}")
        ;;
      *)
        terms=()
        ;;
    esac

    cmd=(
      "$PYTHON"
      "$ROOT/src/download_web_text.py"
      --output-dir "$RAW_TEXT_OUTPUT_DIR"
      --crawl-depth "$WIKI_CRAWL_DEPTH"
      --crawl-max-pages "$WIKI_CRAWL_MAX_PAGES"
      --crawl-links-per-page "$WIKI_CRAWL_LINKS_PER_PAGE"
    )

    for seed in "${seeds[@]}"; do
      cmd+=(--wikipedia-crawl-title "$seed")
    done

    for term in "${terms[@]}"; do
      cmd+=(--crawl-include "$term")
    done

    run_cmd "${cmd[@]}"
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
print "Wikipedia crawl: depth=$WIKI_CRAWL_DEPTH max_pages=$WIKI_CRAWL_MAX_PAGES links_per_page=$WIKI_CRAWL_LINKS_PER_PAGE"

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
