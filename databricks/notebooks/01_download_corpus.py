# Databricks notebook source
# =============================================================================
# 01 · Download Gutenberg Corpus → DBFS
# Downloads Project Gutenberg plain-text books and writes them to DBFS.
# Can be run on a CPU cluster (no GPU needed).
# =============================================================================

# COMMAND ----------
# MAGIC %md
# MAGIC ## 01 · Download Gutenberg Corpus
# MAGIC
# MAGIC Downloads public-domain books from [Project Gutenberg](https://www.gutenberg.org/)
# MAGIC via the [Gutendex API](https://gutendex.com/) and stores them under
# MAGIC `{dbfs_root}/text/`.
# MAGIC
# MAGIC **Cluster**: CPU cluster is sufficient.
# MAGIC **Estimated time**: 30–90 min depending on `max_books_per_query`.

# COMMAND ----------
# MAGIC %pip install --quiet requests truststore

# COMMAND ----------
# Widget parameters — edit before running
dbutils.widgets.text("dbfs_root",           "/dbfs/FileStore/llama32", "DBFS Root")
dbutils.widgets.text("max_books_per_query", "100",                      "Max Books Per Query (scale down for testing)")
dbutils.widgets.dropdown("profile",         "smart_assistant",          ["smart_assistant", "tech_biz", "bookish"], "Corpus Profile")

# COMMAND ----------
import os, sys
dbfs_root = dbutils.widgets.get("dbfs_root")
max_books  = int(dbutils.widgets.get("max_books_per_query"))
profile    = dbutils.widgets.get("profile")

os.environ["LLAMA_DBFS_ROOT"] = dbfs_root
os.environ["LLAMA_TEXT_DIR"]  = f"{dbfs_root}/text"

_nb_path   = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_repo_root = "/Workspace/" + "/".join(_nb_path.lstrip("/").split("/")[1:4])
_src       = os.path.join(_repo_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from pathlib import Path
text_dir = Path(f"{dbfs_root}/text")
text_dir.mkdir(parents=True, exist_ok=True)

print(f"DBFS root  : {dbfs_root}")
print(f"Text dir   : {text_dir}")
print(f"Max books  : {max_books} per query")
print(f"Profile    : {profile}")

# COMMAND ----------
# MAGIC %md ### Download function

# COMMAND ----------
from download_gutenberg import iter_books, pick_text_url, safe_name
import requests

def download_corpus(queries, text_dir: Path):
    """Download books for a list of (query, topic, max_books) tuples."""
    total_downloaded = 0
    total_skipped    = 0

    for query, topic, n_books in queries:
        label = f"{query}" + (f" [{topic}]" if topic else "")
        count = skipped = 0

        for book in iter_books(query, topic, "en", n_books):
            title    = book.get("title", "unknown")
            book_id  = book.get("id")
            authors  = ", ".join(a.get("name", "?") for a in book.get("authors", []))
            url      = pick_text_url(book.get("formats", {}))

            if not url:
                skipped += 1
                continue

            fname = f"{book_id}_{safe_name(title)}.txt"
            path  = text_dir / fname
            if path.exists():
                continue

            try:
                r = requests.get(url, timeout=120)
                r.raise_for_status()
                path.write_text(r.text, encoding="utf-8", errors="ignore")
                count += 1
            except Exception as e:
                print(f"  [warn] {book_id}: {e}")
                skipped += 1

        print(f"  {label}: {count} downloaded, {skipped} skipped")
        total_downloaded += count
        total_skipped    += skipped

    return total_downloaded, total_skipped

# COMMAND ----------
# MAGIC %md ### Corpus profiles

# COMMAND ----------
# Each entry: (query, topic_filter_or_None, max_books)
PROFILES = {
    "smart_assistant": [
        ("science",      None,       min(max_books, 600)),
        ("mathematics",  None,       min(max_books, 250)),
        ("physics",      None,       min(max_books, 200)),
        ("chemistry",    None,       min(max_books, 150)),
        ("biology",      None,       min(max_books, 200)),
        ("astronomy",    None,       min(max_books, 150)),
        ("history",      "Ancient",  min(max_books, 150)),
        ("history",      "Medieval", min(max_books, 150)),
        ("history",      "Modern",   min(max_books, 150)),
        ("philosophy",   None,       min(max_books, 250)),
        ("logic",        None,       min(max_books, 120)),
        ("ethics",       None,       min(max_books, 120)),
        ("psychology",   None,       min(max_books, 150)),
        ("economics",    None,       min(max_books, 180)),
        ("biography",    None,       min(max_books, 250)),
        ("education",    None,       min(max_books, 120)),
    ],
    "tech_biz": [
        ("science",          None,     min(max_books, 400)),
        ("mathematics",      None,     min(max_books, 250)),
        ("statistics",       None,     min(max_books, 120)),
        ("economics",        None,     min(max_books, 180)),
        ("finance",          None,     min(max_books, 100)),
        ("banking",          None,     min(max_books, 100)),
        ("engineering",      None,     min(max_books, 180)),
        ("law",              None,     min(max_books, 150)),
        ("logic",            None,     min(max_books, 100)),
        ("philosophy",       None,     min(max_books, 150)),
        ("history",          "Modern", min(max_books, 120)),
        ("biography",        None,     min(max_books, 150)),
    ],
    "bookish": [
        ("science",    None,       min(max_books, 300)),
        ("history",    "Ancient",  min(max_books, 120)),
        ("history",    "Modern",   min(max_books, 120)),
        ("philosophy", None,       min(max_books, 250)),
        ("psychology", None,       min(max_books, 120)),
        ("biography",  None,       min(max_books, 250)),
        ("travel",     None,       min(max_books, 120)),
        ("geography",  None,       min(max_books, 120)),
        ("language",   None,       min(max_books, 100)),
    ],
}

queries = PROFILES[profile]
print(f"Running profile '{profile}' — {len(queries)} query groups")

# COMMAND ----------
# MAGIC %md ### Run download

# COMMAND ----------
downloaded, skipped = download_corpus(queries, text_dir)
total_files = len(list(text_dir.glob("*.txt")))
print(f"\n{'='*50}")
print(f"Session: {downloaded} downloaded, {skipped} skipped")
print(f"Total files in text_dir: {total_files}")

# COMMAND ----------
# MAGIC %md ### Preview DBFS file list

# COMMAND ----------
files = dbutils.fs.ls(f"dbfs:{text_dir.as_posix().replace('/dbfs', '')}")
display(spark.createDataFrame(
    [(f.name, f"{f.size / 1024:.1f} KB") for f in files[:20]],
    ["filename", "size"]
))
print(f"Showing 20 of {len(files)} files.")

# COMMAND ----------
# MAGIC %md ### ✅ Corpus downloaded — proceed to notebook 02.
