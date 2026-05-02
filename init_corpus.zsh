#!/usr/bin/env zsh
set -euo pipefail

ROOT="${0:A:h}"
cd "$ROOT"

#smart assitant
./src/download_gutenberg.py --query "science" --max-books 600
./src/download_gutenberg.py --query "mathematics" --max-books 250
./src/download_gutenberg.py --query "physics" --max-books 200
./src/download_gutenberg.py --query "chemistry" --max-books 150
./src/download_gutenberg.py --query "biology" --max-books 200
./src/download_gutenberg.py --query "astronomy" --max-books 150

./src/download_gutenberg.py --query "history" --topic "Ancient" --max-books 150
./src/download_gutenberg.py --query "history" --topic "Medieval" --max-books 150
./src/download_gutenberg.py --query "history" --topic "Modern" --max-books 150

./src/download_gutenberg.py --query "philosophy" --max-books 250
./src/download_gutenberg.py --query "logic" --max-books 120
./src/download_gutenberg.py --query "ethics" --max-books 120
./src/download_gutenberg.py --query "psychology" --max-books 150

./src/download_gutenberg.py --query "economics" --max-books 180
./src/download_gutenberg.py --query "commerce" --max-books 120
./src/download_gutenberg.py --query "banking" --max-books 100
./src/download_gutenberg.py --query "finance" --max-books 100
./src/download_gutenberg.py --query "accounting" --max-books 80

./src/download_gutenberg.py --query "law" --max-books 150
./src/download_gutenberg.py --query "government" --max-books 120
./src/download_gutenberg.py --query "political science" --max-books 100

./src/download_gutenberg.py --query "engineering" --max-books 180
./src/download_gutenberg.py --query "mechanics" --max-books 120
./src/download_gutenberg.py --query "electricity" --max-books 120
./src/download_gutenberg.py --query "railway" --max-books 80
./src/download_gutenberg.py --query "navigation" --max-books 80

./src/download_gutenberg.py --query "geography" --max-books 120
./src/download_gutenberg.py --query "travel" --max-books 150
./src/download_gutenberg.py --query "biography" --max-books 250
./src/download_gutenberg.py --query "education" --max-books 120
./src/download_gutenberg.py --query "language" --max-books 120
./src/download_gutenberg.py --query "grammar" --max-books 100

#tech biz
./src/download_gutenberg.py --query "science" --max-books 400
./src/download_gutenberg.py --query "mathematics" --max-books 250
./src/download_gutenberg.py --query "statistics" --max-books 120
./src/download_gutenberg.py --query "economics" --max-books 180
./src/download_gutenberg.py --query "finance" --max-books 100
./src/download_gutenberg.py --query "banking" --max-books 100
./src/download_gutenberg.py --query "commerce" --max-books 120
./src/download_gutenberg.py --query "law" --max-books 150
./src/download_gutenberg.py --query "government" --max-books 100
./src/download_gutenberg.py --query "engineering" --max-books 180
./src/download_gutenberg.py --query "logic" --max-books 100
./src/download_gutenberg.py --query "philosophy" --max-books 150
./src/download_gutenberg.py --query "history" --topic "Modern" --max-books 120
./src/download_gutenberg.py --query "biography" --max-books 150

# If you want a more “bookish but intelligent” assistant tone
./src/download_gutenberg.py --query "science" --max-books 300
./src/download_gutenberg.py --query "history" --topic "Ancient" --max-books 120
./src/download_gutenberg.py --query "history" --topic "Modern" --max-books 120
./src/download_gutenberg.py --query "philosophy" --max-books 250
./src/download_gutenberg.py --query "logic" --max-books 120
./src/download_gutenberg.py --query "psychology" --max-books 120
./src/download_gutenberg.py --query "biography" --max-books 250
./src/download_gutenberg.py --query "travel" --max-books 120
./src/download_gutenberg.py --query "geography" --max-books 120
./src/download_gutenberg.py --query "language" --max-books 100
./src/download_gutenberg.py --query "grammar" --max-books 100