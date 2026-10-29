"""Cache-busting for the generated *_data.js files loaded by index.html.

GitHub Pages lets browsers and its CDN cache files for 10 minutes, so after a push the new
index.html can run against an old data file and fail. Each export script calls stamp_index() after
writing its file: references like "forecast_data.js" or "forecast_data.js?v=..." in index.html get
?v=<content hash>, so the URL changes exactly when the data does.

    python web_stamp.py            # re-stamp every data file without regenerating it
"""

from pathlib import Path
import hashlib
import re

INDEX_FILE = Path(__file__).with_name("index.html")
DATA_FILES = ["forecast_data.js", "study_area_data.js", "effort_data.js"]


def stamp_index(data_file, index_file=INDEX_FILE):
    data_file, index_file = Path(data_file), Path(index_file)
    version = hashlib.md5(data_file.read_bytes()).hexdigest()[:10]
    html = index_file.read_bytes().decode("utf-8")  # bytes keep the file's line endings
    pattern = re.escape(data_file.name) + r'(?:\?v=[0-9a-f]*)?"'
    new_html, n = re.subn(pattern, f'{data_file.name}?v={version}"', html)
    if n == 0:
        print(f"Warning: {data_file.name} is not referenced in {index_file.name}")
    elif new_html != html:
        index_file.write_bytes(new_html.encode("utf-8"))
        print(f"Stamped {index_file.name}: {data_file.name}?v={version}")


if __name__ == "__main__":
    for name in DATA_FILES:
        path = INDEX_FILE.with_name(name)
        if path.exists():
            stamp_index(path)
