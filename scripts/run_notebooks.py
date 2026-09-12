"""Execute notebooks in order, with a fresh kernel per notebook and saved outputs."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--from-notebook", default="00")
    p.add_argument("--through", default="03_model_creation.ipynb")
    args = p.parse_args()
    results = []
    report = ROOT / "reports/current/notebook_execution.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    if report.exists():
        results = json.loads(report.read_text(encoding="utf-8"))
    os.environ["PYTHONUTF8"] = "1"
    os.environ["JUPYTER_PATH"] = str(Path(sys.prefix) / "share/jupyter")
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        if path.name < args.from_notebook or path.name > args.through:
            continue
        results = [r for r in results if r["notebook"] != path.name]
        print(f"Running {path.name}", flush=True)
        started = time.monotonic()
        notebook = nbformat.read(path, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
        try:
            NotebookClient(
                notebook,
                timeout=None,
                kernel_name="music-gnn",
                resources={"metadata": {"path": str(ROOT)}},
            ).execute()
            results.append(
                {
                    "notebook": path.name,
                    "status": "passed",
                    "seconds": time.monotonic() - started,
                }
            )
        except Exception as e:
            results.append(
                {
                    "notebook": path.name,
                    "status": "failed",
                    "error": str(e),
                    "seconds": time.monotonic() - started,
                }
            )
            raise
        finally:
            nbformat.write(notebook, path)
            report.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Completed {path.name} in {time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
