from pathlib import Path

root = Path(__file__).resolve().parents[1]
output = root / "outputs" / "expected" / "README_PLACEHOLDER.txt"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    "Replace this file with frozen final outputs from the run used in the submitted paper.\n",
    encoding="utf-8",
)
print(output)
