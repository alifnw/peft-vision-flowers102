"""Render the Mermaid sources separately from CPU notebook execution."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CLI_VERSION = "12.0.0"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmdc", type=Path)
    parser.add_argument("--chromium", type=Path)
    parser.add_argument("--no-sandbox", action="store_true")
    args = parser.parse_args()
    mmdc = args.mmdc or ROOT / "node_modules/.bin/mmdc"
    if not mmdc.is_file():
        raise FileNotFoundError("Mermaid CLI is missing. Run PUPPETEER_SKIP_DOWNLOAD=1 npm ci --ignore-scripts.")
    version = subprocess.check_output([str(mmdc), "--version"], text=True).strip()
    if version != CLI_VERSION:
        raise ValueError(f"Expected Mermaid CLI {CLI_VERSION}, found {version}")
    browser = args.chromium or next((Path(p) for name in ("chromium", "chromium-browser", "google-chrome")
                                     if (p := shutil.which(name))), None)
    if browser is None or not browser.is_file():
        raise FileNotFoundError("Provide an installed Chromium executable with --chromium PATH.")
    folder = ROOT / "assets/checkpoint1"
    config = folder / "diagrams/mermaid-config.json"
    sources = sorted((folder / "diagrams").glob("*.mmd"))
    if not sources:
        raise FileNotFoundError("No Mermaid sources found")
    output = folder / "figures"
    output.mkdir(parents=True, exist_ok=True)
    launch = {"executablePath": str(browser), "args": ["--disable-gpu", "--disable-dev-shm-usage"]}
    if args.no_sandbox:
        launch["args"].append("--no-sandbox")
    metadata = {
        "renderer": {"name": "@mermaid-js/mermaid-cli", "version": version,
                     "mermaid_version": json.loads((ROOT / "node_modules/mermaid/package.json").read_text())["version"],
                     "browser_version": subprocess.check_output([str(browser), "--version"], text=True).strip(),
                     "scale": 3, "maximum_size": 1800, "png_dpi": 300},
        "config_sha256": digest(config), "diagrams": {},
    }
    with tempfile.TemporaryDirectory(prefix="peft-mermaid-") as temporary:
        temporary = Path(temporary)
        launch_path = temporary / "puppeteer.json"
        launch_path.write_text(json.dumps(launch))
        for source in sources:
            record = {"source": source.relative_to(folder).as_posix(), "source_sha256": digest(source)}
            for extension in ("svg", "png"):
                rendered = temporary / f"{source.stem}.{extension}"
                command = [str(mmdc), "--input", str(source), "--output", str(rendered),
                           "--configFile", str(config), "--puppeteerConfigFile", str(launch_path),
                           "--backgroundColor", "white", "--scale", "3", "--size", "1800", "--quiet"]
                subprocess.run(command, check=True, cwd=ROOT)
                if extension == "svg":
                    rendered.write_text("\n".join(line.rstrip() for line in rendered.read_text().splitlines()) + "\n")
                else:
                    with Image.open(rendered) as image:
                        image.load()
                        image.save(rendered, dpi=(300, 300))
                target = output / rendered.name
                shutil.copy2(rendered, target)
                record[extension] = target.relative_to(folder).as_posix()
                record[extension + "_sha256"] = digest(target)
            metadata["diagrams"][source.stem] = record
            print("Rendered:", source.stem, flush=True)
    (folder / "diagrams/rendered.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
