"""Download public CPTAC NSCLC mRNA matrices from the cBioPortal REST API.

Writes the SlotSPE-compatible layout: rows are gene symbols and columns are
cBioPortal sample IDs.  The script resumes a ``.partial`` file safely.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
from pathlib import Path
import time
import urllib.request


API = "https://www.cbioportal.org/api"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "survot_rank" / "research" / "legacy" / "slotspe_runtime" / "dataset_csv" / "raw_rna_data_inter"
STUDIES = (
    ("luad_cptac_2020", "luad_cptac_2020_mrna", "cptac_luad_rna_inter.csv", "RPKM log2"),
    ("lusc_cptac_2021", "lusc_cptac_2021_rna_seq_mrna", "cptac_lusc_rna_inter.csv", "FPKM log2"),
)


def get_json(path: str) -> object:
    request = urllib.request.Request(API + path, headers={"Accept": "application/json", "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(request, timeout=180) as response:
        raw = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
    return json.loads(raw)


def fetch(profile: str, gene_ids: list[int], sample_ids: list[str]) -> list[dict]:
    payload = json.dumps({"entrezGeneIds": gene_ids, "sampleIds": sample_ids}).encode()
    request = urllib.request.Request(
        API + f"/molecular-profiles/{profile}/molecular-data/fetch",
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json", "Accept-Encoding": "gzip"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        raw = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
    return json.loads(raw)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_genes = get_json("/genes?projection=SUMMARY&pageSize=100000&pageNumber=0")
    if not isinstance(raw_genes, list):
        raw_genes = raw_genes["content"]
    by_symbol = {}
    for gene in raw_genes:
        if gene.get("type") != "protein-coding" or gene.get("entrezGeneId", 0) <= 0 or not gene.get("hugoGeneSymbol"):
            continue
        symbol = gene["hugoGeneSymbol"]
        if symbol not in by_symbol or gene["entrezGeneId"] < by_symbol[symbol]["entrezGeneId"]:
            by_symbol[symbol] = gene
    genes = sorted(by_symbol.values(), key=lambda gene: gene["hugoGeneSymbol"])
    manifest = {"source": "cBioPortal public REST API", "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gene_filter": "protein-coding positive Entrez IDs", "studies": []}

    for study, profile, filename, unit in STUDIES:
        final = OUTPUT_DIR / filename
        partial = OUTPUT_DIR / f"{filename}.partial"
        samples = get_json(f"/studies/{study}/samples")
        sample_ids = [sample["sampleId"] for sample in samples]
        if final.exists():
            completed = len(genes)
        else:
            completed = 0
            if partial.exists():
                with partial.open(encoding="utf-8", newline="") as handle:
                    completed = max(0, sum(1 for _ in handle) - 1)
            mode = "a" if partial.exists() else "w"
            with partial.open(mode, encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                if completed == 0:
                    writer.writerow(["gene_symbol", *sample_ids])
                for start in range(completed, len(genes), 1000):
                    batch = genes[start : start + 1000]
                    values = {(row["entrezGeneId"], row["sampleId"]): row.get("value", "") for row in fetch(profile, [gene["entrezGeneId"] for gene in batch], sample_ids)}
                    for gene in batch:
                        writer.writerow([gene["hugoGeneSymbol"], *[values.get((gene["entrezGeneId"], sample), "") for sample in sample_ids]])
                    handle.flush()
                    print(f"{study}: {min(start + len(batch), len(genes))}/{len(genes)}", flush=True)
            os.replace(partial, final)
        manifest["studies"].append({"study_id": study, "molecular_profile": profile, "unit": unit, "output": str(final.relative_to(OUTPUT_DIR.parent.parent.parent.parent.parent)), "samples": len(sample_ids), "genes": len(genes), "bytes": final.stat().st_size})
        print(f"COMPLETE {study}", flush=True)
    with (OUTPUT_DIR / "cptac_nsclc_download_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
