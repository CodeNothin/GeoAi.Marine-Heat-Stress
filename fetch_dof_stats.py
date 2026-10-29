"""Download Department of Fisheries (กรมประมง) statistics and condense them to real_env/dof_stats.json.

    python fetch_dof_stats.py                 # download, then summarise
    python fetch_dof_stats.py --raw dof_raw   # summarise CSVs already in dof_raw/

Source: the DOF open data catalog, group "strategy-stat" (https://catalog.fisheries.go.th/group/strategy-stat),
read through its CKAN API (no account):
  - thaivessel           จำนวนเรือประมงไทย: licensed commercial vessels per year, province, gear, size, sea
  - dofd07_05_0101_07    ปริมาณการลงแรงทำการประมง: commercial fishing trips/days/hauls/hours per month, gear, size, sea
  - landing              ปริมาณและมูลค่าสัตว์น้ำเค็มขึ้นท่าเทียบเรือรายจังหวัด: landings (t) per month, province, gear, species

Output (Gulf of Thailand only, years in CE):
  vessels[province][year]           licensed vessels
  gulf_vessels[year]                licensed vessels, all Gulf provinces
  gulf_effort_days["YYYY-MM"]       recorded fishing days, all Gulf commercial gears
  landings_t[province]["YYYY-MM"]   landed tonnes of the forecast's target species, all gears
export_effort_web.py reads this file to size each area's fleet and to compute the real CPUE.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import urllib.request
from datetime import date
from pathlib import Path

API = "https://catalog.fisheries.go.th/api/3/action/package_show?id="
OUT_FILE = Path("real_env/dof_stats.json")
RAW_DIR = Path("real_env/dof_raw")
GULF = "อ่าวไทย"
TH_MONTHS = ["มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
             "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม"]
# Study-area provinces (keys used in index.html) and the neighbours whose boats also fish there.
PROVINCES = {"chumphon": "ชุมพร", "surat": "สุราษฎร์ธานี", "nakhon": "นครศรีธรรมราช", "songkhla": "สงขลา",
             "pattani": "ปัตตานี", "narathiwat": "นราธิวาส", "prachuap": "ประจวบคีรีขันธ์"}
# Thai landing names of the forecast's target species (purse_seine_sim.species checklist).
TARGETS = {"ปลาทู": "RAST_BRA", "ปลาลัง": "RAST_KAN", "ปลาหลังเขียว": "SARD_GIB", "ปลาทูแขก": "DECA_MAR",
           "ปลาโอดำ": "THUN_TON", "ปลาโอลาย": "EUTH_AFF"}
LANDING_YEARS_BE = ("2565", "2566", "2567")   # 2022-2024; the catalog has no later province landings yet


def resources(package):
    with urllib.request.urlopen(API + package, timeout=60) as r:
        return json.load(r)["result"]["resources"]


def download(url, path):
    print(f"  {path.name} <- {url}")
    with urllib.request.urlopen(url, timeout=180) as r:
        path.write_bytes(r.read())


def fetch(raw):
    raw.mkdir(parents=True, exist_ok=True)
    download(resources("thaivessel")[0]["url"], raw / "vessels.csv")
    download(resources("dofd07_05_0101_07")[0]["url"], raw / "effort.csv")
    for res in resources("landing"):
        year = next((y for y in LANDING_YEARS_BE if y in res.get("name", "")), None)
        if year:
            download(res["url"], raw / f"landing_{year}.csv")


def rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        return float(str(v).replace(",", ""))
    except ValueError:
        return 0.0


def ym(year_be, month_th):
    return f"{int(year_be) - 543}-{TH_MONTHS.index(month_th.strip()) + 1:02d}"


def summarise(raw):
    by_name = {v: k for k, v in PROVINCES.items()}
    vessels = collections.defaultdict(dict)
    gulf_vessels = collections.Counter()
    for r in rows(raw / "vessels.csv"):
        if r["พื้นที่ทำการประมง"] != GULF:
            continue
        year, n = int(r["ปี"]) - 543, int(num(r["จำนวนเรือ"]))
        gulf_vessels[year] += n
        key = by_name.get(r["จังหวัดทีออกใบอนุญาต"].strip())
        if key:
            vessels[key][year] = vessels[key].get(year, 0) + n

    effort = collections.Counter()
    for r in rows(raw / "effort.csv"):
        if r["พื้นที่ทำการประมง"] == GULF:
            effort[ym(r["ปี"], r["เดือน"])] += num(r["จำนวนวัน"])

    landings = collections.defaultdict(collections.Counter)
    for path in sorted(raw.glob("landing_*.csv")):
        for r in rows(path):
            key = by_name.get(r["จังหวัด"].strip())
            if key and r["ชนิดสัตว์น้ำ"].strip() in TARGETS:
                landings[key][ym(r["ปี"], r["เดือน"])] += num(r["ปริมาณ (ตัน)"])

    return {
        "meta": {"generated": date.today().isoformat(),
                 "source": "Department of Fisheries open data, catalog.fisheries.go.th/group/strategy-stat",
                 "sea": GULF, "provinces": PROVINCES, "target_species": TARGETS},
        "vessels": {k: dict(sorted(v.items())) for k, v in vessels.items()},
        "gulf_vessels": dict(sorted(gulf_vessels.items())),
        "gulf_effort_days": dict(sorted(effort.items())),
        "landings_t": {k: dict(sorted(v.items())) for k, v in landings.items()},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", help="folder with already-downloaded CSVs (skip the download)")
    p.add_argument("--out", default=str(OUT_FILE))
    a = p.parse_args()
    raw = Path(a.raw) if a.raw else RAW_DIR
    if not a.raw:
        print("Downloading DOF statistics:")
        fetch(raw)
    stats = summarise(raw)
    out = Path(a.out)
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    months = list(stats["gulf_effort_days"])
    print(f"Created: {out.resolve()} | effort {months[0]} .. {months[-1]} | "
          f"vessels {', '.join(f'{k} {max(v.items())[1]}' for k, v in stats['vessels'].items())}")


if __name__ == "__main__":
    main()
