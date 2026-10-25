"""Data contract shared by the simulator and real vessel data.

`logbook/*` is what the vessel operator sends, `env/*` is what our satellite
matching pipeline produces. Real data in this schema drops straight into the
same ML code. `truth/*` exists only in simulations: never train on it.
"""
from __future__ import annotations

import csv
import re
from collections import namedtuple
from pathlib import Path

Column = namedtuple("Column", "name dtype unit required description")

SET_TYPES = ("free_school", "fad", "light")
SCHOOL_DETECTION = ("sonar", "visual", "bird", "fad", "light")
CONDITIONS = ("alive", "injured", "dead")
FATES = ("released_alive", "released_injured", "dead_discarded")

TABLES = {
    "logbook/vessels.csv": [
        Column("vessel_id", "str", "", True, "Unique vessel identifier"),
        Column("home_port", "str", "", True, "Home port name"),
        Column("gross_tonnage", "float", "GT", False, "Gross tonnage"),
        Column("length_m", "float", "m", False, "Length overall"),
        Column("net_length_m", "float", "m", True, "Purse seine net length (effort)"),
        Column("net_depth_m", "float", "m", True, "Purse seine net depth (effort)"),
        Column("hold_capacity_kg", "float", "kg", False, "Fish hold capacity"),
        Column("uses_light", "bool", "0/1", True, "Vessel equipped for light luring"),
    ],
    "logbook/fads.csv": [
        Column("fad_id", "str", "", True, "FAD (sung) identifier, referenced by sets.fad_id"),
        Column("vessel_id", "str", "", True, "Owner vessel"),
        Column("year", "int", "", True, "Deployment year"),
        Column("lat", "float", "deg", True, "FAD latitude (WGS84)"),
        Column("lon", "float", "deg", True, "FAD longitude (WGS84)"),
    ],
    "logbook/trips.csv": [
        Column("trip_id", "str", "", True, "Unique trip identifier"),
        Column("vessel_id", "str", "", True, "Vessel"),
        Column("departure_port", "str", "", True, "Departure port"),
        Column("departure_time_utc", "datetime", "UTC", True, "Departure time"),
        Column("return_port", "str", "", True, "Return port"),
        Column("return_time_utc", "datetime", "UTC", True, "Return time"),
        Column("n_sets", "int", "", True, "Number of sets made on the trip"),
        Column("landed_kg", "float", "kg", False, "Total retained catch landed"),
    ],
    "logbook/sets.csv": [
        Column("event_id", "str", "", True, "Unique set identifier (one row per set, including water hauls)"),
        Column("trip_id", "str", "", True, "Trip"),
        Column("vessel_id", "str", "", True, "Vessel"),
        Column("set_type", "enum:" + "|".join(SET_TYPES), "", True, "Free school, own FAD, or light-luring set"),
        Column("fad_id", "str", "", False, "FAD used (fad sets only)"),
        Column("school_detection", "enum:" + "|".join(SCHOOL_DETECTION), "", False, "How the school was found"),
        Column("start_time_utc", "datetime", "UTC", True, "Net shooting starts"),
        Column("end_time_utc", "datetime", "UTC", True, "Net fully hauled"),
        Column("start_lat", "float", "deg", True, "Position at start of set"),
        Column("start_lon", "float", "deg", True, "Position at start of set"),
        Column("end_lat", "float", "deg", False, "Position at end of set"),
        Column("end_lon", "float", "deg", False, "Position at end of set"),
        Column("search_time_h", "float", "h", True, "Steaming + searching hours since previous set or departure (effort)"),
        Column("light_hours", "float", "h", False, "Hours of light luring before the set (light sets)"),
        Column("net_length_m", "float", "m", True, "Net length used"),
        Column("net_depth_m", "float", "m", True, "Net depth used"),
        Column("bottom_depth_m", "float", "m", False, "Echosounder bottom depth at set start"),
        Column("water_temp_c", "float", "degC", False, "Hull sensor water temperature"),
        Column("wind_beaufort", "int", "Bft", False, "Crew-observed wind force"),
        Column("water_haul", "bool", "0/1", True, "1 = set made but nothing caught"),
        Column("total_catch_kg", "float", "kg", True, "All species, retained + discarded"),
        Column("retained_kg", "float", "kg", True, "Kept on board"),
        Column("discard_kg", "float", "kg", True, "Returned to sea"),
    ],
    "logbook/catch.csv": [
        Column("event_id", "str", "", True, "Set"),
        Column("species_code", "str", "", True, "Checklist code: EVERY checklist species appears for EVERY set"),
        Column("catch_kg", "float", "kg", False, "Weight caught (target / discard groups); 0 if none"),
        Column("count", "int", "n", False, "Individuals caught (ETP species); 0 if none"),
        Column("juvenile_kg", "float", "kg", False, "Part of catch_kg below size at maturity"),
        Column("discard_kg", "float", "kg", False, "Part of catch_kg returned to sea"),
    ],
    "logbook/etp_interactions.csv": [
        Column("event_id", "str", "", True, "Set"),
        Column("species_code", "str", "", True, "ETP species code"),
        Column("individual_no", "int", "", True, "1..n within the set"),
        Column("condition", "enum:" + "|".join(CONDITIONS), "", True, "Condition when released / landed"),
        Column("fate", "enum:" + "|".join(FATES), "", True, "What happened to the animal"),
        Column("photo_ref", "str", "", False, "Photo file name for species verification"),
    ],
    "logbook/track.csv": [
        Column("vessel_id", "str", "", True, "Vessel"),
        Column("trip_id", "str", "", False, "Trip"),
        Column("time_utc", "datetime", "UTC", True, "Ping time"),
        Column("lat", "float", "deg", True, "GPS latitude"),
        Column("lon", "float", "deg", True, "GPS longitude"),
        Column("speed_kn", "float", "kn", True, "Speed over ground"),
        Column("course_deg", "float", "deg", False, "Course over ground"),
        Column("true_activity", "str", "", False, "SIMULATOR ONLY (steaming/searching/drifting/light_luring/setting); real AIS/VMS has no such field"),
    ],
    "env/env_at_sets.csv": [
        Column("event_id", "str", "", True, "Set"),
        Column("time_utc", "datetime", "UTC", True, "Set start"),
        Column("lat", "float", "deg", True, "Set start latitude"),
        Column("lon", "float", "deg", True, "Set start longitude"),
        Column("month", "int", "", True, "Calendar month"),
        Column("sat_sst_c", "float", "degC", True, "Satellite SST (e.g. OISST / MUR)"),
        Column("sat_sst_anom_c", "float", "degC", True, "SST anomaly vs. climatology"),
        Column("sat_chl_mg_m3", "float", "mg/m3", False, "Satellite chlorophyll-a; empty when cloudy"),
        Column("sst_front_c_per_10km", "float", "degC/10km", True, "SST gradient magnitude"),
        Column("wind_ms", "float", "m/s", True, "Reanalysis wind speed (ERA5)"),
        Column("moon_illum", "float", "0-1", True, "Moon illuminated fraction"),
        Column("depth_m", "float", "m", True, "Bathymetry (GEBCO)"),
        Column("dist_coast_km", "float", "km", True, "Distance to coast"),
        Column("seagrass_idx", "float", "0-1", True, "Proximity to seagrass/reef habitat"),
    ],
}

TRUTH_SET_BASE = ["event_id", "set_type", "p_water_haul", "true_sst_c", "true_sst_anom_c", "true_log10chl",
                  "true_wind_ms", "true_front", "true_depth_m"]
TRUTH_CATCH = ["event_id", "species_code", "catch_kg", "count", "juvenile_kg"]


def columns(table):
    return [c.name for c in TABLES[table]]


def truth_set_columns(species_codes):
    return TRUTH_SET_BASE + [f"lambda_{c}" for c in species_codes] + [f"p_present_{c}" for c in species_codes]


def as_json():
    return {t: [c._asdict() for c in cols] for t, cols in TABLES.items()}


def spec_markdown(checklist_rows) -> str:
    out = ["# Purse-seine logbook data specification", "",
           "Generated from `purse_seine_sim/schema.py`. One CSV per table, UTF-8, comma separated,",
           "times in UTC ISO 8601 (`2025-03-14T02:10:00Z`), positions in decimal degrees WGS84.", "",
           "**The most important rule:** every set appears in `sets.csv` (water hauls included) and has one",
           "`catch.csv` row for **every** checklist species, with `0` when it was not caught. Without zeros",
           "the model cannot learn where a species is absent.", ""]
    for table, cols in TABLES.items():
        out += [f"## `{table}`", "", "| column | type | unit | required | description |", "|---|---|---|---|---|"]
        for c in cols:
            out.append(f"| `{c.name}` | {c.dtype} | {c.unit} | {'yes' if c.required else 'no'} | {c.description} |")
        out.append("")
    out += ["## Species checklist", "", "| code | group | unit | English | Thai | scientific |", "|---|---|---|---|---|---|"]
    for r in checklist_rows:
        out.append(f"| `{r['code']}` | {r['group']} | {r['unit']} | {r['name_en']} | {r['name_th']} | *{r['sci_name']}* |")
    out.append("")
    return "\n".join(out)


# =========================
# VALIDATION (simulated or real data)
# =========================

_DT = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _check_value(dtype, v):
    if dtype == "str":
        return True
    if dtype == "float":
        try:
            float(v)
            return True
        except ValueError:
            return False
    if dtype == "int":
        return re.fullmatch(r"-?\d+", v) is not None
    if dtype == "bool":
        return v in ("0", "1")
    if dtype == "datetime":
        return _DT.match(v) is not None
    if dtype.startswith("enum:"):
        return v in dtype[5:].split("|")
    return True


def validate_dir(root, checklist_codes=None, max_errors=50):
    """Return a list of human-readable problems ([] = valid)."""
    root = Path(root)
    errors = []
    event_ids = None
    for table, cols in TABLES.items():
        path = root / table
        if not path.exists():
            if table not in ("logbook/track.csv", "logbook/fads.csv", "logbook/etp_interactions.csv"):
                errors.append(f"{table}: missing")
            continue
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            missing = [c.name for c in cols if c.required and c.name not in header]
            if missing:
                errors.append(f"{table}: missing required columns {missing}")
                continue
            ids = set() if table == "logbook/sets.csv" else None
            species_per_event = {} if table == "logbook/catch.csv" else None
            for n, row in enumerate(reader, start=2):
                for c in cols:
                    v = (row.get(c.name) or "").strip()
                    if v == "":
                        if c.required:
                            errors.append(f"{table}:{n}: empty required '{c.name}'")
                    elif not _check_value(c.dtype, v):
                        errors.append(f"{table}:{n}: bad {c.dtype} in '{c.name}': {v!r}")
                if ids is not None:
                    ids.add(row["event_id"])
                if species_per_event is not None:
                    species_per_event.setdefault(row["event_id"], set()).add(row["species_code"])
                if len(errors) >= max_errors:
                    return errors
            if ids is not None:
                event_ids = ids
            if species_per_event is not None and checklist_codes:
                need = set(checklist_codes)
                for eid, got in species_per_event.items():
                    if got != need:
                        errors.append(f"{table}: event {eid} lacks checklist species {sorted(need - got)}")
                        break
                if event_ids is not None and set(species_per_event) != event_ids:
                    errors.append(f"{table}: sets without catch rows: {len(event_ids - set(species_per_event))}")
    return errors
