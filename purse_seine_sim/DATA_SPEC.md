# Purse-seine logbook data specification

Generated from `purse_seine_sim/schema.py`. One CSV per table, UTF-8, comma separated,
times in UTC ISO 8601 (`2025-03-14T02:10:00Z`), positions in decimal degrees WGS84.

**The most important rule:** every set appears in `sets.csv` (water hauls included) and has one
`catch.csv` row for **every** checklist species, with `0` when it was not caught. Without zeros
the model cannot learn where a species is absent.

## `logbook/vessels.csv`

| column | type | unit | required | description |
|---|---|---|---|---|
| `vessel_id` | str |  | yes | Unique vessel identifier |
| `home_port` | str |  | yes | Home port name |
| `gross_tonnage` | float | GT | no | Gross tonnage |
| `length_m` | float | m | no | Length overall |
| `net_length_m` | float | m | yes | Purse seine net length (effort) |
| `net_depth_m` | float | m | yes | Purse seine net depth (effort) |
| `hold_capacity_kg` | float | kg | no | Fish hold capacity |
| `uses_light` | bool | 0/1 | yes | Vessel equipped for light luring |

## `logbook/fads.csv`

| column | type | unit | required | description |
|---|---|---|---|---|
| `fad_id` | str |  | yes | FAD (sung) identifier, referenced by sets.fad_id |
| `vessel_id` | str |  | yes | Owner vessel |
| `year` | int |  | yes | Deployment year |
| `lat` | float | deg | yes | FAD latitude (WGS84) |
| `lon` | float | deg | yes | FAD longitude (WGS84) |

## `logbook/trips.csv`

| column | type | unit | required | description |
|---|---|---|---|---|
| `trip_id` | str |  | yes | Unique trip identifier |
| `vessel_id` | str |  | yes | Vessel |
| `departure_port` | str |  | yes | Departure port |
| `departure_time_utc` | datetime | UTC | yes | Departure time |
| `return_port` | str |  | yes | Return port |
| `return_time_utc` | datetime | UTC | yes | Return time |
| `n_sets` | int |  | yes | Number of sets made on the trip |
| `landed_kg` | float | kg | no | Total retained catch landed |

## `logbook/sets.csv`

| column | type | unit | required | description |
|---|---|---|---|---|
| `event_id` | str |  | yes | Unique set identifier (one row per set, including water hauls) |
| `trip_id` | str |  | yes | Trip |
| `vessel_id` | str |  | yes | Vessel |
| `set_type` | enum:free_school|fad|light |  | yes | Free school, own FAD, or light-luring set |
| `fad_id` | str |  | no | FAD used (fad sets only) |
| `school_detection` | enum:sonar|visual|bird|fad|light |  | no | How the school was found |
| `start_time_utc` | datetime | UTC | yes | Net shooting starts |
| `end_time_utc` | datetime | UTC | yes | Net fully hauled |
| `start_lat` | float | deg | yes | Position at start of set |
| `start_lon` | float | deg | yes | Position at start of set |
| `end_lat` | float | deg | no | Position at end of set |
| `end_lon` | float | deg | no | Position at end of set |
| `search_time_h` | float | h | yes | Steaming + searching hours since previous set or departure (effort) |
| `light_hours` | float | h | no | Hours of light luring before the set (light sets) |
| `net_length_m` | float | m | yes | Net length used |
| `net_depth_m` | float | m | yes | Net depth used |
| `bottom_depth_m` | float | m | no | Echosounder bottom depth at set start |
| `water_temp_c` | float | degC | no | Hull sensor water temperature |
| `wind_beaufort` | int | Bft | no | Crew-observed wind force |
| `water_haul` | bool | 0/1 | yes | 1 = set made but nothing caught |
| `total_catch_kg` | float | kg | yes | All species, retained + discarded |
| `retained_kg` | float | kg | yes | Kept on board |
| `discard_kg` | float | kg | yes | Returned to sea |

## `logbook/catch.csv`

| column | type | unit | required | description |
|---|---|---|---|---|
| `event_id` | str |  | yes | Set |
| `species_code` | str |  | yes | Checklist code: EVERY checklist species appears for EVERY set |
| `catch_kg` | float | kg | no | Weight caught (target / discard groups); 0 if none |
| `count` | int | n | no | Individuals caught (ETP species); 0 if none |
| `juvenile_kg` | float | kg | no | Part of catch_kg below size at maturity |
| `discard_kg` | float | kg | no | Part of catch_kg returned to sea |

## `logbook/etp_interactions.csv`

| column | type | unit | required | description |
|---|---|---|---|---|
| `event_id` | str |  | yes | Set |
| `species_code` | str |  | yes | ETP species code |
| `individual_no` | int |  | yes | 1..n within the set |
| `condition` | enum:alive|injured|dead |  | yes | Condition when released / landed |
| `fate` | enum:released_alive|released_injured|dead_discarded |  | yes | What happened to the animal |
| `photo_ref` | str |  | no | Photo file name for species verification |

## `logbook/track.csv`

| column | type | unit | required | description |
|---|---|---|---|---|
| `vessel_id` | str |  | yes | Vessel |
| `trip_id` | str |  | no | Trip |
| `time_utc` | datetime | UTC | yes | Ping time |
| `lat` | float | deg | yes | GPS latitude |
| `lon` | float | deg | yes | GPS longitude |
| `speed_kn` | float | kn | yes | Speed over ground |
| `course_deg` | float | deg | no | Course over ground |
| `true_activity` | str |  | no | SIMULATOR ONLY (steaming/searching/drifting/light_luring/setting); real AIS/VMS has no such field |

## `env/env_at_sets.csv`

| column | type | unit | required | description |
|---|---|---|---|---|
| `event_id` | str |  | yes | Set |
| `time_utc` | datetime | UTC | yes | Set start |
| `lat` | float | deg | yes | Set start latitude |
| `lon` | float | deg | yes | Set start longitude |
| `month` | int |  | yes | Calendar month |
| `sat_sst_c` | float | degC | yes | Satellite SST (e.g. OISST / MUR) |
| `sat_sst_anom_c` | float | degC | yes | SST anomaly vs. climatology |
| `sat_chl_mg_m3` | float | mg/m3 | no | Satellite chlorophyll-a; empty when cloudy |
| `sst_front_c_per_10km` | float | degC/10km | yes | SST gradient magnitude |
| `wind_ms` | float | m/s | yes | Reanalysis wind speed (ERA5) |
| `moon_illum` | float | 0-1 | yes | Moon illuminated fraction |
| `depth_m` | float | m | yes | Bathymetry (GEBCO) |
| `dist_coast_km` | float | km | yes | Distance to coast |
| `seagrass_idx` | float | 0-1 | yes | Proximity to seagrass/reef habitat |

## Species checklist

| code | group | unit | English | Thai | scientific |
|---|---|---|---|---|---|
| `RAST_BRA` | target | kg | Indo-Pacific mackerel | ปลาทู | *Rastrelliger brachysoma* |
| `RAST_KAN` | target | kg | Indian mackerel | ปลาลัง | *Rastrelliger kanagurta* |
| `SARD_GIB` | target | kg | Goldstripe sardinella | ปลาหลังเขียว | *Sardinella gibbosa* |
| `DECA_MAR` | target | kg | Japanese scad | ปลาทูแขก | *Decapterus maruadsi* |
| `THUN_TON` | target | kg | Longtail tuna | ปลาโอดำ | *Thunnus tonggol* |
| `EUTH_AFF` | target | kg | Kawakawa | ปลาโอลาย | *Euthynnus affinis* |
| `TRASH_MIX` | discard | kg | Mixed trash fish (fishmeal grade) | ปลาเป็ด | *mixed* |
| `RHIN_TYP` | etp | count | Whale shark | ฉลามวาฬ | *Rhincodon typus* |
| `MOBU_SPP` | etp | count | Devil rays (mobulids) | กระเบนราหู | *Mobula spp.* |
| `CARC_SPP` | etp | count | Requiem sharks (silky / blacktip) | ฉลาม (วงศ์ Carcharhinidae) | *Carcharhinidae* |
| `CHEL_MYD` | etp | count | Green turtle | เต่าตนุ | *Chelonia mydas* |
| `ORCA_BRE` | etp | count | Irrawaddy dolphin | โลมาอิรวดี | *Orcaella brevirostris* |
