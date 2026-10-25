"""Vessel and skipper behaviour: trips, FADs, set planning, location choice, movement.

Each vessel loops: port -> trip (days of sets) -> port. Every planned set picks a
legal spot by weighing the skipper's (noisy) knowledge of target density, price
and steaming distance, then travels, searches and sets the net. Movement is
recorded as time-stamped segments, later turned into AIS/VMS-like pings.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .catch import FATES, report, simulate_catch
from .environment import (local_day_of, local_hour, local_time, month_day, moon_illumination,
                          satellite_view, to_unix_days, unix_days_to_iso, year_of)
from .geo import distance_km, offset_point
from .species import gear_multiplier

KN_TO_KMH = 1.852
BEAUFORT_MS = [0.3, 1.6, 3.4, 5.5, 8.0, 10.8, 13.9, 17.2, 20.8, 24.5, 28.5, 32.7]


@dataclass
class Vessel:
    vessel_id: str
    home_port: str
    port_lat: float
    port_lon: float
    gross_tonnage: float
    length_m: float
    net_length_m: float
    net_depth_m: float
    hold_capacity_kg: float
    net_capacity_kg: float
    uses_light: bool
    complies_closures: bool
    skipper_noise_sd: float
    fads: list = field(default_factory=list)    # (fad_id, lat, lon)
    fad_year: int = -1


@dataclass
class _TripState:
    lat: float
    lon: float
    clock: float
    segments: list = field(default_factory=list)
    effort_h: float = 0.0
    hold_kg: float = 0.0
    landed_kg: float = 0.0
    n_sets: int = 0


def _closure_active(closure, t):
    md = month_day(t)
    s = tuple(int(v) for v in closure["start_mmdd"].split("-"))
    e = tuple(int(v) for v in closure["end_mmdd"].split("-"))
    return s <= md <= e if s <= e else (md >= s or md <= e)


def _beaufort(ms):
    return int(np.searchsorted(BEAUFORT_MS, ms, side="right"))


class FleetSimulator:
    def __init__(self, cfg, geo, env, models, recorder, rng):
        self.cfg = cfg
        self.geo = geo
        self.env = env
        self.models = models
        self.targets = [m for m in models if m.sp.group == "target"]
        self.etps = [m for m in models if m.sp.group == "etp"]
        self.rec = recorder
        self.rng = rng
        self.t_start = to_unix_days(cfg.start)
        self.t_end = to_unix_days(cfg.end) + 1.0
        self.ports = []
        for p in cfg.fleet.ports:
            lat, lon = geo.nearest_ocean(p["lat"], p["lon"])
            self.ports.append((p["name"], lat, lon))

    # =========================
    # FLEET SETUP
    # =========================

    def make_fleet(self):
        fc, rng = self.cfg.fleet, self.rng
        vessels = []
        for k in range(fc.n_vessels):
            name, lat, lon = self.ports[rng.integers(len(self.ports))]
            v = Vessel(
                vessel_id=f"V{k + 1:02d}", home_port=name, port_lat=lat, port_lon=lon,
                gross_tonnage=round(rng.uniform(*fc.gross_tonnage_range)),
                length_m=round(rng.uniform(*fc.length_m_range), 1),
                net_length_m=round(rng.uniform(*fc.net_length_m_range), -1),
                net_depth_m=round(rng.uniform(*fc.net_depth_m_range), -1),
                hold_capacity_kg=round(rng.uniform(*fc.hold_capacity_t_range)) * 1000.0,
                net_capacity_kg=fc.net_capacity_t * 1000.0,
                uses_light=bool(rng.random() < fc.light_fraction),
                complies_closures=bool(rng.random() < self.cfg.regulation.compliance_prob),
                skipper_noise_sd=fc.skipper_noise_sd * float(np.exp(rng.normal(0.0, 0.2))),
            )
            vessels.append(v)
            self.rec.write("logbook/vessels.csv", {
                "vessel_id": v.vessel_id, "home_port": v.home_port, "gross_tonnage": v.gross_tonnage,
                "length_m": v.length_m, "net_length_m": v.net_length_m, "net_depth_m": v.net_depth_m,
                "hold_capacity_kg": v.hold_capacity_kg, "uses_light": v.uses_light})
        return vessels

    def _ensure_fads(self, v, t):
        year = year_of(t)
        if v.fad_year == year:
            return
        fc, reg, rng = self.cfg.fleet, self.cfg.regulation, self.rng
        n = int(rng.integers(fc.fads_per_vessel_range[0], fc.fads_per_vessel_range[1] + 1))
        fads = []
        for _ in range(20):
            lat, lon = self._disk(v.port_lat, v.port_lon, fc.fad_max_dist_port_km, 400)
            depth = self.geo.lookup("depth_m", lat, lon)
            ok = (self.geo.is_ocean(lat, lon)
                  & (self.geo.lookup("dist_coast_km", lat, lon) >= reg.min_dist_coast_km)
                  & (depth >= fc.fad_depth_range_m[0]) & (depth <= fc.fad_depth_range_m[1]))
            for la, lo in zip(lat[ok], lon[ok]):
                fads.append((f"{v.vessel_id}-FAD{year}-{len(fads) + 1:02d}", float(la), float(lo)))
                if len(fads) >= n:
                    break
            if len(fads) >= n:
                break
        v.fads, v.fad_year = fads, year
        for fad_id, la, lo in fads:
            self.rec.write("logbook/fads.csv", {"fad_id": fad_id, "vessel_id": v.vessel_id, "year": year,
                                                "lat": round(la, 5), "lon": round(lo, 5)})

    # =========================
    # MAIN LOOP
    # =========================

    def run(self, vessels):
        for v in vessels:
            self._run_vessel(v)

    def _next_local_hour(self, t, hour):
        cand = local_time(local_day_of(t), hour)
        return cand if cand >= t else cand + 1.0

    def _run_vessel(self, v):
        fc, rng = self.cfg.fleet, self.rng
        t = self._next_local_hour(self.t_start, 4.0)
        trip_no = 0
        while t < self.t_end - 1.0:
            if self.env.wind(v.port_lat, v.port_lon, t) > fc.max_departure_wind_ms:
                t += 1.0
                continue
            self._ensure_fads(v, t)
            trip_no += 1
            t = self._run_trip(v, f"{v.vessel_id}-T{trip_no:03d}", t)
            stay = int(rng.integers(fc.port_days_range[0], fc.port_days_range[1] + 1))
            t = self._next_local_hour(t + stay, 4.0)

    def _run_trip(self, v, trip_id, t_depart):
        fc, rng = self.cfg.fleet, self.rng
        st = _TripState(lat=v.port_lat, lon=v.port_lon, clock=t_depart)
        target_end = t_depart + int(rng.integers(fc.trip_days_range[0], fc.trip_days_range[1] + 1))
        day = local_day_of(t_depart)
        full = False
        while not full:
            morning = local_time(day, 6.0)
            if morning >= target_end or morning >= self.t_end:
                break
            for plan in self._plan_day(v, day):
                if st.hold_kg >= fc.hold_fill_return * v.hold_capacity_kg:
                    full = True
                    break
                self._attempt_set(v, trip_id, st, plan, target_end)
            day += 1
            next_morning = min(local_time(day, 5.0), target_end)
            if not full and st.clock < next_morning:
                self._drift(st, next_morning)

        self._travel(st, v.port_lat, v.port_lon, fc.steam_speed_kn, "steaming")
        self.rec.write("logbook/trips.csv", {
            "trip_id": trip_id, "vessel_id": v.vessel_id,
            "departure_port": v.home_port, "departure_time_utc": unix_days_to_iso(t_depart).item(),
            "return_port": v.home_port, "return_time_utc": unix_days_to_iso(st.clock).item(),
            "n_sets": st.n_sets, "landed_kg": round(st.landed_kg, 1)})
        self.rec.n_trips += 1
        self.rec.track(v.vessel_id, trip_id, st.segments)
        return st.clock

    def _plan_day(self, v, day):
        fc, rng = self.cfg.fleet, self.rng
        plans = []
        n = min(fc.max_day_sets, int(rng.poisson(fc.day_sets_mean)))
        starts = np.sort(rng.uniform(6.5, 15.5, n))
        for i in range(1, n):
            starts[i] = max(starts[i], starts[i - 1] + 2.8)
        for s in starts[starts <= 16.0]:
            kind = "fad" if (v.fads and rng.random() < fc.fad_set_prob) else "free_school"
            plans.append((kind, local_time(day, float(s)), 0.0))
        if v.uses_light and moon_illumination(local_time(day, 23.0)) < fc.light_max_moon:
            plans.append(("light", local_time(day, rng.uniform(19.0, 20.5)), rng.uniform(*fc.light_hours_range)))
        return plans

    # =========================
    # ONE SET
    # =========================

    def _attempt_set(self, v, trip_id, st, plan, target_end):
        fc, rng = self.cfg.fleet, self.rng
        set_type, ws, light_h = plan
        if ws > target_end or ws >= self.t_end or st.clock > ws + 2.0 / 24.0:
            return
        choice = self._choose_location(v, st, set_type, ws)
        if choice is None:
            return
        lat, lon, set_type, fad_id = choice

        far = float(distance_km(st.lat, st.lon, lat, lon)) > 15.0
        self._travel(st, lat, lon, fc.steam_speed_kn if far else fc.search_speed_kn,
                     "steaming" if far else "searching")
        if set_type == "light":
            if st.clock < ws:
                self._drift(st, ws)
            self._drift(st, st.clock + light_h / 24.0, activity="light_luring")
            detection = "light"
        else:
            search_h = rng.uniform(0.2, 0.6) if set_type == "fad" else rng.gamma(2.0, fc.search_hours_mean / 2.0)
            if st.clock < ws - search_h / 24.0:
                self._drift(st, ws - search_h / 24.0)
            self._search(st, lat, lon, search_h)
            if local_hour(st.clock) > 17.5 or local_hour(st.clock) < 5.5:
                return                                   # too dark for a day set
            detection = "fad" if set_type == "fad" else str(rng.choice(["sonar", "visual", "bird"], p=[0.6, 0.3, 0.1]))

        t_set = st.clock
        set_lat, set_lon = st.lat, st.lon
        env = self.env.sample(set_lat, set_lon, t_set)
        if env["wind"][0] > fc.max_fishing_wind_ms:
            return                                       # too rough to set the net
        env_pt = {k: float(v_[0]) for k, v_ in env.items()}
        lam = {m.sp.code: float(m.density(env)[0]) for m in self.models}
        c = simulate_catch(rng, self.targets, self.etps, lam, set_type, env_pt, v, self.cfg)

        duration_h = float(np.clip(rng.normal(*fc.set_duration_h), 1.2, 4.0))
        elat, elon = offset_point(set_lat, set_lon, rng.uniform(0.3, 1.2), rng.uniform(0, 360))
        if not self.geo.is_ocean(elat, elon):
            elat, elon = set_lat, set_lon
        search_time_h = st.effort_h
        self._leg_timed(st, float(elat), float(elon), duration_h / 24.0, "setting")
        st.effort_h = 0.0
        st.n_sets += 1
        event_id = f"{trip_id}-S{st.n_sets:02d}"
        self._record_set(v, trip_id, event_id, set_type, fad_id, detection, t_set, st.clock,
                         set_lat, set_lon, float(elat), float(elon), search_time_h,
                         light_h if set_type == "light" else 0.0, env, env_pt, lam, c, st)

    def _choose_location(self, v, st, set_type, t):
        fc, rng = self.cfg.fleet, self.rng
        fad_ids = None
        lat = lon = None
        if set_type == "fad":
            if v.fads:
                f = np.array([(la, lo) for _, la, lo in v.fads])
                ok = (distance_km(st.lat, st.lon, f[:, 0], f[:, 1]) <= fc.search_radius_km * 1.5) & self._legal(f[:, 0], f[:, 1], t, v)
                if ok.any():
                    lat, lon = f[ok, 0], f[ok, 1]
                    fad_ids = [v.fads[i][0] for i in np.flatnonzero(ok)]
            if fad_ids is None:
                set_type = "free_school"
        if fad_ids is None:
            radius = fc.search_radius_km
            for _ in range(3):
                clat, clon = self._disk(st.lat, st.lon, radius, fc.n_candidates)
                ok = self._legal(clat, clon, t, v)
                if ok.sum() >= 5:
                    lat, lon = clat[ok], clon[ok]
                    break
                radius *= 2.0
            if lat is None:
                return None

        env = self.env.sample(lat, lon, t)
        lam = {m.sp.code: m.density(env) for m in self.models}
        value = sum(m.sp.price_thb_kg * gear_multiplier(m.sp, set_type, env["moon"]) * lam[m.sp.code]
                    for m in self.targets)
        logv = (np.log(value + 1e-9) + rng.normal(0.0, v.skipper_noise_sd, value.shape)
                - distance_km(st.lat, st.lon, lat, lon) / fc.distance_cost_km)
        if fc.bycatch_avoidance > 0:
            risk = sum(gear_multiplier(m.sp, set_type, env["moon"]) * lam[m.sp.code] for m in self.etps)
            logv = logv - fc.bycatch_avoidance * risk
        if rng.random() < fc.explore_prob:
            k = int(rng.integers(lat.size))
        else:
            z = (logv - logv.max()) / fc.choice_temperature
            p = np.exp(z)
            k = int(rng.choice(lat.size, p=p / p.sum()))
        return float(lat[k]), float(lon[k]), set_type, (fad_ids[k] if fad_ids else "")

    def _legal(self, lat, lon, t, v):
        reg, geo = self.cfg.regulation, self.geo
        ok = (geo.is_ocean(lat, lon)
              & (geo.lookup("dist_coast_km", lat, lon) >= reg.min_dist_coast_km)
              & (np.nan_to_num(geo.lookup("depth_m", lat, lon)) >= reg.min_set_depth_m))
        if v.complies_closures:
            for c in reg.closures:
                if _closure_active(c, t):
                    ok &= ~((lat >= c["lat_min"]) & (lat <= c["lat_max"]) & (lon >= c["lon_min"]) & (lon <= c["lon_max"]))
        return ok

    def _disk(self, lat, lon, radius_km, n):
        rng = self.rng
        r = radius_km * np.sqrt(rng.random(n))
        return offset_point(lat, lon, r, rng.uniform(0.0, 360.0, n))

    # =========================
    # MOVEMENT
    # =========================

    def _leg_timed(self, st, lat1, lon1, dt_days, activity):
        st.segments.append((st.clock, st.clock + dt_days, st.lat, st.lon, lat1, lon1, activity))
        st.clock += dt_days
        st.lat, st.lon = lat1, lon1
        if activity in ("steaming", "searching"):
            st.effort_h += dt_days * 24.0

    def _travel(self, st, lat1, lon1, speed_kn, activity):
        for la, lo in self.geo.route(st.lat, st.lon, lat1, lon1)[1:]:
            d = float(distance_km(st.lat, st.lon, la, lo))
            self._leg_timed(st, float(la), float(lo), d / (speed_kn * KN_TO_KMH) / 24.0, activity)

    def _drift(self, st, until, activity="drifting"):
        if until <= st.clock:
            return
        hours = (until - st.clock) * 24.0
        speed = 0.3 if activity == "drifting" else 0.2
        lat1, lon1 = offset_point(st.lat, st.lon, min(hours * speed * KN_TO_KMH, 8.0), self.rng.uniform(0, 360))
        if not self.geo.is_ocean(lat1, lon1):
            lat1, lon1 = st.lat, st.lon
        self._leg_timed(st, float(lat1), float(lon1), until - st.clock, activity)

    def _search(self, st, lat, lon, hours):
        rng = self.rng
        k = int(rng.integers(2, 5))
        for i in range(k):
            if i == k - 1:
                la, lo = lat, lon
            else:
                la, lo = offset_point(lat, lon, rng.uniform(0.5, 4.0), rng.uniform(0, 360))
                if not self.geo.is_ocean(la, lo):
                    la, lo = lat, lon
            self._leg_timed(st, float(la), float(lo), hours / k / 24.0, "searching")

    # =========================
    # RECORDING
    # =========================

    def _record_set(self, v, trip_id, event_id, set_type, fad_id, detection, t0, t1,
                    lat0, lon0, lat1, lon1, search_h, light_h, env, env_pt, lam, c, st):
        rng, rec = self.rng, self.rec
        kg, juv, disc, counts, individuals = report(c, rng, self.cfg.reporting)
        total = sum(kg.values())
        discard = sum(disc.values())
        st.hold_kg += c.total_kg - c.discard_total_kg
        st.landed_kg += total - discard

        rec.write("logbook/sets.csv", {
            "event_id": event_id, "trip_id": trip_id, "vessel_id": v.vessel_id,
            "set_type": set_type, "fad_id": fad_id, "school_detection": detection,
            "start_time_utc": unix_days_to_iso(t0).item(), "end_time_utc": unix_days_to_iso(t1).item(),
            "start_lat": round(lat0, 5), "start_lon": round(lon0, 5),
            "end_lat": round(lat1, 5), "end_lon": round(lon1, 5),
            "search_time_h": round(search_h, 2), "light_hours": round(light_h, 2),
            "net_length_m": v.net_length_m, "net_depth_m": v.net_depth_m,
            "bottom_depth_m": round(max(env_pt["depth"] + rng.normal(0, 1.0), 0.5), 1),
            "water_temp_c": round(env_pt["sst"] + rng.normal(0, 0.2), 2),
            "wind_beaufort": _beaufort(max(env_pt["wind"] + rng.normal(0, 1.0), 0.0)),
            "water_haul": c.water_haul,
            "total_catch_kg": round(total, 1), "retained_kg": round(total - discard, 1),
            "discard_kg": round(discard, 1)})

        for code in kg:
            rec.write("logbook/catch.csv", {"event_id": event_id, "species_code": code, "catch_kg": kg[code],
                                            "juvenile_kg": juv[code], "discard_kg": disc[code]})
            rec.write("truth/truth_catch.csv", {"event_id": event_id, "species_code": code,
                                                "catch_kg": round(c.kg[code], 2),
                                                "juvenile_kg": round(c.juvenile_kg[code], 2)})
        for code in counts:
            rec.write("logbook/catch.csv", {"event_id": event_id, "species_code": code, "count": counts[code]})
            rec.write("truth/truth_catch.csv", {"event_id": event_id, "species_code": code, "count": c.counts[code]})
        seen = {}
        for code, cond in individuals:
            seen[code] = seen.get(code, 0) + 1
            rec.write("logbook/etp_interactions.csv", {
                "event_id": event_id, "species_code": code, "individual_no": seen[code],
                "condition": cond, "fate": FATES[cond], "photo_ref": f"{event_id}-{code}-{seen[code]}.jpg"})

        sat = satellite_view(env, rng, self.cfg.env)
        row = {"event_id": event_id, "time_utc": unix_days_to_iso(t0).item(),
               "lat": round(lat0, 5), "lon": round(lon0, 5), "month": month_day(t0)[0]}
        for k, val in sat.items():
            row[k] = round(float(val[0]), 4)
        rec.write("env/env_at_sets.csv", row)

        truth = {"event_id": event_id, "set_type": set_type, "p_water_haul": round(c.p_water_haul, 4),
                 "true_sst_c": round(env_pt["sst"], 3), "true_sst_anom_c": round(env_pt["sst_anom"], 3),
                 "true_log10chl": round(env_pt["log10chl"], 3), "true_wind_ms": round(env_pt["wind"], 2),
                 "true_front": round(env_pt["front"], 4), "true_depth_m": round(env_pt["depth"], 1)}
        for code, val in lam.items():
            truth[f"lambda_{code}"] = round(val, 5)
        for code, val in c.p_present.items():
            truth[f"p_present_{code}"] = round(val, 5)
        rec.write("truth/truth_sets.csv", truth)

        true_present = {k: x > 0 for k, x in c.kg.items()} | {k: x > 0 for k, x in c.counts.items()}
        rep_present = {k: x > 0 for k, x in kg.items()} | {k: x > 0 for k, x in counts.items()}
        rec.note_set(lat0, lon0, set_type, true_present, rep_present)
