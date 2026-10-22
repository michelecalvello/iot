#!/usr/bin/env python3
"""Importa nel portale i dati dei pluviometri della rete dpcn-campania scaricati da MeteoHub
(Agenzia ItaliaMeteo) in formato BUFR (un messaggio per stazione e per minuto).

Uso:  python3 tools/import_bufr_dpc.py FILE.bufr [--repo .]

Cosa fa
  * decodifica i messaggi (layout dei descrittori locali ricostruito a mano, vedi LAYOUT);
  * porta i tempi da UTC a UTC+1 (convenzione del portale) e mantiene il timestamp come fine del minuto
    di misura (verificato: sommando a 10 min si ottiene esattamente il dato CF di Pozzuoli);
  * scrive data/raw/<serie>/<anno>.csv (1 min), data/agg/*_1h.csv e *_1d.csv e le serie cumulate;
  * aggiorna catalog.json (stazioni e serie).
Lo script è idempotente: rilanciarlo rigenera gli stessi file e sostituisce le voci del catalogo.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from rainlib import CUM_HOURS, aggregate, regular_grid, rolling_cumulative, write_agg, write_raw

MSG_LEN = 152
# Descrittori attesi in sezione 3 (001194 e 007192/004192-4194 sono locali, non nelle tabelle WMO)
EXPECTED_DESC = [1194, 4001, 4002, 4003, 4004, 4005, 4006, 1019, 5001, 6001, 7030, 7192, 4192, 4193, 4194, 13011]
# Posizione (bit, dall'inizio dei dati della sezione 4) dei campi usati
LAYOUT = dict(
    network=(0, 128),        # 001194 stringa 16 caratteri
    year=(128, 12), month=(140, 4), day=(144, 6), hour=(150, 5), minute=(155, 6), second=(161, 6),
    name=(167, 256),         # 001019 stringa 32 caratteri
    lat=(423, 25),           # 005001: (raw/1e5) - 90
    lon=(448, 26),           # 006001: (raw/1e5) - 180
    height=(474, 17),        # 007030: (raw - 4000)/10 m
    rain=(573, 14),          # campo pioggia: (raw - 1)/10 mm (13011, ref -1, scala 1); individuato dai dati
)

# stazione nel BUFR -> (station_id, nome, id serie 1 min, etichetta, già presente nel catalogo?)
STATIONS = {
    "Pozzuoli":           dict(id="PZZ_CF", rain_id="PZZ_CF_RAIN1MIN", existing=True),
    "Nisida METEO":       dict(id="NIS_CF", name="Nisida – Centro Funzionale Campania", short="Nisida", code="Nisida METEO"),
    "Lago Patria METEO":  dict(id="LPA_CF", name="Lago Patria – Centro Funzionale Campania", short="Lago Patria", code="Lago Patria METEO"),
    "Napoli Camaldoli":   dict(id="CAM_CF", name="Napoli Camaldoli – Centro Funzionale Campania", short="Napoli Camaldoli", code="Napoli Camaldoli"),
    "Napoli Capodimonte": dict(id="CAP_CF", name="Napoli Capodimonte – Centro Funzionale Campania", short="Napoli Capodimonte", code="Napoli Capodimonte"),
}
SPIKE_MM_PER_MIN = 10.0   # >600 mm/h in un minuto: fisicamente non plausibile -> flag 2 (escluso)
RAW_MAX_DAYS_1MIN = 14


def bits(a: np.ndarray, pos: int, width: int) -> np.ndarray:
    b0, sh = divmod(pos, 8)
    w = np.zeros(len(a), dtype=np.uint64)
    for k in range(5):
        w = (w << np.uint64(8)) | a[:, b0 + k].astype(np.uint64)
    return ((w >> np.uint64(40 - sh - width)) & np.uint64((1 << width) - 1)).astype(np.int64)


def decode(path: Path) -> pd.DataFrame:
    buf = np.fromfile(path, dtype=np.uint8)
    if len(buf) % MSG_LEN:
        sys.exit(f"Dimensione file non multipla di {MSG_LEN} byte: i messaggi non hanno lunghezza fissa")
    m = buf.reshape(-1, MSG_LEN)
    assert (m[:, :4] == np.frombuffer(b"BUFR", np.uint8)).all() and (m[:, -4:] == np.frombuffer(b"7777", np.uint8)).all()
    # sezioni: 0 (8 byte) + 1 (lunghezza in byte 8-10) + 3 (descrittori) + 4
    s1 = int.from_bytes(bytes(m[0, 8:11]), "big")
    s3 = 8 + s1
    s3len = int.from_bytes(bytes(m[0, s3:s3 + 3]), "big")
    desc = [((int(m[0, s3 + 7 + 2 * i]) & 0x3F) * 1000 + int(m[0, s3 + 8 + 2 * i])) for i in range((s3len - 7) // 2)]
    if desc != EXPECTED_DESC:
        sys.exit(f"Descrittori inattesi: {desc}")
    s4 = s3 + s3len
    d = np.pad(m[:, s4 + 4:-4], ((0, 0), (0, 8)))   # padding per le letture a 5 byte in fondo
    assert (d[:, :16] == d[0, :16]).all() and bytes(d[0, :16]) == b"dpcn-campania   ", "rete inattesa"
    f = {k: bits(d, *v) for k, v in LAYOUT.items() if k not in ("network", "name")}
    name = np.stack([bits(d, LAYOUT["name"][0] + 8 * j, 8) for j in range(32)], axis=1).astype(np.uint8)
    names = np.array([bytes(r).decode("ascii").strip() for r in np.unique(name, axis=0)])
    inv = np.unique(name, axis=0, return_inverse=True)[1].ravel()
    t = pd.to_datetime(pd.DataFrame({k: f[k] for k in ("year", "month", "day", "hour", "minute", "second")}))
    return pd.DataFrame({
        "station": names[inv], "t_utc": t, "rain": (f["rain"] - 1) / 10.0,
        "lat": f["lat"] / 1e5 - 90, "lon": f["lon"] / 1e5 - 180, "height": (f["height"] - 4000) / 10.0,
    })


def num_stats(s: pd.Series):
    return round(float(s.min()), 1), round(float(s.max()), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bufr", type=Path)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    a = ap.parse_args()
    root = a.repo
    cat_path = root / "catalog.json"
    cat = json.loads(cat_path.read_text(encoding="utf-8"))

    df = decode(a.bufr)
    print(f"{len(df):,} record; stazioni: {sorted(df.station.unique())}")
    unknown = set(df.station.unique()) - set(STATIONS)
    if unknown:
        sys.exit(f"Stazioni non previste: {unknown}")

    # UTC -> UTC+1; il timestamp BUFR è la fine del minuto di misura (verificato: sommando a 10 min
    # si ottengono esattamente i valori della serie CF di Pozzuoli, 0 differenze su 41.715 intervalli)
    df["t"] = df["t_utc"] + pd.Timedelta(hours=1)
    new_ids = {s["id"] for s in STATIONS.values() if not s.get("existing")}
    new_series_prefix = tuple(f"{i}_" for i in new_ids) + ("PZZ_CF_RAIN1MIN",)
    cat["series"] = [s for s in cat["series"] if not s["series_id"].startswith(new_series_prefix)]
    cat["stations"] = [s for s in cat["stations"] if s["station_id"] not in new_ids]
    src_name = a.bufr.name
    new_series, new_stations, report = [], [], []

    for bufr_name, cfg in STATIONS.items():
        g = df[df.station == bufr_name].sort_values("t")
        dup = g.duplicated("t").sum()
        g = g.drop_duplicates("t", keep="first")
        lat, lon, h = g.lat.iloc[0], g.lon.iloc[0], g.height.iloc[0]
        rain = g.set_index("t")["rain"]
        bad = rain > SPIKE_MM_PER_MIN
        flag = bad.astype(int) * 2
        valid = rain.where(~bad)                     # per aggregati e cumulate: i dati errati non contano
        step = 1
        sid_rain = cfg.get("rain_id", f"{cfg['id']}_RAIN")
        grid = regular_grid(valid, step)
        gaps = grid.isna()
        # lacune > 24 h
        gl = []
        if gaps.any():
            run = (gaps != gaps.shift()).cumsum()
            for _, seg in grid[gaps].groupby(run[gaps]):
                if len(seg) > 24 * 60:
                    gl.append({"from": seg.index[0].strftime("%Y-%m-%dT%H:%M+01:00"), "to": seg.index[-1].strftime("%Y-%m-%dT%H:%M+01:00")})
        raw = pd.DataFrame({"value": rain, "flag": flag})
        years = write_raw(raw, sid_rain, root)
        vdf = pd.DataFrame({"value": valid.dropna()})
        write_agg(aggregate(vdf, "h", step, "sum"), sid_rain, "1h", root)
        write_agg(aggregate(vdf, "D", step, "sum"), sid_rain, "1d", root)
        t0, t1 = rain.index[0], rain.index[-1]
        iso = lambda t: t.strftime("%Y-%m-%dT%H:%M+01:00")
        vmin, vmax = num_stats(valid.dropna())
        tot = round(float(valid.sum()), 1)
        notes = [
            "Dati originali al minuto, rete dpcn-campania (Protezione Civile), scaricati da MeteoHub (Agenzia ItaliaMeteo) in formato BUFR.",
            "Orari convertiti da UTC a UTC+1; il valore al timestamp t è la pioggia caduta nel minuto che termina a t. Risoluzione tipica 0,2 mm (pochi valori con passo 0,1 mm).",
            f"Totale registrato: {tot:.1f} mm ({t0:%d/%m/%Y} – {t1:%d/%m/%Y}).",
        ]
        if bad.any():
            for t_, v_ in rain[bad].items():
                notes.append(f"Dato escluso (flag 2): {v_:.1f} mm in 1 minuto il {t_:%d/%m/%Y alle %H:%M}, isolato in assenza di pioggia: non plausibile (valore mantenuto nel CSV ma escluso da grafici, aggregati e cumulate).")
        if cfg.get("existing"):
            notes.append("Stessa stazione della serie PZZ_CF_RAIN (10 min): sommando a 10 min i valori coincidono con quelli di PZZ_CF_RAIN; fa eccezione solo il picco anomalo di 23,2 mm, che in PZZ_CF_RAIN compare il 09/09/2026 alle 00:00.")
        label = f"Pioggia 1 min – Pozzuoli (CF)" if cfg.get("existing") else f"Pioggia – {cfg['short']} (CF)"
        base = dict(
            station_id=cfg["id"], variable="rain", label=label, unit="mm", aggregation="sum", step_min=1,
            start=iso(t0), end=iso(t1), n_values=int(len(rain)), min=vmin, max=vmax, status="active",
            lat=(lat if not cfg.get("existing") else next(s for s in cat["stations"] if s["station_id"] == cfg["id"])["lat"]),
            lon=(lon if not cfg.get("existing") else next(s for s in cat["stations"] if s["station_id"] == cfg["id"])["lon"]),
            depth_m=None, sensor_model=None, logger_port=None,
        )
        s = dict(series_id=sid_rain, **base)
        s.update(
            short_label="Pioggia 1 min" if cfg.get("existing") else "Pioggia",
            raw_max_days=RAW_MAX_DAYS_1MIN,
            qc_notes=notes, sources=[{"file": src_name, "sheet": "–", "column": "13011 (precipitazione totale)"}],
            qc_summary={"n_total": int(len(rain)), "n_suspect": 0, "n_bad": int(bad.sum())},
            gaps_over_24h=gl,
            data={"years": years, "raw": f"data/raw/{sid_rain}/{{year}}.csv", "agg_1h": f"data/agg/{sid_rain}_1h.csv", "agg_1d": f"data/agg/{sid_rain}_1d.csv"},
        )
        new_series.append(s)
        n_cum = 0
        if not cfg.get("existing"):
            new_stations.append(dict(
                station_id=cfg["id"], name=cfg["name"], lat=round(lat, 8), lon=round(lon, 8), elevation_m=round(float(h)),
                owner="Centro Funzionale Multirischi della Protezione Civile, Regione Campania (rete dpcn-campania, via MeteoHub)",
                external_code=cfg["code"],
                notes="Stazione pluviometrica della rete dpcn-campania. Dati al minuto (2026). Quota dai metadati BUFR, da verificare.",
            ))
            for hrs in CUM_HOURS:
                cid = f"{cfg['id']}_RAIN{hrs}H"
                cum = rolling_cumulative(grid, step, hrs, marks_every_min=10)
                cyears = write_raw(cum, cid, root)
                write_agg(aggregate(cum, "h", 10, "last"), cid, "1h", root)
                write_agg(aggregate(cum, "D", 10, "last"), cid, "1d", root)
                nsusp = int(cum.flag.sum())
                c = dict(series_id=cid, **{k: v for k, v in base.items()})
                c.update(
                    variable=f"rain_{hrs}h", label=f"Pioggia cumulata {hrs} h – {cfg['short']} (CF)", aggregation="last", step_min=10,
                    start=iso(cum.index[0]), end=iso(cum.index[-1]), n_values=int(len(cum)),
                    min=round(float(cum.value.min()), 1), max=round(float(cum.value.max()), 1),
                    qc_notes=[
                        f"Serie derivata da {sid_rain}: pioggia caduta nelle {hrs} ore che terminano al timestamp (finestra mobile, intervallo (t-{hrs} h, t]), calcolata ogni 10 minuti dai dati al minuto.",
                        f"Massimo {hrs} h: {cum.value.max():.1f} mm.",
                        f"flag 1 = copertura <90% della finestra: valore sottostimato. {nsusp:,} dati segnalati.".replace(",", "."),
                    ],
                    qc_summary={"n_total": int(len(cum)), "n_suspect": nsusp, "n_bad": 0},
                    derived_from=sid_rain,
                    data={"years": cyears, "raw": f"data/raw/{cid}/{{year}}.csv", "agg_1h": f"data/agg/{cid}_1h.csv", "agg_1d": f"data/agg/{cid}_1d.csv"},
                )
                c.pop("short_label", None); c.pop("raw_max_days", None)
                new_series.append(c); n_cum += 1
        report.append((bufr_name, len(rain), dup, int(bad.sum()), tot, iso(t0), iso(t1), len(gl), int(gaps.sum()), n_cum, round(float(h)), round(lat, 5), round(lon, 5)))

    cat["stations"] += new_stations
    cat["series"] += new_series
    cat["catalog_version"] = "0.6"
    cat["generated"] = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S+01:00")
    cat_path.write_text(json.dumps(cat, indent=2, ensure_ascii=False), encoding="utf-8")
    print("stazione | n | duplicati | errati | totale mm | da | a | lacune>24h | minuti mancanti | cumulate | quota | lat | lon")
    for r in report:
        print(" | ".join(map(str, r)))


if __name__ == "__main__":
    main()
