#!/usr/bin/env python3
"""Importa nel portale i pluviometri orari della rete 'mnw' (MeteoHub, Agenzia ItaliaMeteo), formato BUFR
identico a quello della rete dpcn-campania (vedi import_bufr_dpc.py per il layout dei campi).

Uso:  python3 tools/import_bufr_mnw.py FILE.bufr [--repo .]

  * dati orari; il timestamp BUFR (UTC) è la fine dell'ora di misura (verificato contro i pluviometri CF vicini:
    correlazione oraria 0,74–0,89 con scarto 0, ≤0,4 con scarto ±1 h). Convertito in UTC+1 (+1 h).
  * scrive data/raw/<serie>/<anno>.csv, data/agg/*_1h.csv e *_1d.csv, cumulate 3-72 h e aggiorna catalog.json.
  * le ore mancanti restano mancanti (nessuna interpolazione); le cumulate con copertura <90% hanno flag 1.
"""
import argparse, json, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from import_bufr_dpc import decode
from rainlib import aggregate, regular_grid, rolling_cumulative, write_agg, write_raw

NETWORK = b"mnw             "
CUM = (3, 6, 24, 48, 72)          # la cumulata 1 h coincide con il dato orario
MAX_MM_PER_H = 200.0              # soglia di plausibilità (flag 2)
# codice BUFR -> id stazione, quota (da metadati BUFR, da verificare), ordine nell'elenco
STATIONS = {
    "cmp080": dict(id="CMP080_MNW", elev=215, order=6),
    "cmp292": dict(id="CMP292_MNW", elev=50, order=7),
    "cmp304": dict(id="CMP304_MNW", elev=82, order=8),
    "cmp307": dict(id="CMP307_MNW", elev=90, order=9),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bufr", type=Path)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    a = ap.parse_args(); root = a.repo
    cat_path = root / "catalog.json"
    cat = json.loads(cat_path.read_text(encoding="utf-8"))
    df = decode(a.bufr, NETWORK)
    unknown = set(df.station.unique()) - set(STATIONS)
    if unknown:
        sys.exit(f"Stazioni non previste: {unknown}")
    df["t"] = df["t_utc"] + pd.Timedelta(hours=1)
    ids = {c["id"] for c in STATIONS.values()}
    cat["series"] = [s for s in cat["series"] if not s["series_id"].startswith(tuple(f"{i}_" for i in ids))]
    cat["stations"] = [s for s in cat["stations"] if s["station_id"] not in ids]
    iso = lambda t: t.strftime("%Y-%m-%dT%H:%M+01:00")
    src = a.bufr.name
    for code, cfg in STATIONS.items():
        g = df[df.station == code].sort_values("t").drop_duplicates("t")
        sid, lat, lon = cfg["id"], round(g.lat.iloc[0], 8), round(g.lon.iloc[0], 8)
        rain = g.set_index("t")["rain"]
        bad = rain > MAX_MM_PER_H
        valid = rain.where(~bad)
        grid = regular_grid(valid, 60)
        raw = pd.DataFrame({"value": rain, "flag": bad.astype(int) * 2})
        rid = f"{sid}_RAIN"
        years = write_raw(raw, rid, root)
        v = pd.DataFrame({"value": valid.dropna()})
        write_agg(aggregate(v, "h", 60, "sum"), rid, "1h", root)
        write_agg(aggregate(v, "D", 60, "sum"), rid, "1d", root)
        gaps = grid.isna()
        run = (gaps != gaps.shift()).cumsum()
        gl = [{"from": iso(x.index[0]), "to": iso(x.index[-1])} for _, x in grid[gaps].groupby(run[gaps]) if len(x) > 24]
        cat["stations"].append(dict(
            station_id=sid, name=f"Napoli {code} (mnw)", lat=lat, lon=lon, elevation_m=cfg["elev"],
            owner="Rete mnw, via MeteoHub (Agenzia ItaliaMeteo) – gestore da verificare",
            external_code=code, notes="Pluviometro della rete mnw (non Centro Funzionale). Dati orari (2026). Quota dai metadati BUFR.",
            order=cfg["order"], network="mnw"))
        base = dict(station_id=sid, unit="mm", status="active", lat=lat, lon=lon, depth_m=None, sensor_model=None, logger_port=None)
        miss = int(gaps.sum())
        cat["series"].append(dict(
            series_id=rid, variable="rain", label=f"Pioggia 1 h – Napoli {code} (mnw)", aggregation="sum", step_min=60,
            start=iso(rain.index[0]), end=iso(rain.index[-1]), n_values=int(len(rain)),
            min=round(float(valid.min()), 1), max=round(float(valid.max()), 1), **base, short_label="Pioggia 1 h",
            qc_notes=[
                "Dati orari rete mnw, scaricati da MeteoHub (Agenzia ItaliaMeteo) in formato BUFR.",
                "Orari convertiti da UTC a UTC+1; il valore al timestamp t è la pioggia caduta nell'ora che termina a t. Risoluzione 0,1 mm.",
                f"Totale registrato: {valid.sum():.1f} mm ({rain.index[0]:%d/%m/%Y} – {rain.index[-1]:%d/%m/%Y}).",
                f"Ore mancanti: {miss:,} su {len(grid):,} ({miss / len(grid):.0%}); non interpolate.".replace(",", "."),
            ],
            sources=[{"file": src, "sheet": "–", "column": "13011 (precipitazione totale)"}],
            qc_summary={"n_total": int(len(rain)), "n_suspect": 0, "n_bad": int(bad.sum())}, gaps_over_24h=gl,
            data={"years": years, "raw": f"data/raw/{rid}/{{year}}.csv", "agg_1h": f"data/agg/{rid}_1h.csv", "agg_1d": f"data/agg/{rid}_1d.csv"}))
        for hrs in CUM:
            cid = f"{sid}_RAIN{hrs}H"
            cum = rolling_cumulative(grid, 60, hrs)
            cy = write_raw(cum, cid, root)
            write_agg(aggregate(cum, "h", 60, "last"), cid, "1h", root)
            write_agg(aggregate(cum, "D", 60, "last"), cid, "1d", root)
            ns = int(cum.flag.sum())
            cat["series"].append(dict(
                series_id=cid, variable=f"rain_{hrs}h", label=f"Pioggia cumulata {hrs} h – Napoli {code} (mnw)", aggregation="last", step_min=60,
                start=iso(cum.index[0]), end=iso(cum.index[-1]), n_values=int(len(cum)),
                min=round(float(cum.value.min()), 1), max=round(float(cum.value.max()), 1), **base,
                qc_notes=[f"Serie derivata da {rid}: pioggia caduta nelle {hrs} ore che terminano al timestamp (finestra mobile, intervallo (t-{hrs} h, t]).",
                          f"Massimo {hrs} h: {cum.value.max():.1f} mm.",
                          f"flag 1 = copertura <90% della finestra: valore sottostimato. {ns:,} dati segnalati.".replace(",", ".")],
                qc_summary={"n_total": int(len(cum)), "n_suspect": ns, "n_bad": 0}, gaps_over_24h=gl, derived_from=rid,
                data={"years": cy, "raw": f"data/raw/{cid}/{{year}}.csv", "agg_1h": f"data/agg/{cid}_1h.csv", "agg_1d": f"data/agg/{cid}_1d.csv"}))
        print(code, sid, len(rain), f"tot {valid.sum():.1f} mm", f"mancanti {miss}", f"max {valid.max():.1f}", lat, lon, cfg["elev"])
    cat["generated"] = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S+01:00")
    cat_path.write_text(json.dumps(cat, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
