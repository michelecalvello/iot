"""Funzioni comuni per costruire le serie di pioggia del portale (raw, aggregati orari/giornalieri,
pioggia cumulata a finestra mobile) con le stesse convenzioni dei dati già presenti in data/."""
import numpy as np
import pandas as pd

CUM_HOURS = (1, 3, 6, 24, 48, 72)
TIME_FMT = "%Y-%m-%d %H:%M"


def regular_grid(s: pd.Series, step_min: int) -> pd.Series:
    """Serie (indice temporale) -> griglia regolare con NaN dove manca il dato."""
    idx = pd.date_range(s.index.min(), s.index.max(), freq=f"{step_min}min")
    return s.reindex(idx)


def rolling_cumulative(rain: pd.Series, step_min: int, hours: int, marks_every_min: int | None = None):
    """Pioggia caduta nelle `hours` ore che terminano al timestamp: intervallo (t-N h, t].
    `rain` ha indice regolare con passo `step_min`, NaN = dato mancante/escluso.
    flag=1 se i dati presenti coprono meno del 90% della finestra (valore sottostimato).
    Restituisce DataFrame time,value,flag nei soli istanti in cui il dato originale esiste
    (e, se `marks_every_min`, solo multipli di quel passo)."""
    w = hours * 60 // step_min
    present = rain.notna().to_numpy()
    cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(rain.to_numpy(float)))])
    cc = np.concatenate([[0], np.cumsum(present)])
    i = np.arange(len(rain))
    lo = np.maximum(i + 1 - w, 0)
    val = cs[i + 1] - cs[lo]
    cnt = cc[i + 1] - cc[lo]
    flag = (cnt < 0.9 * w).astype(int)
    out = pd.DataFrame({"value": np.round(val, 1), "flag": flag}, index=rain.index)
    keep = present.copy()
    if marks_every_min:
        keep &= (rain.index.minute % marks_every_min == 0) & (rain.index.second == 0)
    return out[keep]


def aggregate(df: pd.DataFrame, freq: str, step_min: int, how: str) -> pd.DataFrame:
    """Aggregazione oraria ('h') o giornaliera ('D'): etichetta = floor del timestamp,
    value = somma (pioggia) o ultimo valore (cumulata), max, n, coverage = n/attesi."""
    bins = df.index.floor(freq)
    g = df["value"].groupby(bins)
    value = g.sum() if how == "sum" else g.last()
    expected = (60 if freq == "h" else 1440) / step_min
    n = g.count()
    out = pd.DataFrame({"value": value.round(1), "max": g.max().round(1), "n": n,
                        "coverage": (n / expected).round(3)})
    return out[out["n"] > 0]


def write_raw(df: pd.DataFrame, series_id: str, root) -> list[int]:
    years = []
    for y, part in df.groupby(df.index.year):
        d = root / "data" / "raw" / series_id
        d.mkdir(parents=True, exist_ok=True)
        p = part.reset_index()
        p.columns = ["time", "value", "flag"]
        p["time"] = p["time"].dt.strftime(TIME_FMT)
        p.to_csv(d / f"{y}.csv", index=False, float_format="%.1f")
        years.append(int(y))
    return years


def write_agg(agg: pd.DataFrame, series_id: str, suffix: str, root):
    d = root / "data" / "agg"
    d.mkdir(parents=True, exist_ok=True)
    o = agg.reset_index()
    o.columns = ["time", "value", "max", "n", "coverage"]
    o["time"] = o["time"].dt.strftime(TIME_FMT)
    o.to_csv(d / f"{series_id}_{suffix}.csv", index=False, float_format="%.3f".replace(".3f", ".1f"))
