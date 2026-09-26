"""Залить CSV хакатона в базу — для локальной проверки воркера и как образец схемы.

    python -m ml.csv_to_db --data ../dataset --url sqlite:///artifacts/demo.db
    python -m ml.csv_to_db --data ../dataset --url postgresql+psycopg2://u:p@localhost:5432/transport

Создаёт таблицы ``telemetry`` (validate/traffic.csv) и ``schedule_plan``
(validate/schedule_plan.csv — только план, без факта). Имена колонок — как в CSV,
то есть совпадают с настройками ``ml.db.DBConfig`` по умолчанию.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--url", required=True)
    args = ap.parse_args()
    eng = create_engine(args.url)
    tel = pd.read_csv(args.data / "validate/traffic.csv", low_memory=False,
                      usecols=["tr_id", "unit_id", "event_time", "receive_time", "location_valid",
                               "lon", "lat", "speed", "heading"])
    for c in ("event_time", "receive_time"):
        tel[c] = pd.to_datetime(tel[c], format="mixed")
    tel["location_valid"] = tel["location_valid"].astype(str).str.lower().eq("true")
    sch = pd.read_csv(args.data / "validate/schedule_plan.csv")
    sch["time_begin"] = pd.to_datetime(sch["time_begin"])
    tel.to_sql("telemetry", eng, if_exists="replace", index=False, chunksize=20000)
    sch.to_sql("schedule_plan", eng, if_exists="replace", index=False)
    with eng.begin() as c:
        from sqlalchemy import text
        c.execute(text('CREATE INDEX IF NOT EXISTS ix_tel_tr_time ON telemetry (tr_id, event_time)'))
        c.execute(text('CREATE INDEX IF NOT EXISTS ix_tel_time ON telemetry (event_time)'))
        c.execute(text('CREATE INDEX IF NOT EXISTS ix_sch_time ON schedule_plan (time_begin)'))
    print(f"telemetry: {len(tel)} строк, schedule_plan: {len(sch)} строк → {args.url}")


if __name__ == "__main__":
    main()
