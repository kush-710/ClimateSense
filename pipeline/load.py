"""SQLAlchemy 2.0 ORM models + idempotent upsert loaders.

Runs unchanged on:
  - SQLite (local dev, zero cost)
  - Supabase Postgres free tier
  - AWS RDS Postgres free tier (db.t3.micro, 12 months) - just change DATABASE_URL.
"""
from __future__ import annotations
import logging

import pandas as pd
from sqlalchemy import (
    create_engine, Integer, BigInteger, Float, Text, TIMESTAMP,
    UniqueConstraint, Index, insert,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATABASE_URL, CITIES

logger = logging.getLogger(__name__)
engine = create_engine(DATABASE_URL, pool_pre_ping=True)


class Base(DeclarativeBase):
    pass


class City(Base):
    __tablename__ = "cities"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(Text, unique=True)
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    aqicn_slug: Mapped[str] = mapped_column(Text)


class WeatherData(Base):
    __tablename__ = "weather_data"
    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"),
                                    primary_key=True, autoincrement=True)
    city_id: Mapped[int] = mapped_column(Integer)
    ts: Mapped[str] = mapped_column(TIMESTAMP(timezone=True))
    temp_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    humidity_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_kph: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_dir_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    uv_index: Mapped[float | None] = mapped_column(Float, nullable=True)
    rainfall_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    pressure_hpa: Mapped[float | None] = mapped_column(Float, nullable=True)
    heat_index_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    solar_wm2: Mapped[float | None] = mapped_column(Float, nullable=True)
    wbgt_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    __table_args__ = (UniqueConstraint("city_id", "ts", name="uq_weather_city_ts"),
                      Index("ix_weather_city_ts", "city_id", "ts"))


class AirQuality(Base):
    __tablename__ = "air_quality"
    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"),
                                    primary_key=True, autoincrement=True)
    city_id: Mapped[int] = mapped_column(Integer)
    ts: Mapped[str] = mapped_column(TIMESTAMP(timezone=True))
    aqi_cpcb: Mapped[float | None] = mapped_column(Float, nullable=True)
    pm25: Mapped[float | None] = mapped_column(Float, nullable=True)
    pm10: Mapped[float | None] = mapped_column(Float, nullable=True)
    no2: Mapped[float | None] = mapped_column(Float, nullable=True)
    so2: Mapped[float | None] = mapped_column(Float, nullable=True)
    o3: Mapped[float | None] = mapped_column(Float, nullable=True)
    co: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(Text, default="open-meteo")
    __table_args__ = (UniqueConstraint("city_id", "ts", "source", name="uq_aq_city_ts_src"),
                      Index("ix_aq_city_ts", "city_id", "ts"))


class EnsoIndex(Base):
    __tablename__ = "enso_index"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    year: Mapped[int] = mapped_column(Integer)
    month: Mapped[int] = mapped_column(Integer)
    season: Mapped[str] = mapped_column(Text)
    oni: Mapped[float] = mapped_column(Float)
    phase: Mapped[str] = mapped_column(Text)
    __table_args__ = (UniqueConstraint("year", "month", name="uq_enso_ym"),)


class EnsoOutlook(Base):
    """Official NOAA CPC/IRI probabilistic ENSO forecast - a genuine forward
    outlook (unlike EnsoIndex, which is current/historical). Re-issued monthly,
    so rows are fully replaced each load rather than append-only upserted."""
    __tablename__ = "enso_outlook"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    year: Mapped[int] = mapped_column(Integer)
    month: Mapped[int] = mapped_column(Integer)
    season: Mapped[str] = mapped_column(Text)
    el_nino_pct: Mapped[int] = mapped_column(Integer)
    neutral_pct: Mapped[int] = mapped_column(Integer)
    la_nina_pct: Mapped[int] = mapped_column(Integer)
    issued_year: Mapped[int] = mapped_column(Integer)
    issued_month: Mapped[int] = mapped_column(Integer)
    __table_args__ = (UniqueConstraint("year", "month", name="uq_enso_outlook_ym"),)


# NOTE: `predictions` and `alerts` tables were removed - they were defined
# here and in db/schema.sql but never written to or read from anywhere.
# Forecast-vs-actual accuracy tracking is a worthwhile feature to build on a
# `predictions` table later; it just needs a scheduled job that scores past
# forecasts against observations, which doesn't exist yet.


def init_db() -> None:
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        for cid, c in CITIES.items():
            if s.get(City, cid) is None:
                s.add(City(id=cid, name=c["name"], lat=c["lat"], lon=c["lon"],
                           aqicn_slug=c["aqicn_slug"]))
        s.commit()


def upsert_df(df: pd.DataFrame, model: type[Base], conflict_cols: list[str]) -> int:
    """Idempotent bulk upsert. Uses dialect-native ON CONFLICT DO NOTHING.

    Batched to stay under SQLite's bound-parameter ceiling (SQLITE_MAX_VARIABLE_NUMBER,
    999 on many builds) - a single-statement insert of a multi-year hourly seed would
    otherwise raise 'too many SQL variables'. Batching is harmless on Postgres too.
    """
    if df.empty:
        return 0
    cols = [c.name for c in model.__table__.columns if c.name != "id"]
    records = df[[c for c in cols if c in df.columns]].where(pd.notna(df), None) \
                                                      .to_dict(orient="records")
    dialect = engine.dialect.name
    batch_size = max(1, 900 // max(1, len(cols)))
    with engine.begin() as conn:
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            if dialect == "postgresql":
                from sqlalchemy.dialects.postgresql import insert as pg_insert
                stmt = pg_insert(model.__table__).values(batch)
                stmt = stmt.on_conflict_do_nothing(index_elements=conflict_cols)
                conn.execute(stmt)
            elif dialect == "sqlite":
                from sqlalchemy.dialects.sqlite import insert as lite_insert
                stmt = lite_insert(model.__table__).values(batch)
                stmt = stmt.on_conflict_do_nothing(index_elements=conflict_cols)
                conn.execute(stmt)
            else:
                conn.execute(insert(model.__table__), batch)
    logger.info("upserted %d rows into %s", len(records), model.__tablename__)
    return len(records)


def replace_all(df: pd.DataFrame, model: type[Base]) -> int:
    """Full replace (delete-then-insert), for small tables that are re-issued
    wholesale each run rather than appended to (e.g. EnsoOutlook - a forecast
    that gets revised, not an immutable historical record)."""
    cols = [c.name for c in model.__table__.columns if c.name != "id"]
    with engine.begin() as conn:
        conn.execute(model.__table__.delete())
        if not df.empty:
            records = df[[c for c in cols if c in df.columns]] \
                        .where(pd.notna(df), None).to_dict(orient="records")
            conn.execute(insert(model.__table__), records)
    logger.info("replaced %s with %d rows", model.__tablename__, len(df))
    return len(df)
