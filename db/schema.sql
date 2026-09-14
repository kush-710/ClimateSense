-- ClimateSense schema. Written for PostgreSQL 15 (Supabase / AWS RDS free tier).
-- SQLAlchemy models in pipeline/load.py mirror this and also run on SQLite for local dev.

CREATE TABLE IF NOT EXISTS cities (
    id            INTEGER PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    lat           DOUBLE PRECISION NOT NULL,
    lon           DOUBLE PRECISION NOT NULL,
    aqicn_slug    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS weather_data (
    id            BIGSERIAL PRIMARY KEY,
    city_id       INTEGER NOT NULL REFERENCES cities(id),
    ts            TIMESTAMPTZ NOT NULL,
    temp_c        DOUBLE PRECISION,
    humidity_pct  DOUBLE PRECISION,
    wind_kph      DOUBLE PRECISION,
    wind_dir_deg  DOUBLE PRECISION,
    uv_index      DOUBLE PRECISION,
    rainfall_mm   DOUBLE PRECISION,
    pressure_hpa  DOUBLE PRECISION,
    heat_index_c  DOUBLE PRECISION,
    solar_wm2     DOUBLE PRECISION,      -- shortwave radiation, W/m2 (archive + forecast)
    wbgt_c        DOUBLE PRECISION,      -- estimated outdoor Wet Bulb Globe Temperature
    UNIQUE (city_id, ts)
);
CREATE INDEX IF NOT EXISTS ix_weather_city_ts ON weather_data (city_id, ts DESC);

CREATE TABLE IF NOT EXISTS air_quality (
    id            BIGSERIAL PRIMARY KEY,
    city_id       INTEGER NOT NULL REFERENCES cities(id),
    ts            TIMESTAMPTZ NOT NULL,
    aqi_cpcb      DOUBLE PRECISION,      -- CPCB 0-500 composite
    pm25          DOUBLE PRECISION,      -- ug/m3
    pm10          DOUBLE PRECISION,
    no2           DOUBLE PRECISION,
    so2           DOUBLE PRECISION,
    o3            DOUBLE PRECISION,
    co            DOUBLE PRECISION,      -- mg/m3
    source        TEXT DEFAULT 'open-meteo',
    UNIQUE (city_id, ts, source)
);
CREATE INDEX IF NOT EXISTS ix_aq_city_ts ON air_quality (city_id, ts DESC);

-- ENSO / El Nino state. One row per calendar month (season = 3-month running window label).
CREATE TABLE IF NOT EXISTS enso_index (
    id            BIGSERIAL PRIMARY KEY,
    year          INTEGER NOT NULL,
    month         INTEGER NOT NULL,      -- centre month of the 3-month season
    season        TEXT NOT NULL,         -- e.g. 'DJF'
    oni           DOUBLE PRECISION NOT NULL,  -- SST anomaly, Nino 3.4, degC
    phase         TEXT NOT NULL,         -- 'el_nino' | 'la_nina' | 'neutral'
    UNIQUE (year, month)
);

-- Official NOAA CPC/IRI probabilistic ENSO forecast: a genuine forward outlook
-- (unlike enso_index, which is current/historical only). Re-issued ~monthly -
-- fully replaced on each load (see pipeline/load.py:replace_all), not upserted.
CREATE TABLE IF NOT EXISTS enso_outlook (
    id            BIGSERIAL PRIMARY KEY,
    year          INTEGER NOT NULL,      -- target season's centre-month year
    month         INTEGER NOT NULL,      -- target season's centre month
    season        TEXT NOT NULL,         -- e.g. 'JAS'
    el_nino_pct   INTEGER NOT NULL,
    neutral_pct   INTEGER NOT NULL,
    la_nina_pct   INTEGER NOT NULL,
    issued_year   INTEGER NOT NULL,      -- when CPC published this forecast
    issued_month  INTEGER NOT NULL,
    UNIQUE (year, month)
);

-- NOTE: `predictions` / `alerts` tables removed - defined but never used.
