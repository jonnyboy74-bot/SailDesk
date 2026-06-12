import logging
logging.getLogger("cfgrib").setLevel(logging.ERROR)   # suppress "skipping corrupted Message" noise

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import xarray as xr
import numpy as np
import httpx
import math
import time
import asyncio
import os
import pickle
from scipy.spatial import cKDTree

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENMETEO_API_KEY = "6b7jMhHrbkDV0ZiQ" 

GRIB_DIR = os.environ.get("GRIB_DIR", os.path.expanduser("~/Desktop/gribs"))
if not os.path.exists(GRIB_DIR):
    os.makedirs(GRIB_DIR)

# Parsed GRIB cache — persists between restarts so unchanged GRIBs load instantly
GRIB_CACHE_DIR = os.path.join(GRIB_DIR, "parsed_cache")
os.makedirs(GRIB_CACHE_DIR, exist_ok=True)

def _grib_cache_path(file_path: str) -> str:
    fname = os.path.basename(file_path)
    mtime = int(os.path.getmtime(file_path))
    return os.path.join(GRIB_CACHE_DIR, f"{fname}.{mtime}.pkl")

def _load_grib_cache(file_path: str):
    path = _grib_cache_path(file_path)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return None

def _save_grib_cache(file_path: str, data: dict):
    path = _grib_cache_path(file_path)
    try:
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"  ⚠️  Could not write GRIB cache: {e}")

def _clean_grib_cache():
    """Remove cache files whose parent GRIB no longer exists or has been replaced."""
    live = set()
    for config in MODEL_CONFIG.values():
        fp = find_latest_grib(config) if config.get("grib_prefix") else None
        if fp:
            live.add(os.path.basename(_grib_cache_path(fp)))
    for f in os.listdir(GRIB_CACHE_DIR):
        if f.endswith(".pkl") and f not in live:
            try:
                os.remove(os.path.join(GRIB_CACHE_DIR, f))
            except Exception:
                pass

MASTER_WEATHER_REGISTRY = {}

# ── Point API caches (wind + wave) — avoids hitting Open-Meteo on every pin move ──
# Key: (round_lat, round_lon, frozenset_models)  Value: (result_dict, unix_timestamp)
_WIND_POINT_CACHE: dict = {}
_WAVE_POINT_CACHE: dict = {}
_POINT_CACHE_TTL = 3600  # 1 hour

def _point_cache_key(lat: float, lon: float, models: list[str]) -> tuple:
    # Round to 0.05° (~5 km) so nearby points share cache entries
    return (round(lat * 20) / 20, round(lon * 20) / 20, tuple(sorted(models)))

MODEL_CONFIG = {
    # Wind models
    "ecmwf_ifs":                   {"grib_prefix": "ecmwf_early_0_1",  "api_model": "ecmwf_ifs",                   "type": "wind"},
    "lamma_1k":                    {"grib_prefix": "lamma_0_01",        "api_model": None,                          "type": "wind"},
    "icon_eu":                     {"grib_prefix": "icon_eu",           "api_model": "icon_eu",                     "type": "wind"},
    "meteofrance_arpege_europe":   {"grib_prefix": "arpege",            "api_model": "meteofrance_arpege_europe",   "type": "wind"},
    "meteofrance_arome_france_hd": {"grib_prefix": "arome_0_01",        "api_model": "meteofrance_arome_france_hd", "type": "wind"},
    "gfs_global":                  {"grib_prefix": None,                "api_model": "gfs_seamless",                "type": "wind"},
    # Wave models (GRIB only)
    "mfwam_arome":                 {"grib_prefix": "mfwam_arome_0_025", "api_model": None,                          "type": "wave"},
    "mfwam_ecmwf":                 {"grib_prefix": "mfwam_ecmwf_0_2",  "api_model": None,                          "type": "wave"},
}

def find_latest_grib(config: dict) -> str | None:
    """Return the path to the most recent .grb2 file matching grib_prefix, or None if no file exists."""
    prefix = config.get("grib_prefix")
    if not prefix:
        return None
    matches = sorted(
        [f for f in os.listdir(GRIB_DIR) if f.startswith(prefix) and f.endswith(".grb2")],
        reverse=True
    )
    return os.path.join(GRIB_DIR, matches[0]) if matches else None

def cleanup_old_gribs():
    """Keep only the most recent .grb2 file per model prefix; delete older ones.
    Also removes orphaned .idx files (whose parent .grb2 no longer exists)."""
    print("\n🧹 Running GRIB cleanup...")

    # Build set of all .grb2 files that should be kept after cleanup
    kept_gribs: set[str] = set()
    for model_id, config in MODEL_CONFIG.items():
        prefix = config.get("grib_prefix")
        if not prefix:
            continue
        matches = sorted(
            [f for f in os.listdir(GRIB_DIR) if f.startswith(prefix) and f.endswith(".grb2")],
            reverse=True
        )
        for old_file in matches[1:]:
            try:
                os.remove(os.path.join(GRIB_DIR, old_file))
                print(f"  🗑️  Removed old GRIB: {old_file}")
            except Exception as e:
                print(f"  ⚠️  Could not remove {old_file}: {e}")
        if matches:
            kept_gribs.add(matches[0])
            print(f"  ✅ {model_id}: keeping {matches[0]}")

    # Remove ALL .idx files on startup — orphaned ones and stale/corrupt ones alike.
    # cfgrib rebuilds them cleanly, preventing EOFError from half-written idx files.
    for f in os.listdir(GRIB_DIR):
        if not f.endswith(".idx"):
            continue
        try:
            os.remove(os.path.join(GRIB_DIR, f))
            print(f"  🗑️  Removed idx (will rebuild): {f}")
        except Exception as e:
            print(f"  ⚠️  Could not remove idx {f}: {e}")

    # Remove stale pickle caches (old GRIB versions no longer on disk)
    _clean_grib_cache()

    print("🧹 Cleanup complete.\n")

# 0.1° grid — close to native resolution for ECMWF (9km) and ICON-EU (7km)
# Uses the commercial Open-Meteo API (customer-api.open-meteo.com) which supports this density
API_LAT_START, API_LAT_END, API_LAT_STEP = 51.5, 38.0, -0.1
API_LON_START, API_LON_END, API_LON_STEP = -10.5, 14.5, 0.1

def generate_float_range(start, end, step):
    res = []
    curr = start
    if step > 0:
        while curr <= end + 1e-5:
            res.append(round(curr, 4))
            curr += step
    else:
        while curr >= end - 1e-5:
            res.append(round(curr, 4))
            curr += step
    return res

def delete_idx_for_grib(file_path: str):
    """Remove any .idx sidecar files for a given .grb2 so cfgrib rebuilds them cleanly."""
    for f in os.listdir(os.path.dirname(file_path)):
        if f.startswith(os.path.basename(file_path)) and f.endswith(".idx"):
            try:
                os.remove(os.path.join(os.path.dirname(file_path), f))
            except Exception:
                pass

def parse_and_regrid_grib(file_path: str, _retry=False):
    # Only delete idx on explicit retry (corruption recovery) — NOT on every parse
    if _retry:
        delete_idx_for_grib(file_path)
    import cfgrib
    datasets = cfgrib.open_datasets(file_path, errors='ignore')
    ds = None
    u_key, v_key = None, None
    for candidate_ds in datasets:
        u_key = next((k for k in ['10u', 'u10', 'u'] if k in candidate_ds), None)
        v_key = next((k for k in ['10v', 'v10', 'v'] if k in candidate_ds), None)
        if u_key and v_key:
            ds = candidate_ds
            break
    if ds is None:
        for candidate_ds in datasets: candidate_ds.close()
        raise Exception("GRIB format parsing error: Wind vectors not trackable.")

    raw_lats = ds.latitude.values
    raw_lons = ds.longitude.values
    if raw_lats.ndim == 1 and raw_lons.ndim == 1:
        lon_matrix, lat_matrix = np.meshgrid(raw_lons, raw_lats)
        grib_lats = lat_matrix.ravel()
        grib_lons = lon_matrix.ravel()
    else:
        grib_lats = raw_lats.ravel()
        grib_lons = raw_lons.ravel()
    grib_lons = np.where(grib_lons > 180, grib_lons - 360, grib_lons)
    min_lat, max_lat = float(np.min(grib_lats)), float(np.max(grib_lats))
    min_lon, max_lon = float(np.min(grib_lons)), float(np.max(grib_lons))

    ny = ds[u_key].shape[-2]
    nx = ds[u_key].shape[-1]
    target_lats = np.linspace(max_lat, min_lat, ny)
    target_lons = np.linspace(min_lon, max_lon, nx)
    dx = (max_lon - min_lon) / (nx - 1) if nx > 1 else 0.1
    dy = (max_lat - min_lat) / (ny - 1) if ny > 1 else 0.1

    grib_points = np.column_stack((grib_lats, grib_lons))
    spatial_tree = cKDTree(grib_points)
    lon_mesh, lat_mesh = np.meshgrid(target_lons, target_lats)
    target_points = np.column_stack((lat_mesh.ravel(), lon_mesh.ravel()))
    _, mapping_indices = spatial_tree.query(target_points, k=1)

    time_steps = ds.valid_time.values if 'valid_time' in ds else ds.time.values
    if not isinstance(time_steps, np.ndarray) or time_steps.ndim == 0:
        time_steps = np.array([time_steps])
    timestamps = [str(t)[:16].replace(' ', 'T') for t in time_steps]

    # Find gust variable (may be in a separate dataset with stepType=max)
    gust_lookup: dict = {}
    for candidate_ds in datasets:
        gust_key = next((k for k in ['fg10', '10fg', 'gust', 'fg', 'i10fg', 'wg'] if k in candidate_ds), None)
        if not gust_key:
            continue
        try:
            g_raw_lats = candidate_ds.latitude.values
            g_raw_lons = candidate_ds.longitude.values
            if g_raw_lats.ndim == 1 and g_raw_lons.ndim == 1:
                g_lon_m, g_lat_m = np.meshgrid(g_raw_lons, g_raw_lats)
                g_lats = g_lat_m.ravel(); g_lons = g_lon_m.ravel()
            else:
                g_lats = g_raw_lats.ravel(); g_lons = g_raw_lons.ravel()
            g_lons = np.where(g_lons > 180, g_lons - 360, g_lons)
            _, g_mapping = cKDTree(np.column_stack((g_lats, g_lons))).query(target_points, k=1)
            g_times = candidate_ds.valid_time.values if 'valid_time' in candidate_ds else candidate_ds.time.values
            if not isinstance(g_times, np.ndarray) or g_times.ndim == 0:
                g_times = np.array([g_times])
            g_ts_list = [str(t)[:16].replace(' ', 'T') for t in g_times]
            gust_arr = candidate_ds[gust_key].values
            for gi, gts in enumerate(g_ts_list):
                g_var = gust_arr[gi] if gust_arr.ndim > 2 else gust_arr
                gust_lookup[gts] = np.nan_to_num(g_var.ravel()[g_mapping].reshape(ny, nx)).astype(np.float32)
            print(f"  💨 Gust data found ({gust_key}, {len(g_ts_list)} steps)")
        except Exception as e:
            print(f"  ⚠️  Gust extraction failed: {e}")
        break  # use first gust dataset found

    compiled_frames = []
    for step_idx, ts in enumerate(timestamps):
        u_var = ds[u_key].values[step_idx] if ds[u_key].ndim > 2 else ds[u_key].values
        v_var = ds[v_key].values[step_idx] if ds[v_key].ndim > 2 else ds[v_key].values
        u_resampled = np.nan_to_num(u_var.ravel()[mapping_indices].reshape(ny, nx)).astype(np.float32)
        v_resampled = np.nan_to_num(v_var.ravel()[mapping_indices].reshape(ny, nx)).astype(np.float32)
        compiled_frames.append({
            "uData":    u_resampled,   # float32 numpy array — fast to pickle, compact
            "vData":    v_resampled,
            "gustData": gust_lookup.get(ts),  # already float32 numpy or None
        })

    run_time = None
    try:
        ref = ds["time"].values
        if hasattr(ref, "__iter__"): ref = next(iter(ref))
        run_time = str(ref)[:16].replace(" ", "T")
    except Exception:
        pass

    for candidate_ds in datasets: candidate_ds.close()
    return {
        "header": {"la1": max_lat, "lo1": min_lon, "dx": dx, "dy": dy, "nx": nx, "ny": ny},
        "timestamps": timestamps,
        "frames": compiled_frames,
        "model_type": "wind",
        "run_time": run_time,
    }

def parse_wave_grib(file_path: str, _retry=False):
    """Parse MFWAM wave GRIB: extracts swh, mwd, mwp and encodes for the rendering pipeline."""
    if _retry:
        delete_idx_for_grib(file_path)
    import cfgrib
    datasets = cfgrib.open_datasets(file_path, errors='ignore')

    swh_key = mwd_key = mwp_key = None
    ds = None
    for candidate_ds in datasets:
        swh_key = next((k for k in ['swh','2swh','shww','wh'] if k in candidate_ds), None)
        mwd_key = next((k for k in ['mwd','2mwd','mdww','wdir'] if k in candidate_ds), None)
        mwp_key = next((k for k in ['mwp','2mwp','mpww','pp1d','tm1'] if k in candidate_ds), None)
        if swh_key:
            ds = candidate_ds
            break

    if ds is None or swh_key is None:
        for d in datasets: d.close()
        raise Exception("Wave GRIB: could not find significant wave height variable.")

    raw_lats = ds.latitude.values
    raw_lons = ds.longitude.values
    if raw_lats.ndim == 1 and raw_lons.ndim == 1:
        lon_matrix, lat_matrix = np.meshgrid(raw_lons, raw_lats)
        grib_lats = lat_matrix.ravel()
        grib_lons = lon_matrix.ravel()
    else:
        grib_lats = raw_lats.ravel()
        grib_lons = raw_lons.ravel()
    grib_lons = np.where(grib_lons > 180, grib_lons - 360, grib_lons)

    # Regrid to shared 0.25° API grid — small, consistent, fast to pickle
    target_lats = np.array(generate_float_range(API_LAT_START, API_LAT_END, API_LAT_STEP))
    target_lons = np.array(generate_float_range(API_LON_START, API_LON_END, API_LON_STEP))
    ny = len(target_lats)
    nx = len(target_lons)
    dx = abs(API_LON_STEP)
    dy = abs(API_LAT_STEP)
    max_lat = API_LAT_START
    min_lon = API_LON_START

    grib_points = np.column_stack((grib_lats, grib_lons))
    spatial_tree = cKDTree(grib_points)
    lon_mesh, lat_mesh = np.meshgrid(target_lons, target_lats)
    target_points = np.column_stack((lat_mesh.ravel(), lon_mesh.ravel()))
    _, mapping_indices = spatial_tree.query(target_points, k=1)

    time_steps = ds.valid_time.values if 'valid_time' in ds else ds.time.values
    if not isinstance(time_steps, np.ndarray) or time_steps.ndim == 0:
        time_steps = np.array([time_steps])
    timestamps = [str(t)[:16].replace(' ', 'T') for t in time_steps]

    compiled_frames = []
    for step_idx in range(len(timestamps)):
        def _get(key, idx):
            if key is None: return None
            arr = ds[key].values
            return arr[idx] if arr.ndim > 2 else arr

        swh_raw = _get(swh_key, step_idx)
        mwd_raw = _get(mwd_key, step_idx)
        mwp_raw = _get(mwp_key, step_idx)

        swh_grid = np.nan_to_num(swh_raw.ravel()[mapping_indices].reshape(ny, nx))
        if mwd_raw is not None:
            mwd_grid = np.nan_to_num(mwd_raw.ravel()[mapping_indices].reshape(ny, nx))
            rad = np.radians(mwd_grid)
            u_grid = (-swh_grid * np.sin(rad)).astype(np.float32)
            v_grid = (-swh_grid * np.cos(rad)).astype(np.float32)
        else:
            u_grid = swh_grid.astype(np.float32)
            v_grid = np.zeros_like(swh_grid, dtype=np.float32)

        period_grid = np.nan_to_num(mwp_raw.ravel()[mapping_indices].reshape(ny, nx)).astype(np.float32) \
                      if mwp_raw is not None else None

        compiled_frames.append({
            "uData": u_grid,       # float32 numpy — fast pickle
            "vData": v_grid,
            "periodData": period_grid,
        })

    run_time = None
    try:
        ref = ds["time"].values
        if hasattr(ref, "__iter__"): ref = next(iter(ref))
        run_time = str(ref)[:16].replace(" ", "T")
    except Exception:
        pass

    for d in datasets: d.close()
    return {
        "header": {"la1": max_lat, "lo1": min_lon, "dx": dx, "dy": dy, "nx": nx, "ny": ny},
        "timestamps": timestamps,
        "frames": compiled_frames,
        "model_type": "wave",
        "run_time": run_time,
    }

# ── Open-Meteo models available for point comparison ──────────────────────────
# api_id = the Open-Meteo model slug to pass in requests (may differ from internal id)
OM_POINT_MODELS = {
    "ecmwf_ifs":                   {"label": "ECMWF IFS",    "res": "9km",   "api_id": "ecmwf_ifs"},
    "gfs_seamless":                {"label": "GFS",          "res": "22km",  "api_id": "gfs_seamless"},
    "gfs_global":                  {"label": "GFS",          "res": "22km",  "api_id": "gfs_seamless"},  # alias
    "icon_eu":                     {"label": "ICON-EU",      "res": "7km",   "api_id": "icon_eu"},
    "meteofrance_arpege_europe":   {"label": "ARPEGE",       "res": "11km",  "api_id": "meteofrance_arpege_europe"},
    "meteofrance_arome_france_hd": {"label": "AROME HD",     "res": "1.5km", "api_id": "meteofrance_arome_france_hd"},
}

class OmPointRequest(BaseModel):
    lat: float
    lon: float
    models: list[str]

@app.get("/config.js")
def get_config_js():
    token = os.environ.get("MAPBOX_TOKEN", "")
    if not token:
        try:
            with open("config.js") as f:
                return Response(content=f.read(), media_type="application/javascript")
        except FileNotFoundError:
            pass
    js = f'window.MAPBOX_ACCESS_TOKEN = "{token}";\n'
    return Response(content=js, media_type="application/javascript")

@app.get("/")
def get_interface():
    return FileResponse("index_v3.html")

@app.get("/weather")
def get_weather():
    return FileResponse("weather.html")

@app.get("/api/weather/grib-status")
def get_grib_status():
    status = {}
    for model_id, config in MODEL_CONFIG.items():
        file_path = find_latest_grib(config)
        cache_key = f"{model_id}_grib"
        if file_path and os.path.exists(file_path):
            loaded = cache_key in MASTER_WEATHER_REGISTRY
            status[model_id] = {
                "file": os.path.basename(file_path),
                "loaded": loaded,
                "status": "loaded" if loaded else "file_found"
            }
        else:
            status[model_id] = {"file": None, "loaded": False, "status": "missing"}
    return status

@app.get("/favicon.ico", include_in_schema=False)
async def silence_favicon_errors():
    return Response(status_code=204)

@app.get("/api/weather/metadata")
async def get_metadata(model: str = "ecmwf_ifs", source: str = "api"):
    cache_key = f"{model}_{source}"

    # Serve directly from cache — all loading is handled by startup background tasks.
    # Never parse synchronously here; that blocks the event loop and hangs the UI.
    if cache_key not in MASTER_WEATHER_REGISTRY:
        config = MODEL_CONFIG.get(model)
        if source == "grib":
            # Tell the frontend the file exists but is still loading vs. truly missing
            file_exists = bool(config and find_latest_grib(config))
            msg = "GRIB loading — please wait" if file_exists else f"No GRIB file found for {model}"
            return JSONResponse(status_code=503 if file_exists else 404,
                                content={"error": msg, "loading": file_exists})
        else:
            # API model not yet loaded by background task
            return JSONResponse(status_code=503,
                                content={"error": f"{model} API grid loading — please wait", "loading": True})

    profile = MASTER_WEATHER_REGISTRY[cache_key]
    return {"header": profile["header"], "timestamps": profile["timestamps"], "model_type": profile.get("model_type","wind")}

@app.get("/api/weather/frame")
async def get_frame(model: str = "ecmwf_ifs", source: str = "api", frame: int = 0):
    cache_key = f"{model}_{source}"
    if cache_key not in MASTER_WEATHER_REGISTRY:
        return JSONResponse(status_code=503, content={"error": "Data loading", "loading": True})

    profile = MASTER_WEATHER_REGISTRY[cache_key]
    frames = profile["frames"]
    if frame >= len(frames) or frame < 0: frame = 0
    f = frames[frame]
    def _to_list(v):
        if v is None: return None
        return v.tolist() if isinstance(v, np.ndarray) else v
    return {"uData": _to_list(f["uData"]), "vData": _to_list(f["vData"]), "gustData": _to_list(f.get("gustData"))}

@app.get("/api/weather/frame.bin")
async def get_frame_binary(model: str = "ecmwf_ifs", source: str = "api", frame: int = 0):
    """Return frame as compact binary: 2×int32 (ny,nx) + ny*nx float32 uData + ny*nx float32 vData.
    ~5x smaller than JSON, parses 10x faster via Float32Array in the browser."""
    cache_key = f"{model}_{source}"
    if cache_key not in MASTER_WEATHER_REGISTRY:
        return JSONResponse(status_code=503, content={"error": "Data loading", "loading": True})

    profile = MASTER_WEATHER_REGISTRY[cache_key]
    frames = profile["frames"]
    if frame >= len(frames) or frame < 0: frame = 0
    f = frames[frame]
    h = profile["header"]
    ny, nx = int(h["ny"]), int(h["nx"])

    def _as_f32(v):
        if isinstance(v, np.ndarray): return v.flatten().astype(np.float32)
        return np.array(v, dtype=np.float32).flatten()

    u = _as_f32(f["uData"])
    v = _as_f32(f["vData"])
    g = _as_f32(f["gustData"]) if f.get("gustData") is not None else np.zeros(ny * nx, dtype=np.float32)

    # Layout: [ny int32][nx int32][uData ny*nx float32][vData ny*nx float32][gustData ny*nx float32]
    header_bytes = np.array([ny, nx], dtype=np.int32).tobytes()
    body = header_bytes + u.tobytes() + v.tobytes() + g.tobytes()
    return Response(content=body, media_type="application/octet-stream")

async def load_individual_source_into_registry(model: str, source: str):
    cache_key = f"{model}_{source}"
    config = MODEL_CONFIG.get(model)
    
    if not config: return
        
    if source == "grib":
        file_path = find_latest_grib(config)
        if not file_path:
            print(f"⚠️  No GRIB file found for {model} (prefix: {config.get('grib_prefix')})")
            return
        try:
            if config.get("type") == "wave":
                MASTER_WEATHER_REGISTRY[cache_key] = parse_wave_grib(file_path)
            else:
                MASTER_WEATHER_REGISTRY[cache_key] = parse_and_regrid_grib(file_path)
        except Exception as e:
            print(f"⚠️  Failed to parse GRIB for {model}: {e}")
    else:
        try:
            lats = generate_float_range(API_LAT_START, API_LAT_END, API_LAT_STEP)
            lons = generate_float_range(API_LON_START, API_LON_END, API_LON_STEP)
            nx, ny = len(lons), len(lats)
            MASTER_WEATHER_REGISTRY[cache_key] = await process_live_web_api(config["api_model"], OPENMETEO_API_KEY, nx, ny, lats, lons)
        except Exception: pass

async def fetch_api_chunk(client, semaphore, chunk_coords, api_model, api_key):
    async with semaphore:
        flat_lats = [c[0] for c in chunk_coords]
        flat_lons = [c[1] for c in chunk_coords]
        payload = {
            "latitude": flat_lats, "longitude": flat_lons,
            "hourly": ["wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"],
            "wind_speed_unit": "ms", "forecast_days": 5, "models": [api_model], "apikey": api_key
        }
        res = await client.post("https://customer-api.open-meteo.com/v1/forecast", json=payload, timeout=60.0)
        raw = res.json()
        return raw if isinstance(raw, list) else [raw]

async def process_live_web_api(api_model: str, api_key: str, nx: int, ny: int, lats: list, lons: list):
    all_coords = [(lat, lon) for lat in lats for lon in lons]
    chunk_size = 1000
    coord_chunks = [all_coords[i:i + chunk_size] for i in range(0, len(all_coords), chunk_size)]
    semaphore = asyncio.Semaphore(8)
    all_results = []
    async with httpx.AsyncClient() as client:
        tasks = [fetch_api_chunk(client, semaphore, chunk, api_model, api_key) for chunk in coord_chunks]
        completed = await asyncio.gather(*tasks)
        for chunk_res in completed: all_results.extend(chunk_res)

    target_indices = list(range(0, 120, 1))
    timeline_timestamps = [all_results[0]["hourly"]["time"][t] for t in target_indices]
    compiled_frames = []
    for t in target_indices:
        u_grid = [[0.0 for _ in range(nx)] for _ in range(ny)]
        v_grid = [[0.0 for _ in range(nx)] for _ in range(ny)]
        g_grid = [[None for _ in range(nx)] for _ in range(ny)]
        idx = 0
        for r in range(ny):
            for c in range(nx):
                hourly_data = all_results[idx].get("hourly", {})
                speed = hourly_data.get("wind_speed_10m", [])[t] or 0.0
                direction = hourly_data.get("wind_direction_10m", [])[t] or 0.0
                gust_ms = hourly_data.get("wind_gusts_10m", [None])[t] if hourly_data.get("wind_gusts_10m") else None
                rad = math.radians(direction)
                u_grid[r][c] = -speed * math.sin(rad)
                v_grid[r][c] = -speed * math.cos(rad)
                g_grid[r][c] = gust_ms
                idx += 1
        compiled_frames.append({"uData": u_grid, "vData": v_grid, "gustData": g_grid})

    # Estimate model run time: first timestamp rounded down to the nearest 6-hour boundary
    run_time = None
    try:
        first = timeline_timestamps[0]  # e.g. "2026-06-12T08"
        h = int(first[11:13])
        run_h = (h // 6) * 6
        run_time = f"{first[:10]}T{run_h:02d}:00"
    except Exception:
        pass

    return {
        "header": {"la1": API_LAT_START, "lo1": API_LON_START, "dx": abs(API_LON_STEP), "dy": abs(API_LAT_STEP), "nx": nx, "ny": ny},
        "timestamps": timeline_timestamps,
        "frames": compiled_frames,
        "model_type": "wind",
        "run_time": run_time,
    }

@app.get("/api/weather/point")
async def get_weather_point(lat: float, lon: float, model: str = "ecmwf_ifs", source: str = "grib"):
    """Extract a point time-series (TWS kt, TWD deg) from loaded GRIB or API grid."""
    cache_key = f"{model}_{source}"
    if cache_key not in MASTER_WEATHER_REGISTRY:
        return JSONResponse(status_code=503, content={"error": f"Data loading for {model}/{source}"})

    profile = MASTER_WEATHER_REGISTRY[cache_key]
    header, frames, timestamps = profile["header"], profile["frames"], profile["timestamps"]

    la1, lo1 = header["la1"], header["lo1"]
    dx, dy   = header["dx"],  header["dy"]
    nx, ny   = header["nx"],  header["ny"]

    col = int(round((lon - lo1) / dx))
    row = int(round((la1 - lat) / dy))
    col = max(0, min(nx - 1, col))
    row = max(0, min(ny - 1, row))

    model_type = profile.get("model_type", "wind")
    result = []
    for i, ts in enumerate(timestamps):
        u = frames[i]["uData"][row][col]
        v = frames[i]["vData"][row][col]
        if model_type == "wave":
            # u/v encoded as swh*sin/cos(mwd) — recover swh and direction
            swh = round(math.sqrt(u * u + v * v), 2)
            mwd = round((math.degrees(math.atan2(-u, -v)) + 360) % 360, 0)
            period_raw = frames[i].get("periodData")
            try:
                period_cell = period_raw[row][col] if period_raw is not None else None
                period = round(float(period_cell), 1) if period_cell is not None else None
            except (TypeError, ValueError):
                period = None
            result.append({"timestamp": ts, "tws": swh, "twd": mwd, "gust": period, "cape": None})
        else:
            speed_ms  = math.sqrt(float(u) * float(u) + float(v) * float(v))
            tws_kt    = round(speed_ms * 1.94384, 1)
            twd       = round((math.degrees(math.atan2(-float(u), -float(v))) + 360) % 360, 0)
            gust_cell = frames[i]["gustData"][row][col] if frames[i].get("gustData") is not None else None
            # gust_cell may be None (API list), numpy float32, or Python float — handle all
            try:
                gust_raw = float(gust_cell) if gust_cell is not None else None
            except (TypeError, ValueError):
                gust_raw = None
            gust_kt = round(gust_raw * 1.94384, 1) if gust_raw is not None and not math.isnan(gust_raw) else None
            result.append({"timestamp": ts, "tws": tws_kt, "twd": twd, "gust": gust_kt, "cape": None})

    return {"model": model, "source": source, "model_type": model_type, "run_time": profile.get("run_time"), "data": result}


@app.post("/api/weather/om")
async def get_weather_om(body: OmPointRequest):
    """Fetch point wind forecasts from Open-Meteo for one or more models."""
    ck = _point_cache_key(body.lat, body.lon, body.models)
    if ck in _WIND_POINT_CACHE:
        cached, ts = _WIND_POINT_CACHE[ck]
        if time.time() - ts < _POINT_CACHE_TTL:
            return cached
    results = {}
    async with httpx.AsyncClient() as client:
        tasks = {}
        for model_id in body.models:
            if model_id not in OM_POINT_MODELS:
                continue
            api_id = OM_POINT_MODELS[model_id].get("api_id", model_id)
            tasks[model_id] = client.get(
                "https://customer-api.open-meteo.com/v1/forecast",
                params={
                    "latitude":        body.lat,
                    "longitude":       body.lon,
                    "hourly":          "wind_speed_10m,wind_direction_10m,wind_gusts_10m,cape",
                    "wind_speed_unit": "kn",
                    "forecast_days":   7,
                    "models":          api_id,
                    "apikey":          OPENMETEO_API_KEY,
                },
                timeout=30.0,
            )
        responses = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for model_id, resp in zip(tasks.keys(), responses):
            if isinstance(resp, Exception):
                results[model_id] = {"error": str(resp)}
                continue
            try:
                d = resp.json()
                h = d.get("hourly", {})
                times  = h.get("time", [])
                speeds = h.get("wind_speed_10m", [])
                dirs   = h.get("wind_direction_10m", [])
                gusts  = h.get("wind_gusts_10m", [])
                capes  = h.get("cape", [])
                point_data = []
                for i, ts in enumerate(times):
                    point_data.append({
                        "timestamp": ts,
                        "tws":  round(speeds[i], 1) if speeds and speeds[i] is not None else None,
                        "twd":  round(dirs[i],   0) if dirs   and dirs[i]   is not None else None,
                        "gust": round(gusts[i],  1) if gusts  and gusts[i]  is not None else None,
                        "cape": round(capes[i],  0) if capes  and i < len(capes) and capes[i] is not None else None,
                    })
                run_time = None
                try:
                    h0 = int(times[0][11:13]) if times else 0
                    run_time = f"{times[0][:10]}T{((h0 // 6) * 6):02d}:00"
                except Exception:
                    pass
                results[model_id] = {"source": "om", "run_time": run_time, "data": point_data}
            except Exception as e:
                results[model_id] = {"error": str(e)}
    _WIND_POINT_CACHE[ck] = (results, time.time())
    return results


OM_MARINE_MODELS = {
    "ecmwf_wam":    {"label": "ECMWF WAM",  "res": "9km"},    # global, 9km — confirmed working
    # "ncep_gfswave": {"label": "GFS Wave",   "res": "25km"},  # not returning data via OM Marine API
    # "dwd_ewam":     {"label": "DWD EWAM",   "res": "5km"},   # not returning data via OM Marine API
    # "mfwam":        {"label": "MF WAM",     "res": "8km"},   # not returning data via OM Marine API
}

class MarinePointRequest(BaseModel):
    lat: float
    lon: float
    models: list[str]

@app.post("/api/marine/om")
async def get_marine_om(body: MarinePointRequest):
    """Fetch point wave forecasts from Open-Meteo Marine API."""
    ck = _point_cache_key(body.lat, body.lon, body.models)
    if ck in _WAVE_POINT_CACHE:
        cached, ts = _WAVE_POINT_CACHE[ck]
        if time.time() - ts < _POINT_CACHE_TTL:
            return cached
    results = {}
    async with httpx.AsyncClient() as client:
        tasks = {}
        for model_id in body.models:
            if model_id not in OM_MARINE_MODELS:
                continue
            tasks[model_id] = client.get(
                "https://customer-marine-api.open-meteo.com/v1/marine",
                params={
                    "latitude":      body.lat,
                    "longitude":     body.lon,
                    "hourly":        "wave_height,wave_direction,wave_period",
                    "forecast_days": 7,
                    "models":        model_id,
                    "apikey":        OPENMETEO_API_KEY,
                },
                timeout=30.0,
            )
        responses = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for model_id, resp in zip(tasks.keys(), responses):
            if isinstance(resp, Exception):
                results[model_id] = {"error": str(resp)}
                continue
            try:
                d = resp.json()
                if "error" in d:
                    results[model_id] = {"error": d["error"]}
                    continue
                h = d.get("hourly", {})
                times  = h.get("time", [])
                heights = h.get("wave_height", [])
                dirs    = h.get("wave_direction", [])
                periods = h.get("wave_period", [])
                run_time = None
                try:
                    h0 = int(times[0][11:13]) if times else 0
                    run_time = f"{times[0][:10]}T{((h0 // 6) * 6):02d}:00"
                except Exception:
                    pass
                point_data = []
                for i, ts in enumerate(times):
                    point_data.append({
                        "timestamp": ts,
                        "tws":  round(heights[i], 2) if heights and i < len(heights) and heights[i] is not None else None,
                        "twd":  round(dirs[i],    0) if dirs    and i < len(dirs)    and dirs[i]    is not None else None,
                        "gust": round(periods[i], 1) if periods and i < len(periods) and periods[i]  is not None else None,
                        "cape": None,
                    })
                results[model_id] = {"source": "om", "model_type": "wave", "run_time": run_time, "data": point_data}
            except Exception as e:
                results[model_id] = {"error": str(e)}
    _WAVE_POINT_CACHE[ck] = (results, time.time())
    return results

@app.get("/api/weather/om-models")
def get_om_models():
    """Return available Open-Meteo models for the comparison table."""
    return OM_POINT_MODELS


def _load_one_grib(model_id: str, config: dict):
    """Load a single GRIB into memory.
    Tries pickle cache first (instant). Falls back to cfgrib parse + saves cache.
    Returns (model_id, cache_key, tracking_tag, data) or (model_id, None, None, None)."""
    file_path = find_latest_grib(config)
    if not file_path or not os.path.exists(file_path):
        return model_id, None, None, None
    cache_key    = f"{model_id}_grib"
    mtime        = os.path.getmtime(file_path)
    tracking_tag = f"{cache_key}_{mtime}"
    if tracking_tag in MASTER_WEATHER_REGISTRY:
        return model_id, None, None, None  # already in memory

    # ── Try fast pickle cache first ──────────────────────────────────────────
    cached = _load_grib_cache(file_path)
    if cached is not None:
        print(f"  ⚡ {model_id.upper()} loaded from cache ({len(cached['timestamps'])} frames — no parse needed).")
        return model_id, cache_key, tracking_tag, cached

    # ── First time: parse with cfgrib (sequential — cfgrib is not thread-safe) ─
    fname = os.path.basename(file_path)
    print(f"  📦 Parsing {model_id.upper()} ({fname})...")
    try:
        is_wave  = config.get("type") == "wave"
        parse_fn = parse_wave_grib if is_wave else parse_and_regrid_grib
        try:
            data = parse_fn(file_path)
        except (EOFError, Exception) as e:
            if 'idx' in str(e).lower() or isinstance(e, EOFError):
                print(f"  ⚠️  Corrupt idx for {model_id.upper()}, retrying clean...")
                data = parse_fn(file_path, _retry=True)
            else:
                raise
        _save_grib_cache(file_path, data)  # save so next restart is instant
        print(f"  ✅ {model_id.upper()} ready ({len(data['timestamps'])} frames). Cache saved.")
        return model_id, cache_key, tracking_tag, data
    except Exception as e:
        print(f"  ⚠️  {model_id.upper()} failed: {e}")
        return model_id, None, None, None


def warm_up_local_grib_registry():
    """Load all GRIBs sequentially (cfgrib is not thread-safe).
    Uses pickle disk cache so subsequent restarts skip cfgrib entirely."""
    gribs = [(mid, cfg) for mid, cfg in MODEL_CONFIG.items() if find_latest_grib(cfg)]
    if not gribs:
        print("  ℹ️  No GRIB files found — skipping warmup.")
        return
    print(f"\n🔥 Loading {len(gribs)} GRIB(s) (cache used where available)...")
    for model_id, config in gribs:
        _, cache_key, tracking_tag, data = _load_one_grib(model_id, config)
        if data is not None:
            MASTER_WEATHER_REGISTRY[cache_key]    = data
            MASTER_WEATHER_REGISTRY[tracking_tag] = data
    print("🎉 GRIB load complete.\n")

API_REFRESH_HOURS = 3  # re-fetch Open-Meteo grids every 3 hours

async def load_all_api_models(force=False):
    """
    Fetch Open-Meteo grid data for every unique API model.
    Deduplicates: lamma_1k shares icon_eu's grid so it's only fetched once.
    Set force=True to refresh even if already cached.
    """
    lats = generate_float_range(API_LAT_START, API_LAT_END, API_LAT_STEP)
    lons = generate_float_range(API_LON_START, API_LON_END, API_LON_STEP)
    nx, ny = len(lons), len(lats)

    # Deduplicate: map unique api_model slug → list of cache keys that share it
    api_model_to_keys: dict = {}
    GRIB_ONLY_MODELS = {"lamma_1k"}  # no real API equivalent
    for model_id, config in MODEL_CONFIG.items():
        if model_id in GRIB_ONLY_MODELS:
            continue
        api_slug = config.get("api_model")
        if not api_slug:
            continue
        api_model_to_keys.setdefault(api_slug, []).append(f"{model_id}_api")

    # Load all models concurrently — pure async I/O, safe to parallelise unlike GRIB
    async def _fetch_one(api_slug, cache_keys):
        primary_key = cache_keys[0]
        if not force and primary_key in MASTER_WEATHER_REGISTRY:
            return
        print(f"  🌐 {'Refreshing' if force else 'Loading'} API model {api_slug.upper()}...")
        try:
            data = await process_live_web_api(api_slug, OPENMETEO_API_KEY, nx, ny, lats, lons)
            for key in cache_keys:
                MASTER_WEATHER_REGISTRY[key] = data
            print(f"  ✅ {api_slug.upper()} ready ({len(data['timestamps'])} frames) → {cache_keys}")
        except Exception as e:
            print(f"  ⚠️  {api_slug.upper()} API failed: {e}")

    await asyncio.gather(*[_fetch_one(slug, keys) for slug, keys in api_model_to_keys.items()])

async def api_refresh_loop():
    """Refresh all Cloud API grids every API_REFRESH_HOURS hours."""
    while True:
        await asyncio.sleep(API_REFRESH_HOURS * 3600)
        print(f"\n🔄 Scheduled API refresh (every {API_REFRESH_HOURS}h)...")
        await load_all_api_models(force=True)
        print("✅ API refresh complete.\n")

def reload_grib_if_changed(model_id: str, config: dict) -> bool:
    """
    Check if the latest GRIB file for a model is newer than what's cached.
    If so, parse and replace the registry entry. Returns True if reloaded.
    """
    file_path = find_latest_grib(config)
    if not file_path or not os.path.exists(file_path):
        return False
    cache_key = f"{model_id}_grib"
    mtime = os.path.getmtime(file_path)
    tracking_tag = f"{cache_key}_{mtime}"
    if tracking_tag in MASTER_WEATHER_REGISTRY:
        return False  # already current
    fname = os.path.basename(file_path)
    print(f"  📦 New GRIB detected for {model_id.upper()} ({fname}) — reloading...")
    try:
        # New GRIB file — wipe old idx and stale tracking tags
        delete_idx_for_grib(file_path)
        stale = [k for k in MASTER_WEATHER_REGISTRY if k.startswith(cache_key + "_")]
        for k in stale:
            del MASTER_WEATHER_REGISTRY[k]
        is_wave = config.get("type") == "wave"
        parse_fn = parse_wave_grib if is_wave else parse_and_regrid_grib
        data = parse_fn(file_path)
        _save_grib_cache(file_path, data)  # update disk cache for next restart
        MASTER_WEATHER_REGISTRY[cache_key]    = data
        MASTER_WEATHER_REGISTRY[tracking_tag] = data
        print(f"  ✅ {model_id.upper()} reloaded ({len(data['timestamps'])} frames).")
        return True
    except Exception as e:
        print(f"  ⚠️  {model_id.upper()} reload failed: {e}")
        return False

async def grib_watch_loop():
    """
    Every 2 minutes: scan for new/updated GRIB files, reload changed ones,
    and clean up old files from disk.
    """
    loop = asyncio.get_event_loop()
    while True:
        await asyncio.sleep(120)  # check every 2 minutes
        any_new = False
        for model_id, config in MODEL_CONFIG.items():
            changed = await loop.run_in_executor(
                None, reload_grib_if_changed, model_id, config
            )
            if changed:
                any_new = True
        if any_new:
            # Clean up old GRIB files now that new ones are loaded
            await loop.run_in_executor(None, cleanup_old_gribs)
            print("🧹 Old GRIBs cleaned up after reload.\n")

@app.on_event("startup")
async def on_startup():
    """
    Server is available immediately on startup.
    - GRIBs parse in a background thread (non-blocking, ~30s)
    - API grids fetch in background async tasks (non-blocking, ~2min)
    - API grids auto-refresh every 3 hours
    Any request before warmup completes falls through to lazy-loading.
    """
    cleanup_old_gribs()
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, warm_up_local_grib_registry)  # background thread
    asyncio.create_task(load_all_api_models())                # background async
    asyncio.create_task(api_refresh_loop())                   # refreshes every 3h
    asyncio.create_task(grib_watch_loop())                    # watches for new GRIBs every 2min

@app.get("/api/models/loaded")
def get_models_loaded():
    """Return all keys currently in MASTER_WEATHER_REGISTRY so the frontend
    can auto-update dots without the user clicking each model."""
    return {"loaded": list(MASTER_WEATHER_REGISTRY.keys())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)