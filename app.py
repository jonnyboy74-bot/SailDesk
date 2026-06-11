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

MASTER_WEATHER_REGISTRY = {}

MODEL_CONFIG = {
    "ecmwf_ifs":                   {"grib_prefix": "ecmwf_tst_0_1", "api_model": "ecmwf_ifs"},
    "lamma_1k":                    {"grib_prefix": "lamma_0_01",    "api_model": "icon_eu"},
    "icon_eu":                     {"grib_prefix": "icon_eu",       "api_model": "icon_eu"},
    "meteofrance_arpege_europe":   {"grib_prefix": "arpege",        "api_model": "meteofrance_arpege_europe"},
    "meteofrance_arome_france_hd": {"grib_prefix": "arome_0_01",    "api_model": "meteofrance_arome_france_hd"},
}

def find_latest_grib(config: dict) -> str | None:
    """Return the path to the most recent .grb2 file matching grib_prefix, or None if no file exists."""
    prefix = config["grib_prefix"]
    matches = sorted(
        [f for f in os.listdir(GRIB_DIR) if f.startswith(prefix) and f.endswith(".grb2")],
        reverse=True
    )
    return os.path.join(GRIB_DIR, matches[0]) if matches else None

def cleanup_old_gribs():
    """Keep only the most recent .grb2 file per model prefix; delete older ones."""
    print("\n🧹 Running GRIB cleanup — keeping latest file per model...")
    for model_id, config in MODEL_CONFIG.items():
        prefix = config["grib_prefix"]
        matches = sorted(
            [f for f in os.listdir(GRIB_DIR) if f.startswith(prefix) and f.endswith(".grb2")],
            reverse=True
        )
        for old_file in matches[1:]:
            old_path = os.path.join(GRIB_DIR, old_file)
            os.remove(old_path)
            print(f"  🗑️  Removed old GRIB: {old_file}")
        if matches:
            print(f"  ✅ {model_id}: keeping {matches[0]}")
    print("🧹 Cleanup complete.\n")

# ─── FIXED: STRICT 0.25 DEGREE CLOUD API RESOLUTION FOR INSTANT LOADING ───
API_LAT_START, API_LAT_END, API_LAT_STEP = 51.5, 38.0, -0.25
API_LON_START, API_LON_END, API_LON_STEP = -10.5, 14.5, 0.25

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

def parse_and_regrid_grib(file_path: str):
    import cfgrib
    datasets = cfgrib.open_datasets(file_path)
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

    ny = ds[u_key].shape[-2]
    nx = ds[u_key].shape[-1]

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
    compiled_frames = []
    for step_idx in range(len(timestamps)):
        u_var = ds[u_key].values[step_idx] if ds[u_key].ndim > 2 else ds[u_key].values
        v_var = ds[v_key].values[step_idx] if ds[v_key].ndim > 2 else ds[v_key].values
        u_resampled = u_var.ravel()[mapping_indices].reshape(ny, nx)
        v_resampled = v_var.ravel()[mapping_indices].reshape(ny, nx)
        compiled_frames.append({
            "uData": np.nan_to_num(u_resampled).tolist(),
            "vData": np.nan_to_num(v_resampled).tolist()
        })

    for candidate_ds in datasets: candidate_ds.close()
    return {
        "header": {"la1": max_lat, "lo1": min_lon, "dx": dx, "dy": dy, "nx": nx, "ny": ny},
        "timestamps": timestamps,
        "frames": compiled_frames
    }

# ── Open-Meteo models available for point comparison ──────────────────────────
OM_POINT_MODELS = {
    "ecmwf_ifs":                   {"label": "ECMWF IFS",    "res": "9km"},
    "gfs_seamless":                {"label": "GFS",          "res": "13km"},
    "icon_eu":                     {"label": "ICON-EU",      "res": "7km"},
    "meteofrance_arpege_europe":   {"label": "ARPEGE",       "res": "11km"},
    "meteofrance_arome_france_hd": {"label": "AROME HD",     "res": "1.5km"},
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
    config = MODEL_CONFIG.get(model)
    
    if source == "grib" and config:
        file_path = find_latest_grib(config)
        if file_path and os.path.exists(file_path):
            file_mtime = os.path.getmtime(file_path)
            tracking_tag = f"{cache_key}_{file_mtime}"
            if tracking_tag not in MASTER_WEATHER_REGISTRY:
                print(f"🔄 Fresh GRIB modification detected for {model.upper()}! Re-indexing grid architecture live...")
                try:
                    for old_key in list(MASTER_WEATHER_REGISTRY.keys()):
                        if old_key.startswith(cache_key): del MASTER_WEATHER_REGISTRY[old_key]
                    MASTER_WEATHER_REGISTRY[tracking_tag] = parse_and_regrid_grib(file_path)
                    MASTER_WEATHER_REGISTRY[cache_key] = MASTER_WEATHER_REGISTRY[tracking_tag]
                except Exception as e:
                    print(f"⚠️ Live hot-swap ingestion boundary crash: {e}")
                    
    if cache_key not in MASTER_WEATHER_REGISTRY:
        await load_individual_source_into_registry(model, source)
    if cache_key not in MASTER_WEATHER_REGISTRY:
        msg = f"No GRIB file available for {model}" if source == "grib" else f"Failed to load {model} from API"
        return JSONResponse(status_code=404, content={"error": msg})
        
    profile = MASTER_WEATHER_REGISTRY[cache_key]
    return {"header": profile["header"], "timestamps": profile["timestamps"]}

@app.get("/api/weather/frame")
async def get_frame(model: str = "ecmwf_ifs", source: str = "api", frame: int = 0):
    cache_key = f"{model}_{source}"
    if cache_key not in MASTER_WEATHER_REGISTRY:
        await load_individual_source_into_registry(model, source)
    if cache_key not in MASTER_WEATHER_REGISTRY:
        return JSONResponse(status_code=404, content={"error": "Data frame loading failure"})
        
    profile = MASTER_WEATHER_REGISTRY[cache_key]
    frames = profile["frames"]
    if frame >= len(frames) or frame < 0: frame = 0
    return frames[frame]

async def load_individual_source_into_registry(model: str, source: str):
    cache_key = f"{model}_{source}"
    config = MODEL_CONFIG.get(model)
    
    if not config: return
        
    if source == "grib":
        file_path = find_latest_grib(config)
        if not file_path:
            print(f"⚠️  No GRIB file found for {model} (prefix: {config['grib_prefix']})")
            return
        try:
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
    return {
        "header": {"la1": API_LAT_START, "lo1": API_LON_START, "dx": abs(API_LON_STEP), "dy": abs(API_LAT_STEP), "nx": nx, "ny": ny},
        "timestamps": timeline_timestamps,
        "frames": compiled_frames
    }

@app.get("/api/weather/point")
async def get_weather_point(lat: float, lon: float, model: str = "ecmwf_ifs", source: str = "grib"):
    """Extract a point time-series (TWS kt, TWD deg) from loaded GRIB or API grid."""
    cache_key = f"{model}_{source}"
    if cache_key not in MASTER_WEATHER_REGISTRY:
        await load_individual_source_into_registry(model, source)
    if cache_key not in MASTER_WEATHER_REGISTRY:
        return JSONResponse(status_code=404, content={"error": f"No data for {model}/{source}"})

    profile = MASTER_WEATHER_REGISTRY[cache_key]
    header, frames, timestamps = profile["header"], profile["frames"], profile["timestamps"]

    la1, lo1 = header["la1"], header["lo1"]
    dx, dy   = header["dx"],  header["dy"]
    nx, ny   = header["nx"],  header["ny"]

    col = int(round((lon - lo1) / dx))
    row = int(round((la1 - lat) / dy))
    col = max(0, min(nx - 1, col))
    row = max(0, min(ny - 1, row))

    result = []
    for i, ts in enumerate(timestamps):
        u = frames[i]["uData"][row][col]
        v = frames[i]["vData"][row][col]
        speed_ms = math.sqrt(u * u + v * v)
        tws_kt   = round(speed_ms * 1.94384, 1)
        twd      = round((math.degrees(math.atan2(-u, -v)) + 360) % 360, 0)
        gust_raw = frames[i].get("gustData", [[None]])[row][col] if frames[i].get("gustData") else None
        gust_kt  = round(gust_raw * 1.94384, 1) if gust_raw is not None else None
        result.append({"timestamp": ts, "tws": tws_kt, "twd": twd, "gust": gust_kt, "cape": None})

    return {"model": model, "source": source, "data": result}


@app.post("/api/weather/om")
async def get_weather_om(body: OmPointRequest):
    """Fetch point wind forecasts from Open-Meteo for one or more models."""
    results = {}
    async with httpx.AsyncClient() as client:
        tasks = {}
        for model_id in body.models:
            if model_id not in OM_POINT_MODELS:
                continue
            tasks[model_id] = client.get(
                "https://customer-api.open-meteo.com/v1/forecast",
                params={
                    "latitude":        body.lat,
                    "longitude":       body.lon,
                    "hourly":          "wind_speed_10m,wind_direction_10m,wind_gusts_10m,cape",
                    "wind_speed_unit": "kn",
                    "forecast_days":   7,
                    "models":          model_id,
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
                results[model_id] = {"source": "om", "data": point_data}
            except Exception as e:
                results[model_id] = {"error": str(e)}
    return results


@app.get("/api/weather/om-models")
def get_om_models():
    """Return available Open-Meteo models for the comparison table."""
    return OM_POINT_MODELS


def warm_up_local_grib_registry():
    print("\n🔥 Warming up local GRIB memory matrices via ADAPTIVE normalizations...")
    for model_id, config in MODEL_CONFIG.items():
        file_path = find_latest_grib(config)
        if file_path and os.path.exists(file_path):
            fname = os.path.basename(file_path)
            print(f"📦 Pre-loading {model_id.upper()} ({fname}) into memory layers...")
            try:
                cache_key = f"{model_id}_grib"
                MASTER_WEATHER_REGISTRY[cache_key] = parse_and_regrid_grib(file_path)
                print(f"✅ {model_id.upper()} successfully armed.")
            except Exception as e:
                print(f"⚠️ Failed to pre-load {fname}: {e}")
    print("🎉 Memory alignment complete. Server is active.\n")

if __name__ == "__main__":
    import uvicorn
    cleanup_old_gribs()
    warm_up_local_grib_registry()
    uvicorn.run(app, host="0.0.0.0", port=8000)