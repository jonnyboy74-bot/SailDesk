# SailDesk — Server Commands

## Directory
Always run from the project folder:
```bash
cd /Users/jonnymalbon/WeatherLab/SailDesk
```

---

## Start / Build

**First time or after code changes:**
```bash
docker compose up --build -d
```

**Start without rebuilding (just restart):**
```bash
docker compose up -d
```

---

## Stop

```bash
docker compose down
```

---

## Restart

**Quick restart (no rebuild):**
```bash
docker compose restart
```

**Full rebuild + restart (after code changes):**
```bash
docker compose up --build -d
```

---

## Logs

**Follow live logs:**
```bash
docker logs saildesk -f
```

**Last 50 lines:**
```bash
docker logs saildesk --tail 50
```

**Stop watching logs:** `Ctrl+C` (server keeps running)

---

## Status

**Is it running?**
```bash
docker ps
```

**Full container details:**
```bash
docker inspect saildesk
```

---

## Access

| Location | URL |
|----------|-----|
| Local | http://localhost:8000/weather |
| Tailscale | http://100.76.96.55:8000/weather |
| Wind Map | http://100.76.96.55:8000 |

**Find Tailscale IP (if it changes):**
```bash
tailscale ip -4
```

---

## GRIBs

GRIBs live at `~/Desktop/gribs` — drop new `.grb2` files there.  
Server auto-detects and reloads within **2 minutes**. Old files are cleaned up automatically.

**Check what's loaded:**
```bash
docker logs saildesk | grep "✅"
```

---

## Git / Backup

```bash
cd /Users/jonnymalbon/WeatherLab/SailDesk
git add -A
git commit -m "your message"
git push
```
