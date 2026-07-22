import httpx, uuid, json, time, subprocess, sys

proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "gpuopt.main:app", "--port", "8080"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
)
time.sleep(3)

def api(method, path, **kw):
    fn = getattr(httpx, method)
    r = fn(f"http://localhost:8080{path}", **kw, timeout=10)
    print(f"{method.upper()} {path} -> {r.status_code}")
    if r.status_code < 300:
        return r.json()
    print(r.text[:500])
    return None

print("=" * 60)
print("  GPUOpt Backend - Live Demo")
print("=" * 60)

print("\n1. HEALTH CHECK")
h = api("get", "/health/ready")
if h: print(f"   Status: {h['status']}")

print("\n2. GPU SNAPSHOT")
snap = api("get", "/api/v1/monitoring/gpu/snapshot")
if snap:
    for d in snap["devices"]:
        print(f"   GPU {d['index']}: {d['model']} - {d['memory_used_mb']}/{d['memory_total_mb']}MB "
              f"({d['utilization_gpu_percent']:.0f}% util, {d['temperature_celsius']:.0f}C)")
        for p in d["processes"]:
            print(f"     Process PID {p['pid']}: {p['process_name']} ({p['used_gpu_memory_mb']}MB)")

print("\n3. GPU START MONITOR")
api("post", "/api/v1/monitoring/gpu/start", json={"poll_interval": 5.0})

print("\n4. CREATE SLACK CHANNEL")
ch = api("post", "/api/v1/alerts/channels", json={
    "name": "slack-alerts", "channel_type": "slack",
    "config": {"webhook_url": "https://hooks.slack.com/test"}
})

print("\n5. CREATE PAGERDUTY CHANNEL")
api("post", "/api/v1/alerts/channels", json={
    "name": "pagerduty-prod", "channel_type": "pagerduty",
    "config": {"routing_key": "abc123"}
})

print("\n6. CREATE OPSGENIE CHANNEL")
api("post", "/api/v1/alerts/channels", json={
    "name": "opsgenie-alerts", "channel_type": "opsgenie",
    "config": {"api_key": "genie-key-123"}
})

print("\n7. LIST CHANNELS")
chs = api("get", "/api/v1/alerts/channels")
if chs: print(f"   {len(chs)} channels")

print("\n8. CREATE ALERT RULE")
cid = str(uuid.uuid4())
rule = api("post", "/api/v1/alerts/rules", json={
    "name": "High GPU Temperature", "cluster_id": cid,
    "condition_type": "gpu_temperature", "operator": "gt",
    "threshold": 85.0, "severity": "critical",
})
if rule: print(f"   Rule: {rule['name']} ({rule['severity']}, threshold={rule['threshold']})")

print("\n9. LIST ALERT RULES")
rules = api("get", f"/api/v1/alerts/rules?cluster_id={cid}")
if rules: print(f"   {len(rules)} rule(s)")

print("\n10. EVALUATE RULES")
evals = api("post", f"/api/v1/alerts/rules/{cid}/evaluate")
if evals: print(f"   {len(evals)} evaluation(s)")

print("\n11. LIST ALERT RECORDS")
alerts = api("get", "/api/v1/alerts/records")
if alerts: print(f"   {len(alerts)} alert(s)")

print("\n12. TEST CHANNEL NOTIFICATION")
if ch:
    test = api("post", f"/api/v1/alerts/channels/{ch['id']}/test")
    if test: print(f"   Status: {test['status']}, Error: {test.get('error','none')}")

print("\n13. PREEMPTION CYCLE")
actions = api("post", "/api/v1/monitoring/preemption/cycle")
if actions:
    for a in actions:
        print(f"   Preempt {a['workload_name']}/{a['namespace']} ({a['priority']}) -> {a['status']}")

print("\n14. PREEMPTION CONFIG")
cfg = api("get", "/api/v1/monitoring/preemption/config")
if cfg: print(f"   Policy: {cfg['policy']}, Delta: {cfg['min_priority_delta']}")

print("\n15. AUTOSCALER STATUS")
s = api("get", "/api/v1/monitoring/autoscaler/status")
if s: print(f"   Running: {s['running']}, Events: {s['event_count']}")

print("\n16. AUTOSCALER MANUAL SCALE")
ev = api("post", "/api/v1/monitoring/autoscaler/scale", json={
    "node_group": "gpu-group", "target_size": 5
})
if ev: print(f"   {ev['event']['direction']} to {ev['event']['target_size']} nodes ({ev['event']['status']})")

print("\n17. AUTOSCALER EVENTS")
events = api("get", "/api/v1/monitoring/autoscaler/events")
if events: print(f"   {len(events)} event(s)")

print("\n18. AUTOSCALER CONFIG")
acfg = api("get", "/api/v1/monitoring/autoscaler/config")
if acfg: print(f"   Policy: {acfg['policy']}, Min: {acfg['min_nodes']}, Max: {acfg['max_nodes']}")

print("\n" + "=" * 60)
print("  Demo complete - stopping server")
print("=" * 60)
proc.terminate()
proc.wait()
