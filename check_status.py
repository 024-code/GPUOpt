import urllib.request, json

r = urllib.request.urlopen('http://localhost:8080/', timeout=5)
print(f'Frontend: {r.status} ({len(r.read())} bytes)')

r = urllib.request.urlopen('http://localhost:8080/api/v1/rtx/status', timeout=5)
s = json.loads(r.read())
g = s['gpus'][0]
print(f'GPU: {g["name"]}')
print(f'VRAM: {g["memory_total_gb"]} GB total, {g["memory_free_gb"]} GB free')
print(f'CPU: {s["cpu"]["cores"]} cores @ {s["cpu"]["usage_percent"]}%')
print(f'RAM: {s["memory"]["total_gb"]} GB total, {s["memory"]["free_gb"]} GB free')

r = urllib.request.urlopen('http://localhost:8080/api/v1/rtx/jobs', timeout=5)
j = json.loads(r.read())
print(f'Jobs submitted: {j["total_jobs"]}')
for job in j['jobs']:
    print(f'  [{job["status"]}] {job["job_id"]}: {job["name"]} -> GPU {job["assigned_gpu"]}')

print()
print('=' * 50)
print('OPEN http://localhost:8080 IN YOUR BROWSER')
print('Click "RTX 4090" in the sidebar to see your GPU')
print('=' * 50)
