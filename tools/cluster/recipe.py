#!/usr/bin/env python3
"""Implementation behind the six Bash entry points; only start/prepare/stop write remotely.

All topology is inherited from the sourced cluster.env. Optional keys in addition
 to deploy.py's settings: OUT (local build prefix), API_CONFIG, API_ENV_FILE,
API_INSTALLATION. No API credential is fetched to the launch machine: smoke reads
it only inside the head's Python process. Status/preflight need no credentials.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location("recipe_deploy", HERE / "deploy.py")
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


class RecipeFailure(RuntimeError):
    pass


def remote(host, code, data, *, input_text=None):
    """No shell wrapper, no credential in argv, no forwarding of local env secrets."""
    command = deploy.remote_python(code, json.dumps(data))
    if code in (SPLIT, ASSETS):
        command = "nice -n 10 ionice -c3 " + command
    result = subprocess.run(deploy.ssh_argv(host, command),
                            input=input_text, capture_output=True, text=True)
    if result.returncode:
        raise RecipeFailure(f"remote operation failed on {host} (exit {result.returncode}); check SSH, paths and prerequisites")
    return json.loads(result.stdout) if result.stdout.strip() else {}


def base_data():
    return dict(deploy.SETTINGS)


PREFLIGHT = '''import json,pathlib,shutil,subprocess,sys
x=json.loads(sys.argv[1]);problems=[]
os_release={}
for line in pathlib.Path('/etc/os-release').read_text().splitlines():
 if '=' in line:
  k,v=line.split('=',1);os_release[k]=v.strip('"')
dgx=pathlib.Path('/etc/dgx-release')
gpu=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],capture_output=True,text=True)
if gpu.returncode or not gpu.stdout.strip():problems.append('GPU/driver query failed')
if not dgx.exists():problems.append('/etc/dgx-release not found')
links={}
for device in x['TF_RDMA_DEVICES'].split(','):
 root=(pathlib.Path('/sys/class/infiniband')/device/'ports').resolve()
 if root==pathlib.Path('/') or not root.is_relative_to('/sys'):raise ValueError('invalid RoCE sysfs path')
 states=[p.read_text().strip() for p in root.glob('*/state')] if root.is_dir() else []
 links[device]=states
 if not states or not all('ACTIVE' in state for state in states):problems.append('RoCE link not active: '+device)
mem=next(line.split()[1] for line in pathlib.Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
root=pathlib.Path(x['SPARK_ROOT']).resolve()
while not root.exists() and root!=root.parent:root=root.parent
free=shutil.disk_usage(root).free
peers={}
if x['rank']==0:
 for host in x['SPARK_HOSTS'].split()[1:]:
  result=subprocess.run(['ssh','-oBatchMode=yes','-oStrictHostKeyChecking=yes','-oConnectTimeout=10','-i',x['HEAD_SSH_KEY'],x['SPARK_USER']+'@'+host,'true'],capture_output=True)
  peers[host]=result.returncode==0
  if result.returncode:problems.append('head-to-worker SSH failed: '+host)
print(json.dumps({'os':os_release.get('PRETTY_NAME','unknown'),'dgx_os':dgx.read_text().splitlines()[:2] if dgx.exists() else [],'gpu_driver':gpu.stdout.strip(),'roce':links,'mem_available_gib':round(int(mem)/1048576,2),'disk_free_gib':round(free/1024**3,2),'disk_checked_at':str(root),'head_ssh':peers,'problems':problems}))'''


STATUS = '''import json,pathlib,subprocess,sys,urllib.request
x=json.loads(sys.argv[1]);problems=[]
pattern=x['ENGINE_LABEL']+'-rank'+str(x['rank'])+'-g*.service'
r=subprocess.run(['systemctl','--user','list-units','--type=service','--all','--plain','--no-legend',pattern],capture_output=True,text=True)
# The description is the full ExecStart (and could contain environment data).
# Return only unit identity and states, never the description/command text.
units=[dict(zip(('unit','load','active','sub'),line.split()[:4])) for line in r.stdout.splitlines() if len(line.split())>=4]
if r.returncode:problems.append('could not inspect rank units')
active=[u['unit'] for u in units if u['active']=='active' and '-guard.' not in u['unit']]
guards=[u['unit'] for u in units if u['active']=='active' and '-guard.' in u['unit']]
if len(active)!=1:problems.append('expected one active rank unit')
if len(guards)!=1:problems.append('expected one active memory guard')
mem=next(line.split()[1] for line in pathlib.Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
result={'units':units,'mem_available_gib':round(int(mem)/1048576,2),'problems':problems}
if x['rank']==0:
 for name,port in [('engine',x['ENGINE_PORT']),('api',x['API_PORT'])]:
  host=x['ENGINE_HOST'] if name=='engine' else x['API_HOST']
  try:
   with urllib.request.urlopen('http://'+host+':'+port+'/health',timeout=10) as r:data=json.load(r)
   fields=('status','warming','numeric_effort') if name=='engine' else ('ok','version','max_concurrency','sampling','default_profile')
   result[name]={k:data.get(k) for k in fields}
   if name=='engine' and (data.get('status')!='ok' or data.get('warming') is not False or data.get('numeric_effort') is not True):problems.append('engine is not warm with numeric_effort')
   if name=='api' and data.get('ok') is not True:problems.append('API is not healthy')
  except (OSError,ValueError):
   result[name]={'reachable':False};problems.append(name+' health unavailable')
print(json.dumps(result))'''


SMOKE = '''import json,pathlib,stat,sys,time,urllib.request
x=json.loads(sys.argv[1]);config=pathlib.Path(x['API_CONFIG']);raw=json.loads(config.read_text());api=raw['api']
p=pathlib.Path(raw.get('env_file',x['API_ENV_FILE'])).expanduser()
if not p.is_absolute():p=config.parent/p
if p.is_symlink() or stat.S_IMODE(p.stat().st_mode)&0o077:raise ValueError('API credential file is not private')
values=dict(line.split('=',1) for line in p.read_text().splitlines() if line and not line.startswith('#') and '=' in line)
key_name=next(ref for name,ref in api.get('clients',{'default':'SPARK_API_KEY'}).items() if name not in api.get('admin_clients',[]))
key=values[key_name]
body={'model':api.get('model','deepseek-v4.1-flash'),'messages':[{'role':'system','content':'You are checking a model deployment with a harmless short test. Follow the user instruction exactly, without explaining your reasoning or adding any introduction, formatting, quotation marks or extra punctuation. This is a simple operational check of the serving interface, not a request to change any server settings. Return only the requested words.'},{'role':'user','content':'Say exactly: deployment ready'}],'max_tokens':32,'temperature':0,'reasoning_effort':'none','stream':False}
request=urllib.request.Request('http://'+x['API_HOST']+':'+x['API_PORT']+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
t0=time.monotonic()
with urllib.request.urlopen(request,timeout=120) as response:data=json.load(response)
elapsed=time.monotonic()-t0
message=data.get('choices',[{}])[0].get('message',{})
if not message.get('content'):raise ValueError('smoke response has no text')
tokens=data['usage']['completion_tokens']
if not tokens:raise ValueError('smoke response has no completion tokens')
print(json.dumps({'completion_tokens':tokens,'elapsed_seconds':round(elapsed,3),'tokens_per_second':round(tokens/elapsed,2),'rate_basis':'end-to-end wall time','ok':True}))'''


DOWNLOAD = '''import json,pathlib,shutil,subprocess,sys
x=json.loads(sys.argv[1]);root=pathlib.Path(x['SPARK_ROOT']).resolve();root.mkdir(parents=True,exist_ok=True)
weights=pathlib.Path(x['MODEL_DIR']);weights.mkdir(parents=True,exist_ok=True)
cli=shutil.which('hf') or shutil.which('huggingface-cli')
if not cli:raise RuntimeError('install the Hugging Face CLI on this Spark first')
# Authentication, if needed, stays on this Spark in the CLI's own credential store.
result=subprocess.run(['nice','-n','10','ionice','-c3',cli,'download',x['HF_REPO'],'--revision',x['HF_REVISION'],'--local-dir',str(weights)],stdout=sys.stderr)
if result.returncode:raise RuntimeError('weights download failed')
print(json.dumps({'downloaded':x['HF_REPO'],'revision':x['HF_REVISION']}))'''


SPLIT = '''import hashlib,json,pathlib,subprocess,sys,tempfile,time
x=json.loads(sys.argv[1]);started=time.monotonic()
def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while data:=f.read(8<<20):h.update(data)
 return h.hexdigest()
def tool(path,args,log):
 # -I excludes the downloaded folder and remote working directory from imports.
 bootstrap="import pathlib,runpy,sys;sys.argv=sys.argv[1:];sys.path.insert(0,str(pathlib.Path(sys.argv[0]).parent));runpy.run_path(sys.argv[0],run_name='__main__')"
 with log.open('a') as output:
  result=subprocess.run([x['SPARK_PYTHON'],'-I','-c',bootstrap,str(path),*map(str,args)],stdout=output,stderr=subprocess.STDOUT)
 if result.returncode:raise RuntimeError(path.name+' failed (exit '+str(result.returncode)+'); see '+str(log))
def split():
 root=pathlib.Path(x['MODEL_DIR']).resolve();cache=pathlib.Path(x['RANK_CACHE']).resolve()
 if cache==pathlib.Path('/'):raise ValueError('rank cache must not be the filesystem root')
 if cache==root or root in cache.parents or cache in root.parents:raise ValueError('RANK_CACHE and MODEL_DIR must be separate folders: the downloaded weights stay unmodified')
 rank=x['rank'];expected=x['expected_sha256'];prefix='rank'+str(rank)+'of3-'
 existing=list(cache.glob('rank*of3*.bin')) if cache.is_dir() else []
 if existing:
  if len(existing)!=1 or existing[0].is_symlink() or not existing[0].is_file() or not existing[0].name.startswith(prefix):
   raise ValueError('rank cache must contain only one regular '+prefix+'*.bin; fix rank placement before preparing')
  target=existing[0];actual=sha256(target)
  if actual!=expected:raise ValueError('rank '+str(rank)+' SHA-256 mismatch: expected '+expected+', got '+actual+' ('+str(target)+')')
  return {'ok':True,'status':'SKIPPED','path':str(target),'sha256':actual}
 cache.mkdir(parents=True,exist_ok=True)
 metadata=cache/'preparation';metadata.mkdir(exist_ok=True)
 work=pathlib.Path(tempfile.mkdtemp(prefix='rank'+str(rank)+'-',dir=metadata))
 plan=work/'plan.json';receipt=work/'source.receipt.json';log=work/'prepare.log';tools=pathlib.Path(x['tools_root'])
 native=tools/'engine/tensorfold/tools/dsv41-tp3'
 tool(tools/'tools/weights/make_plan.py',['--root',root,'--out',plan,'--arm',x['arm'],'--repo',x['HF_REPO'],'--revision',x['HF_REVISION']],log)
 tool(native/'source_verify.py',['--plan',plan,'--root',root,'--out',receipt],log)
 target=cache/(prefix+expected[:12]+'.bin')
 tool(native/'rank_cache.py',['--plan',plan,'--keys',root/'tp3'/('rank'+str(rank)+'.keys'),'--mia',root,'--pollard',root,'--source-receipt','mia='+str(receipt),'--out',target,'--write'],log)
 actual=sha256(target)
 if actual!=expected:raise ValueError('rank '+str(rank)+' SHA-256 mismatch: expected '+expected+', got '+actual+' ('+str(target)+')')
 return {'ok':True,'status':'WRITTEN','path':str(target),'sha256':actual,'metadata':str(work)}
try:result=split()
except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as error:
 result={'ok':False,'error':str(error)}
result['elapsed_seconds']=round(time.monotonic()-started,3)
print(json.dumps(result))'''


ASSETS = '''import json,pathlib,shutil,subprocess,sys
x=json.loads(sys.argv[1]);tools=pathlib.Path(x['tools_root'])
def tool(*args):
 result=subprocess.run([x['SPARK_PYTHON'],'-I',str(tools/'tools/assets/runtime_assets.py'),*map(str,args)],stdout=subprocess.PIPE,text=True)
 try:row=json.loads(result.stdout or '{}')
 except ValueError:row={}
 if result.returncode or row.get('ok') is not True:raise RuntimeError(row.get('error') or '; '.join(row.get('problems',[])[:3]) or args[0]+' step failed')
 return row
def hf(repo,revision,patterns,dest):
 cli=shutil.which('hf') or shutil.which('huggingface-cli')
 if not cli:raise RuntimeError('install the Hugging Face CLI on this Spark first')
 if subprocess.run(['nice','-n','10','ionice','-c3',cli,'download',repo,'--revision',revision,'--include',*patterns,'--local-dir',str(dest)],stdout=sys.stderr).returncode:raise RuntimeError('download from '+repo+' failed')
def assets():
 pins=json.loads((tools/'release/runtime-assets.json').read_text());kit,eng=pins['kit'],pins['engram']
 kit_dir,engram=pathlib.Path(x['KIT_DIR']),pathlib.Path(x['ENGRAM_DIR']);marker=engram/'SOURCE.json'
 need=(0 if kit_dir.exists() else x['kit_bytes'])+(0 if marker.is_file() else x['engram_bytes'])
 root=engram.resolve()
 while not root.exists() and root!=root.parent:root=root.parent
 if shutil.disk_usage(root).free<need:raise RuntimeError('runtime assets need '+str(round(need/1e9,1))+' GB free under '+str(root)+'; see docs/runtime-assets.md')
 download=pathlib.Path(x['SPARK_ROOT'])/'downloads'/('runtime-'+kit['revision'][:12])
 if not kit_dir.exists():hf(kit['repo'],kit['revision'],['runtime/*'],download)
 rows={'kit':tool('kit','--runtime',download,'--out',kit_dir)['status']}
 if x['ENGRAM_SOURCE']=='full' and not marker.is_file():hf(eng['repo'],eng['revision'],sorted(eng['files']),engram)
 rows['engram']=tool('engram','--rank',x['rank'],'--dir',engram,'--mode',x['ENGRAM_SOURCE'])['status']
 tool('check','--kit',kit_dir,'--engram',engram,'--rank',x['rank'])
 return rows
try:result={'ok':True,**assets()}
except (OSError,ValueError,KeyError,RuntimeError,subprocess.SubprocessError) as error:result={'ok':False,'error':str(error)}
print(json.dumps(result))'''


WEIGHT_TOOLS = ("tools/weights/make_plan.py", "tools/weights/unshard_ranks.py", *(
    "engine/tensorfold/tools/dsv41-tp3/" + name for name in
    ("rank_cache.py", "source_verify.py", "materialize_checkpoint.py", "tp3_layout.py", "io_budget.py")),
    "tools/assets/runtime_assets.py", "release/runtime-assets.json",
    "engine/tensorfold/zig/kernels/prefill_128/MANIFEST.json", "engine/tensorfold/zig/kernels/prefill_128/payload.tar.gz")


def inspect_cluster(code):
    failed = False
    for rank, host in deploy.SPARKS.items():
        row = remote(host, code, {**base_data(), "rank": rank})
        print(f"{host} rank {rank}: {json.dumps(row, sort_keys=True)}")
        failed |= bool(row.get("problems"))
    if failed:
        raise RecipeFailure("cluster checks reported problems")


def rsync(source, host, destination):
    transport = shlex.join(deploy.ssh_argv(host, "")[:-2])
    result = subprocess.run(["rsync", "-a", "--protect-args", "--exclude=__pycache__", "--exclude=.venv", "--exclude=*.egg-info",
                             "-e", transport, str(source), f"{deploy.SETTINGS['SPARK_USER']}@{host}:{destination}"], capture_output=True)
    if result.returncode:
        raise RecipeFailure(f"file staging failed on {host} (exit {result.returncode})")


def mkdirs(host, paths):
    remote(host, "import json,pathlib,sys; [pathlib.Path(p).mkdir(parents=True,exist_ok=True) for p in json.loads(sys.argv[1])]; print('{}')", paths)


def stage_engine():
    out = Path(os.environ.get("OUT", str(ROOT / "engine/build/out"))).expanduser().resolve()
    binaries = [out / "native/bin/tensorfold-native", out / "bin/tf-dsv41-lanes", out / "bin/tf-dsv41-check"]
    for path in binaries:
        if not path.is_file():
            raise RecipeFailure(f"build engine binaries first: missing {path}")
    # The grammar shared library is optional for a build that links it statically.
    if (out / "lib/libtfgrammar.so").is_file():
        binaries.append(out / "lib/libtfgrammar.so")
    s = deploy.SETTINGS
    for host in deploy.SPARKS.values():
        mkdirs(host, [s["BIN_DIR"], str(Path(s["GUARD_SCRIPT"]).parent), str(Path(s["LOGGER_SCRIPT"]).parent), s["SPARK_ROOT"] + "/logs", s["SPILL_DIR"]])
        for path in binaries:
            rsync(path, host, s["BIN_DIR"] + "/")
        rsync(ROOT / "serving/src/chimera_sparks/assets/memguard.py", host, s["GUARD_SCRIPT"])
        rsync(ROOT / "serving/src/chimera_sparks/assets/rotating_run.py", host, s["LOGGER_SCRIPT"])
        print(f"{host}: engine binaries, memory guard and rotating logger staged")


def stage_weight_tools(host):
    # Preserve repo-relative imports without copying the engine source tree.
    root = Path(deploy.SETTINGS["SPARK_ROOT"]) / "prepare-tools"
    mkdirs(host, sorted({str((root / name).parent) for name in WEIGHT_TOOLS}))
    for name in WEIGHT_TOOLS:
        rsync(ROOT / name, host, str(root / name))
    return str(root)


def asset_bytes(rank, mode):
    """Free space the kit (download plus staged copy) and this rank's Engram files need."""
    pins = json.loads((ROOT / "release/runtime-assets.json").read_text())
    kit = 2 * sum(pin["bytes"] for pin in pins["kit"]["files"].values())
    if mode == "full":
        return kit, sum(pin["bytes"] for pin in pins["engram"]["files"].values())
    files = pins["engram"]["ranks"][rank]["source_json"]["files"].values()
    return kit, sum(meta["header_bytes"] + sum(r["bytes"] for r in meta["ranges"]) for meta in files)


def prepare(manifest):
    s = deploy.SETTINGS
    if not s["HF_REPO"] or s["HF_REPO"].startswith("OPERATOR_") or not s["HF_REVISION"] or s["HF_REVISION"].startswith("REPLACE_"):
        raise RecipeFailure("set HF_REVISION to the pinned weights commit from HF_REPO (see cluster.env.example)")
    if s["ENGRAM_SOURCE"] not in ("ranged", "full"):
        raise RecipeFailure("ENGRAM_SOURCE must be ranged or full (see docs/runtime-assets.md)")
    expected = manifest["weights"]["rank_sha256"]
    if (not isinstance(expected, list) or len(expected) != 3
            or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in expected)):
        raise RecipeFailure("release manifest requires three rank SHA-256 values in rank order")
    for rank, host in deploy.SPARKS.items():
        # Kit and Engram first: a missing input or short disk fails before the weights download.
        tools_root = stage_weight_tools(host)
        kit_bytes, engram_bytes = asset_bytes(rank, s["ENGRAM_SOURCE"])
        row = remote(host, ASSETS, {**base_data(), "rank": rank, "tools_root": tools_root,
                                    "kit_bytes": kit_bytes, "engram_bytes": engram_bytes})
        if row.get("ok") is not True:
            raise RecipeFailure(f"{host}: runtime assets: {row.get('error', 'not confirmed')}; see docs/runtime-assets.md")
        print(f"{host}: kit {row['kit']}, rank {rank} Engram {row['engram']} ({s['ENGRAM_SOURCE']})")
        row = remote(host, DOWNLOAD, base_data())
        print(f"{host}: downloaded {row['downloaded']} at {row['revision']}")
        row = remote(host, SPLIT, {**base_data(), "rank": rank, "expected_sha256": expected[rank],
                                   "arm": manifest["weights"].get("artifact_arm", manifest["weights"]["arm"]), "tools_root": tools_root})
        if row.get("ok") is not True:
            raise RecipeFailure(f"{host}: {row.get('error', 'rank preparation was not confirmed')}")
        if row.get("sha256") != expected[rank]:
            raise RecipeFailure(f"{host}: rank {rank} SHA-256 mismatch in preparation result")
        print(f"{host}: rank {rank} {row['status']}: {row['path']}; SHA-256 {row['sha256']}; {row['elapsed_seconds']} s")
    stage_engine()


INSTALL_API = '''import json,pathlib,subprocess,sys
x=json.loads(sys.argv[1]);venv=pathlib.Path(x['venv'])
if not (venv/'bin/python').is_file():subprocess.run([x['SPARK_PYTHON'],'-m','venv',str(venv)],check=True,stdout=sys.stderr)
subprocess.run(['nice','-n','10','ionice','-c3',str(venv/'bin/python'),'-m','pip','install','--disable-pip-version-check',x['source']],check=True,stdout=sys.stderr)
print('{}')'''


WARMUP = '''import json,math,sys,time,urllib.request
x=json.loads(sys.argv[1]);rows=[]
for prompt in ('Say hello briefly.','What is 2 + 2?','Name one color.'):
 body={'model':x['model'],'messages':[{'role':'user','content':prompt}],'max_tokens':16,'temperature':0,'reasoning_effort':'none','chat_template_kwargs':{'enable_thinking':False},'stream':False}
 request=urllib.request.Request('http://'+x['ENGINE_HOST']+':'+x['ENGINE_PORT']+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 t0=time.monotonic()
 with urllib.request.urlopen(request,timeout=120) as response:data=json.load(response)
 elapsed=time.monotonic()-t0;tokens=data['usage']['completion_tokens']
 if not tokens or elapsed<=0:raise ValueError('warmup response has no completion tokens')
 rate=data.get('tensorfold',{}).get('tokens_per_second')
 native=isinstance(rate,(int,float)) and not isinstance(rate,bool) and math.isfinite(rate) and rate>0
 rows.append({'completion_tokens':tokens,'elapsed_seconds':round(elapsed,3),'tokens_per_second':round(rate if native else tokens/elapsed,2),'rate_basis':'engine decode' if native else 'end-to-end wall time'})
print(json.dumps({'warmups':rows}))'''


def head_api(action, manifest, venv):
    """Only staging forwards optional keys, via stdin; never in SSH argv/logs."""
    payload = {"action": action, "settings": deploy.SETTINGS, "manifest": manifest}
    if action == "stage":
        payload.update(key=os.environ.get("SPARK_API_KEY", ""), admin_key=os.environ.get("SPARK_ADMIN_KEY", ""))
    command = shlex.join([venv + "/bin/python", "-I", "-c", (HERE / "head_api.py").read_text()])
    try:
        result = subprocess.run(deploy.ssh_argv(deploy.SPARKS[0], command), input=json.dumps(payload), capture_output=True, text=True)
    except (OSError, subprocess.SubprocessError):
        return {"ok": False, "ready": False, "error": "API operation not confirmed; check the head user journal"}
    try:
        row = json.loads(result.stdout)
        if not isinstance(row, dict):
            raise ValueError("not an API result")
    except ValueError:
        row = {"ok": False, "error": "API operation not confirmed; check the head user journal"}
    if result.returncode:
        row["ok"] = False
    return row


def start(manifest_path, manifest):
    # Install and validate pending API files before any rank can be stopped.
    # head_api.start publishes them only when replacing the API, so an engine
    # preflight/deployment failure leaves the live config and credentials intact.
    s, head = deploy.SETTINGS, deploy.SPARKS[0]
    source = s["SPARK_ROOT"] + "/serving-source"
    venv = s["SPARK_ROOT"] + "/api/venv-" + manifest["serving"]["version"]
    mkdirs(head, [source, s["SPARK_ROOT"] + "/api"])
    rsync(str(ROOT / "serving") + "/", head, source + "/")
    remote(head, INSTALL_API, {**s, "venv": venv, "source": source})
    staged = head_api("stage", manifest, venv)
    if staged.get("ok") is not True or staged.get("staged") is not True:
        raise RecipeFailure("API staging failed before ranks were changed; check the head API config and private credentials")
    # A rank-ready receipt is not an overall successful start. Finalize only
    # after API readiness and the three direct-engine warmups have succeeded.
    code = deploy.main(["--manifest", str(manifest_path)], ready_status="engine-ready")
    if code not in (0, 2):
        raise RecipeFailure(f"engine deployment exited {code}")
    receipt = json.loads(deploy.RECEIPT.read_text())
    result = {"affinity": receipt["affinity"], "engine_status": "ready", "ranks_status": "left-running"}
    phase = "api"
    try:
        api = head_api("start", manifest, venv)
        result["api"] = api
        if api.get("ok") is not True or api.get("ready") is not True:
            cleanup = "API unit stopped" if api.get("stopped") is True else "API stop not confirmed"
            raise RecipeFailure("API start/readiness failed; " + cleanup + "; engine ranks left running")
        print(f"API {api['unit']} is ready (serving {api['version']})")
        phase = "warmup"
        row = remote(head, WARMUP, {**s, "model": manifest["serving"]["model"]})
        result["warmups"] = row["warmups"]
        for index, warmup in enumerate(row["warmups"], 1):
            print(f"warmup {index}/3: {warmup['completion_tokens']} tokens; {warmup['tokens_per_second']} tokens/s ({warmup['rate_basis']})")
        if code == 2:
            phase = "affinity"
            raise RecipeFailure("engine and API are warm, but configured optional affinity controller affinity was not confirmed; API and ranks left running")
    except BaseException as error:
        result.update(status="failed", phase=phase, error=str(error) or type(error).__name__)
        try:
            deploy.write_execution_receipt(receipt["execution"], receipt["units"], **result)
        except OSError:
            print("PROBLEM: could not record the failed start receipt", file=sys.stderr)
        if phase == "warmup":
            raise RecipeFailure("direct-engine warmup failed; API and engine ranks left running") from None
        raise
    result.update(status="ready", ranks_status="running")
    deploy.write_execution_receipt(receipt["execution"], receipt["units"], **result)


def stop():
    # No wildcard can stop another recipe: old_units validates exact generated
    # rank/guard addresses. Stop the API first, then the guarded engine ranks.
    unit = deploy.SETTINGS["API_INSTALLATION"] + "-api"
    code = '''import json,subprocess,sys
x=json.loads(sys.argv[1]);unit=x['unit']
load=subprocess.run(['systemctl','--user','show',unit,'-p','LoadState','--value'],capture_output=True,text=True)
if load.stdout.strip()!='not-found':subprocess.run(['systemctl','--user','stop',unit],check=True,capture_output=True)
print('{}')'''
    remote(deploy.SPARKS[0], code, {"unit": unit})
    for rank, host in deploy.SPARKS.items():
        units = deploy.old_units(host)
        if units:
            deploy.run_ssh(host, shlex.join(["systemctl", "--user", "stop", *units]))
        print(f"{host}: rank {rank} and its guards stopped")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "prepare", "start", "status", "smoke-test", "stop"))
    parser.add_argument("--manifest", type=Path, default=deploy.DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    deploy.parse_args(["--manifest", str(args.manifest)])
    manifest = json.loads(args.manifest.read_text())
    if args.action == "preflight":
        inspect_cluster(PREFLIGHT)
    elif args.action == "status":
        inspect_cluster(STATUS)
    elif args.action == "smoke-test":
        row = remote(deploy.SPARKS[0], SMOKE, base_data())
        print(f"smoke: {row['completion_tokens']} tokens in {row['elapsed_seconds']} s; {row['tokens_per_second']} tokens/s (end-to-end)")
    elif args.action == "prepare":
        prepare(manifest)
    elif args.action == "start":
        start(args.manifest, manifest)
    else:
        stop()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"FAIL: {sys.argv[1] if len(sys.argv)>1 else 'recipe'}: {error}", file=sys.stderr)
        sys.exit(1)
