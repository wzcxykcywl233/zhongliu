"""Transport-only repair. Original audit and training files remain unchanged."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request

SOURCE=Path('/original/dataset.py')
if not SOURCE.exists(): SOURCE=Path(__file__).resolve().parents[1]/'lld-mixed-domain-v1/dataset.py'
spec=importlib.util.spec_from_file_location('lld_original_data',SOURCE)
D=importlib.util.module_from_spec(spec); spec.loader.exec_module(D)
OFFICIAL='https://huggingface.co'
ENDPOINTS=(OFFICIAL,'https://hf-mirror.com')
# Computed from the fixed OFFICIAL API response, not from the mirror.
# Canonical encoding of the original D.catalog() output (7970 files).
CATALOG_SHA256='5a16960e0d9e62108dd99d6362aeaa439fee943b10299844cb0e3ee710bb55e7'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()


def verify_inventory(value):
    if digest(value)!=CATALOG_SHA256:
        raise ValueError('Catalog differs from pinned official inventory; no download permitted')
    return value


def error_kind(exc):
    if isinstance(exc,NetworkFailure): return exc.kind
    if isinstance(exc,urllib.error.HTTPError): return 'http_'+str(exc.code)
    reason=getattr(exc,'reason',exc)
    if isinstance(reason,ssl.SSLError): return 'tls_verification_or_handshake'
    if getattr(reason,'errno',None) in (111,61,10061): return 'connection_refused'
    if isinstance(reason,TimeoutError): return 'timeout'
    return type(reason).__name__


class NetworkFailure(ConnectionError):
    def __init__(self,kind):
        self.kind=kind
        super().__init__('network_'+kind+'; check selected endpoint / container proxy')


class Transport:
    def __init__(self,endpoint,mode='inherit',proxy_url=''):
        if endpoint not in ENDPOINTS: raise ValueError('Choose an allowed HTTPS download endpoint')
        if mode not in ('inherit','direct','explicit'): raise ValueError('Unknown proxy mode')
        self.endpoint=endpoint; self.mode=mode
        if mode=='explicit':
            p=urllib.parse.urlsplit(proxy_url)
            if p.scheme not in ('http','https') or not p.hostname or p.username or p.password or p.path not in ('','/') or p.query or p.fragment:
                raise ValueError('Explicit proxy needs an HTTP(S) origin, without embedded credentials')
            if p.hostname.lower() in ('localhost','127.0.0.1','::1'):
                raise ValueError('Container localhost is not Windows; use host.docker.internal with your actual proxy port')
            # Only this process is changed; explicit mode must not be bypassed by stale NO_PROXY.
            for name in ('no_proxy','NO_PROXY'): os.environ.pop(name,None)
            handler=urllib.request.ProxyHandler({'http':proxy_url,'https':proxy_url})
        elif proxy_url:
            raise ValueError('Proxy URL requires explicit mode')
        elif mode=='direct': handler=urllib.request.ProxyHandler({})
        else: handler=urllib.request.ProxyHandler()
        self.opener=urllib.request.build_opener(handler)

    def __call__(self,request,timeout=30):
        if isinstance(request,str): request=urllib.request.Request(request)
        url=request.full_url
        if not url.startswith(OFFICIAL+'/'): raise ValueError('Unexpected source URL')
        if request.get_method()!='GET' or request.data is not None: raise ValueError('Only public GET downloads allowed')
        if any(k.lower() in ('authorization','cookie','proxy-authorization') for k in request.headers):
            raise ValueError('Do not send credentials to a download endpoint')
        url=self.endpoint+url[len(OFFICIAL):]
        req=urllib.request.Request(url,headers=dict(request.header_items()))
        try: return self.opener.open(req,timeout=timeout)
        except (OSError,urllib.error.URLError) as exc:
            # Do not print proxy credentials, signed redirect URLs, or environment values.
            raise NetworkFailure(error_kind(exc)) from None


def metadata(transport):
    url=OFFICIAL+'/api/datasets/'+D.REPO+'/revision/'+D.REVISION+'?blobs=true'
    for attempt in range(3):
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'MedCoTracker-research/1'})
            with transport(request,timeout=30) as response:
                raw=response.read(8*2**20+1)
            if len(raw)>8*2**20: raise ValueError('Metadata response too large')
            # Integrity failures do NOT retry/fall back or silently accept another revision.
            return verify_inventory(D.catalog(json.loads(raw)))
        except (ConnectionError,OSError) as exc:
            if attempt==2:
                raise ConnectionError('Metadata unreachable after 3 attempts ('+error_kind(exc)+'). Try the other endpoint, '
                    'direct mode, or a working container-accessible proxy. No data/training started.') from None
            print(f'NETWORK RETRY metadata {attempt+1}/3 {error_kind(exc)}',flush=True)
            time.sleep(2*(attempt+1))


def network(transport,output):
    inv=metadata(transport)
    # Metadata availability alone does not prove file/CDN availability.
    url=OFFICIAL+'/datasets/'+D.REPO+'/resolve/'+D.REVISION+'/README.md?download=true'
    with transport(url,timeout=30) as response: raw=response.read(65537)
    entry=inv['files']['README.md']
    h=hashlib.sha1(('blob '+str(len(raw))+'\0').encode()+raw).hexdigest()
    if len(raw)!=entry['size'] or h!=entry['git_blob']: raise ValueError('README transport probe checksum mismatch')
    result=dict(complete=True,endpoint=transport.endpoint,proxy_mode=transport.mode,
        revision=D.REVISION,catalog_sha256=CATALOG_SHA256,metadata_and_small_file_verified=True,
        note='Small-file probe only, not a guarantee of all large-file/CDN connectivity; no image download/training')
    D.save(output/'network.json',result)
    print('NETWORK VERIFIED '+json.dumps(result),flush=True)


def download(root,accepted,transport):
    if not accepted: raise ValueError('Explicit research terms acceptance required')
    root.mkdir(parents=True,exist_ok=True)
    # Identical to v1: existing receipts and successful partial downloads are retained.
    D.freeze(root/'research-terms.json',dict(accepted=True,scope='noncommercial research; no external redistribution; '
        'same-organization single-site internal use only',dataset_card=D.CARD,original_terms=D.TERMS,revision=D.REVISION))
    lock=root/'download-manifest.json'
    inv=verify_inventory(D.read(lock)) if lock.exists() else metadata(transport)
    D.freeze(lock,inv)
    print(f'CATALOG VERIFIED files={len(inv["files"])}; original revision and checksums unchanged',flush=True)
    for i,(name,entry) in enumerate(inv['files'].items()):
        D.fetch_file(root,name,entry,opener=transport)
        if i%25==0: print(f'DOWNLOAD VERIFIED {i+1}/{len(inv["files"])} {name}',flush=True)
    D.save(root/'download-complete.json',dict(complete=True,manifest_sha256=D.sha(lock),
        files=len(inv['files']),bytes=sum(e['size'] for e in inv['files'].values())))
    print('DOWNLOAD COMPLETE; original audit may proceed; no training started',flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['network','download'])
    p.add_argument('--data',type=Path,default=Path('/data')); p.add_argument('--output',type=Path,default=Path('/output'))
    p.add_argument('--endpoint',choices=ENDPOINTS,default=OFFICIAL)
    p.add_argument('--proxy-mode',choices=['inherit','direct','explicit'],default='inherit')
    p.add_argument('--proxy-url',default=''); p.add_argument('--accept-research-terms',action='store_true')
    a=p.parse_args()
    keys=sorted(k for k in os.environ if k.lower() in ('http_proxy','https_proxy','all_proxy','no_proxy'))
    loopback=[]
    for scheme,value in urllib.request.getproxies().items():
        if scheme in ('http','https','all'):
            try:
                if urllib.parse.urlsplit(value).hostname in ('localhost','127.0.0.1','::1'): loopback.append(scheme)
            except ValueError: pass
    print('NETWORK CONFIG '+json.dumps(dict(endpoint=a.endpoint,proxy_mode=a.proxy_mode,
        proxy_environment_keys=keys,loopback_proxy_schemes=sorted(loopback),proxy_values_redacted=True)),flush=True)
    transport=Transport(a.endpoint,a.proxy_mode,a.proxy_url)
    if a.stage=='network': network(transport,a.output)
    else: download(a.data,a.accept_research_terms,transport)


if __name__=='__main__': main()
