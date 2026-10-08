"""Public library metadata plus a resumable exact-board cache; standard library only.

The public SHA convention is unknown. Never equate it to the local digest.
is_known returns None until the candidate's score tier is fully cached.
"""
import gzip
import hashlib
import json
import numbers
import os
import re
import sqlite3
import struct
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

API = 'https://eternity-control-plane-prod.eternity-cp.workers.dev'
CLUES = {34:831,45:1019,135:554,210:723,221:992}
SHA_RE = re.compile(r'^[0-9a-f]{64}$')
EDGES = [(i,i+1,1,3) for i in range(256) if i%16<15]+[(i,i+16,2,0) for i in range(240)]

def board_bytes(board):
    board=list(board)
    if any(not isinstance(x,numbers.Integral) or isinstance(x,bool) for x in board):
        raise ValueError('Placement codes must be integers')
    board = [int(x) for x in board]
    if len(board)!=256 or any(x<0 or x>1023 for x in board):
        raise ValueError('Expected 256 placement codes in 0..1023')
    if sorted(x//4 for x in board)!=list(range(256)):
        raise ValueError('Duplicate or missing piece')
    if any(board[c]!=v for c,v in CLUES.items()):
        raise ValueError('Five fixed clues differ')
    return struct.pack('<256H',*board)

def local_board_hash(board):
    return hashlib.sha256(board_bytes(board)).hexdigest()

class LibraryMonitor:
    def __init__(self, data_dir, interval_seconds=900, *, pieces_path=None,
                 request_timeout=20, detail_interval_seconds=1.0):
        self.data_dir=Path(data_dir);self.data_dir.mkdir(parents=True,exist_ok=True)
        self.interval_seconds=max(1,float(interval_seconds))
        self.request_timeout=float(request_timeout)
        self.detail_interval_seconds=max(1.0,float(detail_interval_seconds))
        self._lock=threading.RLock();self._poll_lock=threading.Lock()
        self._stop=threading.Event();self._thread=None
        self._db=sqlite3.connect(self.data_dir/'library.sqlite3',check_same_thread=False)
        self._db.execute('PRAGMA journal_mode=WAL')
        self._db.executescript('''
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS boards(
          public_sha TEXT PRIMARY KEY,score INTEGER NOT NULL,has_content INTEGER NOT NULL,
          active INTEGER NOT NULL DEFAULT 0,local_sha TEXT,placement BLOB,
          retry_after REAL NOT NULL DEFAULT 0,failures INTEGER NOT NULL DEFAULT 0);
        CREATE INDEX IF NOT EXISTS board_lookup ON boards(local_sha);
        CREATE INDEX IF NOT EXISTS board_pending ON boards(active,has_content,local_sha,score);
        ''');self._db.commit()
        self._known={r[0] for r in self._db.execute('SELECT local_sha FROM boards WHERE local_sha IS NOT NULL')}
        self._faces=None;self._last_error=None;self._requests=0;self._detail_downloads=0
        stored=self._meta('canonical_faces')
        if stored:self._set_base(json.loads(stored))
        if pieces_path:
            base=[]
            for line in Path(pieces_path).read_text().splitlines():
                u,d,l,r=map(int,line.split());base.append([u,r,d,l])
            if len(base)!=256:raise ValueError('Expected 256 canonical pieces')
            # Do not replace an existing letter-based palette: both are valid labels.
            if self._faces is None:self._set_base(base,persist=True)
        bootstrap=self.data_dir/'library-index.json'
        if not self._meta('index_checked_at') and bootstrap.is_file():
            self.ingest_index(json.loads(bootstrap.read_text(encoding='utf-8')))

    def _meta(self,key):
        row=self._db.execute('SELECT value FROM metadata WHERE key=?',(key,)).fetchone()
        return row[0] if row else None

    def _put_meta(self,key,value):
        self._db.execute('INSERT INTO metadata VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',(key,str(value)))

    def _set_base(self,base,persist=False):
        self._faces=[tuple(p[(side-rot)%4] for side in range(4)) for p in base for rot in range(4)]
        if persist:
            self._put_meta('canonical_faces',json.dumps(base,separators=(',',':')));self._db.commit()

    def _request(self,url,headers=None):
        headers={'Accept-Encoding':'gzip','User-Agent':'EternityIISolver-library-monitor/1.0',**(headers or {})}
        req=urllib.request.Request(url,headers=headers)
        try:
            with urllib.request.urlopen(req,timeout=self.request_timeout) as response:
                data=response.read(80*1024*1024+1)
                if len(data)>80*1024*1024:raise ValueError('Compressed response size limit')
                return response.status,dict(response.headers),data
        except urllib.error.HTTPError as exc:
            if exc.code==304:return 304,dict(exc.headers),b''
            raise

    @staticmethod
    def _decode(headers,data):
        headers={k.lower():v for k,v in headers.items()}
        if headers.get('content-encoding','').lower()=='gzip':data=gzip.decompress(data)
        if len(data)>160*1024*1024:raise ValueError('Decoded response size limit')
        return json.loads(data)

    @staticmethod
    def _validate_index(doc):
        if not isinstance(doc,dict) or doc.get('schema')!='eternity2-board-library-index/v2':
            raise ValueError('Unsupported library index schema')
        rows=doc.get('boards')
        if not isinstance(rows,list) or doc.get('count')!=len(rows):raise ValueError('Index count mismatch')
        seen=set();parsed=[]
        for row in rows:
            sha=row.get('sha');score=row.get('score');content=row.get('has_content')
            if not isinstance(sha,str) or not SHA_RE.fullmatch(sha) or sha in seen:raise ValueError('Invalid/duplicate public SHA')
            if type(score)!=int or not 0<=score<=480 or type(content)!=bool:raise ValueError('Invalid index score/content')
            seen.add(sha);parsed.append((sha,score,int(content)))
        return parsed

    def ingest_index(self,doc,headers=None):
        """Load an existing verified public index without downloading it again."""
        parsed=self._validate_index(doc);headers={k.lower():v for k,v in (headers or {}).items()}
        payload=gzip.compress(json.dumps(doc,separators=(',',':')).encode(),compresslevel=1)
        temporary=self.data_dir/('index.'+uuid.uuid4().hex+'.tmp')
        temporary.write_bytes(payload);os.replace(temporary,self.data_dir/'index.json.gz')
        with self._lock,self._db:
            self._db.execute('UPDATE boards SET active=0')
            self._db.executemany('''INSERT INTO boards(public_sha,score,has_content,active) VALUES(?,?,?,1)
                ON CONFLICT(public_sha) DO UPDATE SET score=excluded.score,has_content=excluded.has_content,active=1''',parsed)
            self._put_meta('index_checked_at',time.time());self._put_meta('index_generated_at',doc.get('generated_at',''))
            self._put_meta('etag',headers.get('etag',''));self._put_meta('last_modified',headers.get('last-modified',''))
            self._last_error=None
        return len(parsed)

    def poll_once(self):
        """Refresh all metadata; retain the previous snapshot on network/schema errors."""
        with self._poll_lock:
            try:
                with self._lock:
                    headers={};etag=self._meta('etag');modified=self._meta('last_modified')
                    if etag:headers['If-None-Match']=etag
                    if modified:headers['If-Modified-Since']=modified
                status,h,data=self._request(API+'/library/index',headers)
                with self._lock:self._requests+=1
                if status==304:
                    with self._lock,self._db:self._put_meta('index_checked_at',time.time());self._last_error=None
                elif status==200:self.ingest_index(self._decode(h,data),h)
                else:raise ValueError('Unexpected HTTP status '+str(status))
            except Exception as exc:
                with self._lock:self._last_error=f'{type(exc).__name__}: {exc}'
            return self.status()

    def _score(self,board):
        board_bytes(board)
        if self._faces is None:return None
        for cell,code in enumerate(board):
            if not all((color==0)==outer for color,outer in zip(self._faces[code],
                [cell<16,cell%16==15,cell>=240,cell%16==0])):raise ValueError('Invalid gray frame')
        return sum(self._faces[board[a]][da]==self._faces[board[b]][db] for a,b,da,db in EDGES)

    def register_known_document(self,doc,public_sha):
        """Seed/reuse an already downloaded public board, checking geometry and score."""
        if not SHA_RE.fullmatch(public_sha):raise ValueError('Invalid public SHA')
        board=doc['board'];blob=board_bytes(board);letters=doc['board_edges']
        if len(letters)!=1024 or any(c<'a' or c>'w' for c in letters):raise ValueError('Invalid edge encoding')
        with self._lock:
            if self._faces is None:
                base=[None]*256
                for cell,code in enumerate(board):
                    f=[ord(c)-97 for c in letters[4*cell:4*cell+4]]
                    base[code//4]=[f[(side+code%4)%4] for side in range(4)]
                self._set_base(base,persist=True)
            score=self._score(board)
            independent=sum(letters[4*a+da]==letters[4*b+db] for a,b,da,db in EDGES)
            if score!=independent or score!=doc['score']:raise ValueError('Board score mismatch')
            mapping={}
            for cell,code in enumerate(board):
                if int(doc['board_pieces'][3*cell:3*cell+3])!=code//4+1:raise ValueError('Piece encoding mismatch')
                for color,letter in zip(self._faces[code],letters[4*cell:4*cell+4]):
                    if color in mapping and mapping[color]!=letter:raise ValueError('Edge encoding mismatch')
                    mapping[color]=letter
            if len(mapping)!=23 or len(set(mapping.values()))!=23 or mapping[0]!='a':raise ValueError('Palette mismatch')
            digest=hashlib.sha256(blob).hexdigest()
            with self._db:
                existing=self._db.execute('SELECT score FROM boards WHERE public_sha=?',(public_sha,)).fetchone()
                if existing and existing[0]!=score:raise ValueError('Index/document score mismatch')
                self._db.execute('''INSERT INTO boards(public_sha,score,has_content,active,local_sha,placement) VALUES(?,?,1,0,?,?)
                    ON CONFLICT(public_sha) DO UPDATE SET local_sha=excluded.local_sha,placement=excluded.placement,retry_after=0,failures=0''',
                    (public_sha,score,digest,blob))
            self._known.add(digest)
        return digest

    def warm_once(self):
        """Fetch at most one missing geometry; background loop limits request rate."""
        with self._lock:
            row=self._db.execute('''SELECT public_sha FROM boards WHERE active=1 AND has_content=1
                AND local_sha IS NULL AND retry_after<=? ORDER BY score DESC,public_sha LIMIT 1''',(time.time(),)).fetchone()
        if not row:return False
        sha=row[0]
        try:
            status,headers,data=self._request(API+'/library/board/'+sha)
            with self._lock:self._requests+=1
            if status!=200:raise ValueError('Unexpected document status '+str(status))
            self.register_known_document(self._decode(headers,data),sha)
            with self._lock:self._detail_downloads+=1
            return True
        except Exception as exc:
            with self._lock,self._db:
                self._last_error=f'{type(exc).__name__}: {exc}'
                self._db.execute('UPDATE boards SET failures=failures+1,retry_after=? WHERE public_sha=?',(time.time()+900,sha))
            return False

    def is_known(self,board):
        """True=exact public duplicate; False=absent in complete score tier; None=unknown."""
        digest=local_board_hash(board)
        with self._lock:
            if digest in self._known:return True
            if not self._meta('index_checked_at'):return None
            score=self._score(board)
            if score is None:return None
            missing=self._db.execute('SELECT COUNT(*) FROM boards WHERE active=1 AND score=? AND local_sha IS NULL',(score,)).fetchone()[0]
            return None if missing else False

    def status(self):
        with self._lock:
            total,cached=self._db.execute('SELECT COUNT(*),SUM(local_sha IS NOT NULL) FROM boards WHERE active=1').fetchone()
            tiers={str(score):{'indexed':count,'cached':int(got or 0)} for score,count,got in self._db.execute(
                'SELECT score,COUNT(*),SUM(local_sha IS NOT NULL) FROM boards WHERE active=1 GROUP BY score ORDER BY score DESC')}
            checked=self._meta('index_checked_at')
            return {'schema_version':1,'running':bool(self._thread and self._thread.is_alive()),'indexed_boards':total,
                'cached_geometries':int(cached or 0),'known_exact_boards':len(self._known),'tiers':tiers,
                'pending_geometries':total-int(cached or 0),
                'high_score_indexed':sum(v['indexed'] for s,v in tiers.items() if int(s)>=465),
                'high_score_cached':sum(v['cached'] for s,v in tiers.items() if int(s)>=465),
                'highest_indexed_score':max(map(int,tiers),default=None),
                'full_metadata_loaded':checked is not None,'geometry_cache_complete':bool(checked and total==int(cached or 0)),
                'last_index_check_unix':float(checked) if checked else None,'index_generated_at':self._meta('index_generated_at'),
                'poll_interval_seconds':self.interval_seconds,'detail_interval_seconds':self.detail_interval_seconds,
                'last_error':self._last_error,'network_requests_this_session':self._requests,
                'detail_downloads_this_session':self._detail_downloads,
                'public_hash_formula':'unverified; lookup uses exact downloaded placements and local SHA256 uint16 little endian',
                'novelty_scope':'False means absent from cached current index snapshot, not from future publications.'}

    def _loop(self):
        next_poll=0
        while not self._stop.is_set():
            started=time.monotonic()
            if started>=next_poll:
                self.poll_once();next_poll=time.monotonic()+self.interval_seconds
            if self._stop.is_set():break
            self.warm_once()
            self._stop.wait(max(0,self.detail_interval_seconds-(time.monotonic()-started)))

    def start(self):
        with self._lock:
            if self._thread and self._thread.is_alive():return
            self._stop.clear();self._thread=threading.Thread(target=self._loop,name='eternity-library-monitor',daemon=True);self._thread.start()

    def stop(self,timeout=25):
        self._stop.set()
        if self._thread:self._thread.join(timeout)
        return not (self._thread and self._thread.is_alive())
