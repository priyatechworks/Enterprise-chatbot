import json
import sqlite3
import uuid
from app.config import DB_PATH, SESSION_TTL_SECONDS

def connect():
    con = sqlite3.connect(DB_PATH, timeout=15)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    return con

def init_db():
    with connect() as con:
        con.execute("CREATE TABLE IF NOT EXISTS documents (id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, pages INTEGER DEFAULT 0, chunks INTEGER DEFAULT 0, status TEXT NOT NULL, message TEXT DEFAULT '', allowed_roles TEXT NOT NULL DEFAULT '[]', document_id TEXT DEFAULT '', classification TEXT DEFAULT '', updated_at TEXT DEFAULT CURRENT_TIMESTAMP)")
        con.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, username TEXT NOT NULL, role TEXT NOT NULL, created_at REAL NOT NULL, last_seen REAL NOT NULL, turns INTEGER NOT NULL DEFAULT 0, active INTEGER NOT NULL DEFAULT 1)")
        con.execute("CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, created_at REAL NOT NULL, FOREIGN KEY(session_id) REFERENCES sessions(id))")
        con.execute("CREATE INDEX IF NOT EXISTS ix_sessions_user_active ON sessions(username,active,last_seen)")
        con.execute("CREATE INDEX IF NOT EXISTS ix_messages_session ON messages(session_id,id)")
        con.execute("CREATE TABLE IF NOT EXISTS request_windows (key TEXT PRIMARY KEY, window_start REAL NOT NULL, count INTEGER NOT NULL)")

def list_documents():
    with connect() as con:
        rows = [dict(r) for r in con.execute("SELECT name,pages,chunks,status,message,allowed_roles,document_id,classification,updated_at FROM documents ORDER BY name")]
    for row in rows:
        row['allowed_roles'] = json.loads(row['allowed_roles'])
    return rows

def upsert_document(name, pages, chunks, allowed_roles, document_id='', classification='', status='indexed', message=''):
    with connect() as con:
        old=con.execute("SELECT document_id FROM documents WHERE name=?",(name,)).fetchone()
        document_id=document_id or (old['document_id'] if old and old['document_id'] else str(uuid.uuid4()))
        con.execute("INSERT INTO documents(name,pages,chunks,status,message,allowed_roles,document_id,classification,updated_at) VALUES(?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP) ON CONFLICT(name) DO UPDATE SET pages=excluded.pages,chunks=excluded.chunks,status=excluded.status,message=excluded.message,allowed_roles=excluded.allowed_roles,document_id=excluded.document_id,classification=excluded.classification,updated_at=CURRENT_TIMESTAMP", (name,pages,chunks,status,message,json.dumps(allowed_roles),document_id,classification))
    return document_id

def get_document(name):
    with connect() as con:
        row=con.execute("SELECT * FROM documents WHERE name=?",(name,)).fetchone()
    if not row:return None
    result=dict(row);result['allowed_roles']=json.loads(result['allowed_roles'] or '[]')
    return result

def record_failure(name, message):
    with connect() as con:
        # A failed replacement must not erase the previously indexed document or widen/narrow its ACL.
        old=con.execute("SELECT status FROM documents WHERE name=?",(name,)).fetchone()
        if old and old['status']=='indexed':
            con.execute("UPDATE documents SET message=?,updated_at=CURRENT_TIMESTAMP WHERE name=?",(message[:500],name))
        else:
            con.execute("INSERT INTO documents(name,status,message,allowed_roles,document_id) VALUES(?,'failed',?,'[]',?) ON CONFLICT(name) DO UPDATE SET status='failed',message=excluded.message,updated_at=CURRENT_TIMESTAMP", (name,message[:500],str(uuid.uuid4())))


def create_session(session_id, username, role, now, max_sessions):
    with connect() as con:
        con.execute("UPDATE sessions SET active=0 WHERE username=? AND last_seen<?", (username,now-SESSION_TTL_SECONDS))
        active=con.execute("SELECT id FROM sessions WHERE username=? AND active=1 ORDER BY last_seen DESC",(username,)).fetchall()
        for old in active[max_sessions-1:]:
            con.execute("UPDATE sessions SET active=0 WHERE id=?",(old['id'],))
        con.execute("INSERT INTO sessions(id,username,role,created_at,last_seen,turns,active) VALUES(?,?,?,?,?,0,1)",(session_id,username,role,now,now))

def get_session(session_id):
    with connect() as con:
        row=con.execute("SELECT * FROM sessions WHERE id=?",(session_id,)).fetchone()
        return dict(row) if row else None

def touch_session(session_id, now):
    with connect() as con:
        con.execute("UPDATE sessions SET last_seen=?,turns=turns+1 WHERE id=?",(now,session_id))

def deactivate_session(session_id, username):
    with connect() as con:
        return con.execute("UPDATE sessions SET active=0 WHERE id=? AND username=?",(session_id,username)).rowcount > 0

def add_messages(session_id, user_text, assistant_text, now):
    with connect() as con:
        con.executemany("INSERT INTO messages(session_id,role,content,created_at) VALUES(?,?,?,?)",[(session_id,'user',user_text,now),(session_id,'assistant',assistant_text,now)])
        con.execute("DELETE FROM messages WHERE session_id=? AND id NOT IN (SELECT id FROM messages WHERE session_id=? ORDER BY id DESC LIMIT 12)",(session_id,session_id))

def get_history(session_id):
    with connect() as con:
        return [dict(r) for r in con.execute("SELECT role,content FROM messages WHERE session_id=? ORDER BY id",(session_id,))]

def allow_request(key, now, max_requests, window_seconds):
    with connect() as con:
        row=con.execute("SELECT window_start,count FROM request_windows WHERE key=?",(key,)).fetchone()
        if not row or now-row['window_start']>=window_seconds:
            con.execute("INSERT INTO request_windows(key,window_start,count) VALUES(?,?,1) ON CONFLICT(key) DO UPDATE SET window_start=excluded.window_start,count=1",(key,now)); return True,0
        if row['count']>=max_requests:
            return False,max(1,int(window_seconds-(now-row['window_start'])))
        con.execute("UPDATE request_windows SET count=count+1 WHERE key=?",(key,)); return True,0
