import base64, hashlib, hmac, json, secrets, time
from fastapi import HTTPException, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from app.config import DEMO_PASSWORD,TOKEN_SECRET

USERS = {'admin': 'Admin', 'hr': 'HR', 'employee': 'Employee'}
_secret = TOKEN_SECRET.encode()
_bearer = HTTPBearer(auto_error=False)

def _b64(data):
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode()

def _password_hash(password, salt):
    return hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 180_000)

def authenticate(username, password):
    role = USERS.get(username.lower())
    salt = username.lower().encode()
    valid = hmac.compare_digest(_password_hash(password, salt), _password_hash(DEMO_PASSWORD, salt))
    if not role or not valid:
        raise HTTPException(status_code=401, detail='Invalid username or password')
    body = _b64(json.dumps({'sub': username.lower(), 'role': role, 'exp': int(time.time()) + 8*3600}, separators=(',', ':')).encode())
    signature = _b64(hmac.new(_secret, body.encode(), hashlib.sha256).digest())
    return {'access_token': f'{body}.{signature}', 'token_type': 'bearer', 'role': role}

def current_user(credentials: HTTPAuthorizationCredentials = Depends(_bearer)):
    if not credentials:
        raise HTTPException(status_code=401, detail='Sign-in required')
    try:
        body, signature = credentials.credentials.split('.', 1)
        expected = _b64(hmac.new(_secret, body.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected):
            raise ValueError('bad signature')
        claims = json.loads(base64.urlsafe_b64decode(body + '=' * (-len(body) % 4)))
        if claims['exp'] < time.time() or claims['role'] not in {'Admin', 'HR', 'Employee'}:
            raise ValueError('expired/invalid claims')
        return claims
    except Exception:
        raise HTTPException(status_code=401, detail='Invalid or expired token')
