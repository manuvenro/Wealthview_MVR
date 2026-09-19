import sqlite3
import pandas as pd
import bcrypt
import os
from modules.logger import get_logger

_log = get_logger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'wealthview.db')


def get_conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS portfolios
                 (username TEXT, ticker TEXT, shares REAL,
                  asset_type TEXT DEFAULT 'equity',
                  face_value REAL DEFAULT NULL,
                  coupon_rate REAL DEFAULT NULL,
                  maturity_date TEXT DEFAULT NULL,
                  asset_name TEXT DEFAULT NULL,
                  strike REAL DEFAULT NULL,
                  multiplier REAL DEFAULT NULL,
                  option_type TEXT DEFAULT NULL,
                  premium REAL DEFAULT NULL,
                  FOREIGN KEY(username) REFERENCES users(username))''')
    for col, definition in [
        ('asset_type',    "TEXT DEFAULT 'equity'"),
        ('face_value',    'REAL DEFAULT NULL'),
        ('coupon_rate',   'REAL DEFAULT NULL'),
        ('maturity_date', 'TEXT DEFAULT NULL'),
        ('asset_name',    'TEXT DEFAULT NULL'),
        ('strike',        'REAL DEFAULT NULL'),
        ('multiplier',    'REAL DEFAULT NULL'),
        ('option_type',   'TEXT DEFAULT NULL'),
        ('premium',       'REAL DEFAULT NULL'),
        ('avg_cost',      'REAL DEFAULT NULL'),
        ('currency',      "TEXT DEFAULT 'USD'"),
    ]:
        try:
            c.execute(f'ALTER TABLE portfolios ADD COLUMN {col} {definition}')
        except Exception:
            pass
    c.execute('''CREATE TABLE IF NOT EXISTS watchlist
                 (username TEXT, ticker TEXT, asset_name TEXT DEFAULT NULL,
                  PRIMARY KEY(username, ticker),
                  FOREIGN KEY(username) REFERENCES users(username))''')
    try:
        c.execute('ALTER TABLE watchlist ADD COLUMN asset_name TEXT DEFAULT NULL')
    except Exception:
        pass
    c.execute('''CREATE TABLE IF NOT EXISTS alerts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT, ticker TEXT,
                  direction TEXT, threshold REAL, active INTEGER DEFAULT 1,
                  FOREIGN KEY(username) REFERENCES users(username))''')
    c.execute('''CREATE TABLE IF NOT EXISTS transactions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT NOT NULL,
                  ticker TEXT NOT NULL,
                  date TEXT NOT NULL,
                  type TEXT NOT NULL,
                  quantity REAL NOT NULL,
                  price REAL NOT NULL,
                  commission REAL DEFAULT 0.0,
                  currency TEXT DEFAULT 'USD',
                  notes TEXT DEFAULT '',
                  created_at TEXT DEFAULT (datetime('now')),
                  FOREIGN KEY(username) REFERENCES users(username))''')
    # ── Security tables ────────────────────────────────────────────────────────
    c.execute('''CREATE TABLE IF NOT EXISTS login_attempts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT NOT NULL,
                  success INTEGER NOT NULL,
                  ip TEXT DEFAULT NULL,
                  attempted_at TEXT DEFAULT (datetime('now')))''')
    c.execute('''CREATE TABLE IF NOT EXISTS user_lockouts
                 (username TEXT PRIMARY KEY,
                  locked_until TEXT NOT NULL,
                  failed_count INTEGER DEFAULT 0)''')
    # ── Key-value store genérico por usuario (tesis, notas, etc.) ─────────────
    c.execute('''CREATE TABLE IF NOT EXISTS user_kv
                 (username TEXT NOT NULL,
                  key TEXT NOT NULL,
                  value TEXT,
                  updated_at TEXT DEFAULT (datetime('now')),
                  PRIMARY KEY(username, key),
                  FOREIGN KEY(username) REFERENCES users(username))''')
    # Add last_login column to users if missing
    for col, definition in [
        ('last_login', "TEXT DEFAULT NULL"),
        ('created_at', "TEXT DEFAULT (datetime('now'))"),
    ]:
        try:
            c.execute(f'ALTER TABLE users ADD COLUMN {col} {definition}')
        except Exception:
            pass
    conn.commit()
    conn.close()


def save_watchlist(username, tickers, names=None):
    conn = get_conn()
    c = conn.cursor()
    c.execute('DELETE FROM watchlist WHERE username=?', (username,))
    for ticker in tickers:
        name = (names or {}).get(ticker, None)
        c.execute('INSERT OR IGNORE INTO watchlist (username, ticker, asset_name) VALUES (?, ?, ?)',
                  (username, ticker.upper(), name))
    conn.commit()
    conn.close()


def load_watchlist(username):
    conn = get_conn()
    c = conn.cursor()
    c.execute('SELECT ticker FROM watchlist WHERE username=?', (username,))
    rows = c.fetchall()
    conn.close()
    return [r[0] for r in rows]


def load_watchlist_with_names(username):
    conn = get_conn()
    c = conn.cursor()
    c.execute('SELECT ticker, asset_name FROM watchlist WHERE username=?', (username,))
    rows = c.fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}


def save_alert(username, ticker, direction, threshold):
    conn = get_conn()
    c = conn.cursor()
    c.execute('INSERT INTO alerts (username, ticker, direction, threshold) VALUES (?, ?, ?, ?)',
              (username, ticker.upper(), direction, threshold))
    conn.commit()
    conn.close()


def load_alerts(username):
    conn = get_conn()
    df = pd.read_sql_query(
        'SELECT id, ticker, direction, threshold, active FROM alerts WHERE username=? ORDER BY id DESC',
        conn, params=(username,))
    conn.close()
    return df


def delete_alert(alert_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute('DELETE FROM alerts WHERE id=?', (int(alert_id),))
    conn.commit()
    conn.close()


# ── Configuración de seguridad ─────────────────────────────────────────────────
_MAX_FAILED_ATTEMPTS  = 5      # intentos antes de lockout
_LOCKOUT_MINUTES      = 15     # minutos de bloqueo
_SESSION_TIMEOUT_HOURS = 8     # horas antes de expirar sesión


def hash_password(password: str) -> str:
    """Genera hash bcrypt de la contraseña."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def verify_password(password: str, hashed: str) -> bool:
    """Verifica contraseña contra hash bcrypt. También acepta SHA-256 legacy."""
    # Detectar hash legacy SHA-256 (64 hex chars, no empieza por $2)
    if len(hashed) == 64 and not hashed.startswith('$2'):
        import hashlib
        return hashlib.sha256(password.encode()).hexdigest() == hashed
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    except Exception:
        return False


# ── Rate limiting ──────────────────────────────────────────────────────────────

def _record_attempt(username: str, success: bool) -> None:
    """Registra un intento de login en login_attempts."""
    conn = get_conn()
    c = conn.cursor()
    c.execute('INSERT INTO login_attempts (username, success) VALUES (?, ?)',
              (username, 1 if success else 0))
    conn.commit()
    conn.close()


def _is_locked(username: str) -> tuple[bool, str]:
    """
    Devuelve (is_locked, unlock_time_str).
    Si está bloqueado, unlock_time_str es la hora de desbloqueo en UTC.
    """
    from datetime import datetime, timedelta
    conn = get_conn()
    c = conn.cursor()
    c.execute('SELECT locked_until, failed_count FROM user_lockouts WHERE username=?', (username,))
    row = c.fetchone()
    conn.close()
    if row is None:
        return False, ""
    locked_until_str, failed_count = row
    if failed_count < _MAX_FAILED_ATTEMPTS:
        return False, ""
    try:
        locked_until = datetime.fromisoformat(locked_until_str)
        if datetime.utcnow() < locked_until:
            return True, locked_until.strftime('%H:%M:%S UTC')
        # Lockout expirado — limpiar
        _clear_lockout(username)
        return False, ""
    except Exception:
        return False, ""


def _increment_lockout(username: str) -> int:
    """
    Incrementa el contador de fallos. Si supera el límite, activa lockout.
    Devuelve el número actual de intentos fallidos.
    """
    from datetime import datetime, timedelta
    conn = get_conn()
    c = conn.cursor()
    c.execute('SELECT failed_count FROM user_lockouts WHERE username=?', (username,))
    row = c.fetchone()
    new_count = (row[0] + 1) if row else 1

    if new_count >= _MAX_FAILED_ATTEMPTS:
        locked_until = (datetime.utcnow() + timedelta(minutes=_LOCKOUT_MINUTES)).isoformat()
        c.execute('''INSERT INTO user_lockouts (username, locked_until, failed_count)
                     VALUES (?, ?, ?)
                     ON CONFLICT(username) DO UPDATE SET
                       locked_until=excluded.locked_until,
                       failed_count=excluded.failed_count''',
                  (username, locked_until, new_count))
        _log.warning("Usuario '%s' bloqueado por %d min (intentos: %d)",
                     username, _LOCKOUT_MINUTES, new_count)
    else:
        c.execute('''INSERT INTO user_lockouts (username, locked_until, failed_count)
                     VALUES (?, datetime('now', '+1 hour'), ?)
                     ON CONFLICT(username) DO UPDATE SET failed_count=excluded.failed_count''',
                  (username, new_count))
    conn.commit()
    conn.close()
    return new_count


def _clear_lockout(username: str) -> None:
    """Elimina el registro de lockout tras login exitoso."""
    conn = get_conn()
    c = conn.cursor()
    c.execute('DELETE FROM user_lockouts WHERE username=?', (username,))
    conn.commit()
    conn.close()


def get_failed_attempts(username: str) -> int:
    """Devuelve el número de intentos fallidos recientes del usuario."""
    conn = get_conn()
    c = conn.cursor()
    c.execute('SELECT failed_count FROM user_lockouts WHERE username=?', (username,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else 0


def unlock_user(username: str) -> None:
    """Desbloquea manualmente a un usuario (acción de administrador)."""
    _clear_lockout(username)
    _log.info("Usuario '%s' desbloqueado manualmente", username)


def get_login_history(username: str, limit: int = 10) -> list[dict]:
    """Devuelve los últimos N intentos de login del usuario."""
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        'SELECT success, attempted_at FROM login_attempts '
        'WHERE username=? ORDER BY id DESC LIMIT ?',
        (username, limit)
    )
    rows = c.fetchall()
    conn.close()
    return [{"success": bool(r[0]), "at": r[1]} for r in rows]


# ── Login / Create / Change password ──────────────────────────────────────────

def login_user(username: str, password: str) -> tuple[bool, str]:
    """
    Verifica credenciales con rate limiting.
    Devuelve (success, error_message).
    error_message está vacío en caso de éxito.
    """
    # 1. Comprobar lockout
    locked, until_str = _is_locked(username)
    if locked:
        msg = f"Cuenta bloqueada hasta las {until_str}. Demasiados intentos fallidos."
        _log.warning("Login bloqueado para '%s' hasta %s", username, until_str)
        return False, msg

    # 2. Buscar usuario
    conn = get_conn()
    c = conn.cursor()
    c.execute('SELECT password FROM users WHERE username=?', (username,))
    row = c.fetchone()

    if row is None:
        conn.close()
        _record_attempt(username, False)
        _log.warning("Login fallido: usuario '%s' no existe", username)
        return False, ""

    stored_hash = row[0]
    ok = verify_password(password, stored_hash)

    if ok:
        # Migrar hash SHA-256 legacy a bcrypt en el primer login exitoso
        if len(stored_hash) == 64 and not stored_hash.startswith('$2'):
            new_hash = hash_password(password)
            c.execute('UPDATE users SET password=?, last_login=datetime("now") WHERE username=?',
                      (new_hash, username))
            _log.info("Hash legacy migrado a bcrypt para '%s'", username)
        else:
            c.execute('UPDATE users SET last_login=datetime("now") WHERE username=?', (username,))
        conn.commit()
        conn.close()
        _clear_lockout(username)
        _record_attempt(username, True)
        _log.info("Login exitoso: %s", username)
        return True, ""
    else:
        conn.close()
        count = _increment_lockout(username)
        _record_attempt(username, False)
        remaining = _MAX_FAILED_ATTEMPTS - count
        _log.warning("Login fallido para '%s' (intento %d/%d)", username, count, _MAX_FAILED_ATTEMPTS)
        if remaining > 0:
            return False, f"Contraseña incorrecta. {remaining} intento(s) restantes antes del bloqueo."
        return False, f"Cuenta bloqueada durante {_LOCKOUT_MINUTES} minutos."


def create_user(username: str, password: str) -> bool:
    try:
        conn = get_conn()
        c = conn.cursor()
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)',
                  (username, hash_password(password)))
        conn.commit()
        conn.close()
        _log.info("Usuario creado: %s", username)
        return True
    except sqlite3.IntegrityError:

        _log.warning("Intento de crear usuario duplicado: %s", username)
        return False


def change_password(username: str, old_password: str, new_password: str) -> tuple[bool, str]:
    """
    Cambia la contrasena verificando la actual.
    Devuelve (success, error_message).
    """
    conn = get_conn()
    c = conn.cursor()
    c.execute('SELECT password FROM users WHERE username=?', (username,))
    row = c.fetchone()
    if row is None:
        conn.close()
        return False, "Usuario no encontrado."
    if not verify_password(old_password, row[0]):
        conn.close()
        _log.warning("Cambio de contrasena fallido para '%s'", username)
        return False, "Contrasena actual incorrecta."
    c.execute('UPDATE users SET password=? WHERE username=?',
              (hash_password(new_password), username))
    conn.commit()
    conn.close()
    _log.info("Contrasena cambiada para '%s'", username)
    return True, ""


def get_last_login(username: str) -> str | None:
    """Devuelve la fecha/hora del ultimo login exitoso o None."""
    conn = get_conn()
    c = conn.cursor()
    c.execute('SELECT last_login FROM users WHERE username=?', (username,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


def save_portfolio(username, portfolio_df):
    backup_path = backup_portfolio(username, portfolio_df)
    if backup_path:
        _log.info("Backup creado antes de guardar: %s", os.path.basename(backup_path))

    conn = get_conn()
    c = conn.cursor()
    c.execute('DELETE FROM portfolios WHERE username=?', (username,))
    for _, row in portfolio_df.iterrows():
        if pd.notna(row.get('Ticker')) and str(row.get('Ticker', '')).strip():
            vals = (
                username,
                str(row.get('Ticker', '')).strip(),
                float(row.get('Shares', 0) or 0),
                str(row.get('Asset Type', 'equity')),
                float(row['Face Value']) if pd.notna(row.get('Face Value')) else None,
                float(row['Coupon Rate']) if pd.notna(row.get('Coupon Rate')) else None,
                str(row['Maturity']) if pd.notna(row.get('Maturity')) else None,
                str(row['Name']) if pd.notna(row.get('Name')) else None,
                float(row['Strike']) if pd.notna(row.get('Strike')) else None,
                float(row['Multiplier']) if pd.notna(row.get('Multiplier')) else None,
                str(row['Option Type']) if pd.notna(row.get('Option Type')) else None,
                float(row['Premium']) if pd.notna(row.get('Premium')) else None,
                float(row['Avg Cost']) if pd.notna(row.get('Avg Cost')) else None,
                str(row.get('Currency', 'USD')),
            )
            c.execute(
                'INSERT INTO portfolios'
                ' (username, ticker, shares, asset_type,'
                '  face_value, coupon_rate, maturity_date, asset_name,'
                '  strike, multiplier, option_type, premium, avg_cost, currency)'
                ' VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                vals
            )
    conn.commit()
    conn.close()
    _log.info("Portfolio guardado para '%s': %d posiciones", username, len(portfolio_df))


def load_portfolio(username):
    conn = get_conn()
    df = pd.read_sql_query(
        'SELECT * FROM portfolios WHERE username=?', conn, params=(username,)
    )
    conn.close()
    if df.empty:
        return df
    col_map = {
        'ticker': 'Ticker', 'shares': 'Shares', 'asset_type': 'Asset Type',
        'face_value': 'Face Value', 'coupon_rate': 'Coupon Rate',
        'maturity_date': 'Maturity', 'asset_name': 'Name',
        'strike': 'Strike', 'multiplier': 'Multiplier',
        'option_type': 'Option Type', 'premium': 'Premium',
        'avg_cost': 'Avg Cost', 'currency': 'Currency',
    }
    df = df.rename(columns=col_map)
    df = df.drop(columns=['username'], errors='ignore')
    return df


# ── Backups ────────────────────────────────────────────────────────────────────

_BACKUP_DIR_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backups')
_MAX_BACKUPS = 10


def backup_portfolio(username: str, portfolio_df, max_backups: int = _MAX_BACKUPS) -> str | None:
    """Guarda una copia del portfolio en JSON. Devuelve la ruta del backup o None."""
    if portfolio_df is None or (hasattr(portfolio_df, 'empty') and portfolio_df.empty):
        return None
    os.makedirs(_BACKUP_DIR_BASE, exist_ok=True)
    from datetime import datetime as _dt2
    ts = _dt2.utcnow().strftime('%Y%m%dT%H%M%S')
    fname = f"{username}_{ts}.json"
    path = os.path.join(_BACKUP_DIR_BASE, fname)
    portfolio_df.to_json(path, orient='records', date_format='iso')
    existing = sorted(
        [f for f in os.listdir(_BACKUP_DIR_BASE) if f.startswith(username + '_')],
        reverse=True
    )
    for old in existing[max_backups:]:
        try:
            os.remove(os.path.join(_BACKUP_DIR_BASE, old))
        except Exception:
            pass
    return path


def list_backups(username: str) -> list:
    """Devuelve lista de dicts con info de cada backup del usuario."""
    if not os.path.exists(_BACKUP_DIR_BASE):
        return []
    files = sorted(
        [f for f in os.listdir(_BACKUP_DIR_BASE) if f.startswith(username + '_')],
        reverse=True
    )
    result = []
    for f in files:
        full = os.path.join(_BACKUP_DIR_BASE, f)
        try:
            size_kb = round(os.path.getsize(full) / 1024, 1)
            ts_str = f.replace(username + '_', '').replace('.json', '')
            # Count positions from backup
            try:
                import json as _json
                with open(full) as _bf:
                    records = _json.load(_bf)
                n_pos = len(records)
                tickers = [r.get('Ticker','') for r in records if r.get('Ticker')]
            except Exception:
                n_pos = 0
                tickers = []
            result.append({
                'filename': f, 'path': full,
                'size_kb': size_kb, 'ts': ts_str, 'timestamp_str': ts_str,
                'n_positions': n_pos, 'tickers': tickers,
            })
        except Exception:
            pass
    return result


def restore_backup(username: str, backup_path: str):
    """Restaura el portfolio desde un fichero de backup (nombre o ruta completa)."""
    # Accept filename or full path
    if not os.path.isabs(backup_path):
        backup_path = os.path.join(_BACKUP_DIR_BASE, backup_path)
    if not os.path.exists(backup_path):
        return None
    try:
        df = pd.read_json(backup_path, orient='records')
        _log.info("Backup restaurado para '%s' desde %s", username, os.path.basename(backup_path))
        return df
    except Exception as e:
        _log.error("Error al restaurar backup para '%s': %s", username, e)
        return None


# ── Key-Value store genérico por usuario ──────────────────────────────────────

def save_user_kv(username: str, key: str, value: str) -> None:
    """Guarda un valor arbitrario (string) asociado al usuario y a una clave."""
    try:
        conn = get_conn()
        c = conn.cursor()
        c.execute(
            '''INSERT INTO user_kv (username, key, value, updated_at)
               VALUES (?, ?, ?, datetime('now'))
               ON CONFLICT(username, key) DO UPDATE SET value=excluded.value,
               updated_at=excluded.updated_at''',
            (username, key, value)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        _log.error("save_user_kv error para '%s' key='%s': %s", username, key, e)


def load_user_kv(username: str, key: str) -> str | None:
    """Carga un valor del key-value store del usuario. Devuelve None si no existe."""
    try:
        conn = get_conn()
        c = conn.cursor()
        c.execute('SELECT value FROM user_kv WHERE username=? AND key=?', (username, key))
        row = c.fetchone()
        conn.close()
        return row[0] if row else None
    except Exception as e:
        _log.error("load_user_kv error para '%s' key='%s': %s", username, key, e)
        return None
