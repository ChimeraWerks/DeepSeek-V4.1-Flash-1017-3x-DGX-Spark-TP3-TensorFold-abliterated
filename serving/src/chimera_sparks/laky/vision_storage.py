# Copyright (c) 2026 LakyFx. MIT; see LICENSES/LakyFx-MIT.txt.
# Adapted for the recipe-owned TensorFold API; see serving/UPSTREAM.json.
"""Bound vision storage, with finite completed-image/result retention.

Archives are in the same SQLite transaction as replacing the hot body. Results,
identity, dispatch and cancellation receipts stay readable by the old gateway.
Image bytes expire only after terminal proof and the complete retry window.
Results survive longer for delivery/reconnect; unresolved attempts never expire.
"""
import hashlib
import json
import shutil
import zlib

HOT_LIMIT = 512 * 1024 * 1024
HOT_JOBS = 4096
RESULT_RESERVE = 8 * 1024 * 1024
REPAIR_RESERVE = 64 * 1024
DISK_RESERVE = 64 * 1024 * 1024
PAYLOAD_TTL = 3600
RESULT_TTL = 7 * 24 * 3600
MAINTENANCE_INTERVAL = 60


class VisionStorage:
    def __init__(self, ledger):
        self.ledger = ledger
        self.cached = None
        self.maintained_at = None
        with ledger.connect() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS vision_payload_archive (
                job_id TEXT PRIMARY KEY, codec TEXT NOT NULL, sha256 TEXT NOT NULL,
                original_bytes INTEGER NOT NULL, payload BLOB NOT NULL,
                archived REAL NOT NULL)''')

    def maintain(self, db, *, force=False):
        """Bounded transactional work on the ledger's existing owner thread.

        The expiry baseline is max(deadline, updated), never creation time.
        A successful HTTP write is not treated as client receipt confirmation.
        SQLite reuses freed pages; no VACUUM or live-database replacement here.
        """
        now = self.ledger.clock()
        if not force and self.maintained_at is not None and 0 <= now-self.maintained_at < MAINTENANCE_INTERVAL:
            return
        attempts = ("AND NOT EXISTS (SELECT 1 FROM vision_attempts v WHERE v.job_id=j.id "
                    "AND v.state IN ('running','unknown'))") if self.has_attempts(db) else ''
        safe = "j.state IN ('succeeded','failed') AND j.deadline<? AND j.updated<? " + attempts
        rows = db.execute('''SELECT j.id FROM jobs j LEFT JOIN vision_payload_archive a ON a.job_id=j.id
            WHERE ''' + safe + ''' AND (j.body<>'' OR (a.payload IS NOT NULL AND length(a.payload)>0))
            ORDER BY j.updated,j.created LIMIT 256''', (now-PAYLOAD_TTL, now-PAYLOAD_TTL)).fetchall()
        for row in rows:
            identity = row['id']
            # A tiny tombstone distinguishes expiry from missing/corrupt data.
            db.execute('''INSERT INTO vision_payload_archive VALUES (?, 'expired-v1', '', 0, X'', ?)
                ON CONFLICT(job_id) DO UPDATE SET codec='expired-v1',sha256='',original_bytes=0,
                payload=X'',archived=excluded.archived''', (identity, now))
            db.execute("UPDATE jobs SET body='' WHERE id=?", (identity,))
            if self.has_attempts(db):
                db.execute("UPDATE vision_attempts SET body='' WHERE job_id=?", (identity,))
        expired = db.execute('SELECT j.id FROM jobs j WHERE '+safe+' ORDER BY j.updated,j.created LIMIT 256',
                             (now-RESULT_TTL, now-RESULT_TTL)).fetchall()
        for row in expired:
            identity = row['id']
            if self.has_attempts(db):
                db.execute('DELETE FROM vision_attempts WHERE job_id=?', (identity,))
            db.execute('DELETE FROM vision_dispatch WHERE job_id=?', (identity,))
            db.execute('DELETE FROM vision_payload_archive WHERE job_id=?', (identity,))
            db.execute('DELETE FROM jobs WHERE id=?', (identity,))
        if rows or expired:
            self.cached = None
        # Set only after the transaction commits, in Ledger.connect().
        return now

    def maintenance(self):
        now = self.ledger.clock()
        if self.maintained_at is not None and 0 <= now-self.maintained_at < MAINTENANCE_INTERVAL:
            return
        with self.ledger.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            measured = self.maintain(db)
        if measured is not None:
            self.maintained_at = measured

    @staticmethod
    def has_attempts(db):
        return bool(db.execute("SELECT 1 FROM sqlite_master WHERE name='vision_attempts'").fetchone())

    def rows(self, db):
        # UTF-8 bytes throughout; TEXT length would undercount non-ASCII input.
        rows = {r['id']: dict(r) for r in db.execute('''SELECT j.id,j.state,
            length(cast(j.body as blob)) AS body_bytes,
            length(cast(j.body as blob))+coalesce(length(cast(j.result as blob)),0) AS used
            FROM jobs j LEFT JOIN vision_payload_archive a ON a.job_id=j.id WHERE a.job_id IS NULL''')}
        for r in db.execute('''SELECT d.job_id,length(cast(d.body as blob))+
                coalesce(length(cast(d.receipt as blob)),0) AS used FROM vision_dispatch d'''):
            if r['job_id'] in rows:
                rows[r['job_id']]['used'] += r['used']
        if self.has_attempts(db):
            for r in db.execute('''SELECT job_id,sum(length(cast(body as blob))+
                coalesce(length(cast(result as blob)),0)+coalesce(length(cast(dispatch as blob)),0)+
                coalesce(length(cast(receipt as blob)),0)) AS used FROM vision_attempts GROUP BY job_id'''):
                if r['job_id'] in rows:
                    rows[r['job_id']]['used'] += r['used']
        return rows

    def charge(self, body_bytes):
        return (1 + self.ledger.attempt_limit) * body_bytes + RESULT_RESERVE + REPAIR_RESERVE

    def usage(self, db):
        rows = self.rows(db)
        used = sum(r['used'] for r in rows.values())
        reserved = sum(max(0, self.charge(r['body_bytes']) - r['used']) for r in rows.values()
                       if r['state'] in {'queued', 'running', 'unknown'})
        return {'hot_jobs': len(rows), 'hot_bytes': used, 'reserved_bytes': reserved,
                'charged_bytes': used + reserved}

    def eligible(self, db):
        # Keep the entire dedup/repair interval. UNKNOWN and any unresolved
        # physical attempt are never archived, regardless of their age.
        attempts = ("AND NOT EXISTS (SELECT 1 FROM vision_attempts v WHERE v.job_id=j.id "
                    "AND v.state IN ('running','unknown'))") if self.has_attempts(db) else ''
        return db.execute('''SELECT j.id FROM jobs j LEFT JOIN vision_payload_archive a ON a.job_id=j.id
            WHERE a.job_id IS NULL AND j.state IN ('succeeded','failed') AND j.deadline<? AND j.updated<? '''
            + attempts + ' ORDER BY j.updated,j.created LIMIT 128',
            (self.ledger.clock()-900, self.ledger.clock()-900)).fetchall()

    @staticmethod
    def hydrate(db, identity):
        row = db.execute('SELECT * FROM vision_payload_archive WHERE job_id=?', (identity,)).fetchone()
        if row is None:
            raise ValueError('Missing retained vision payload archive')
        if row['codec'] == 'expired-v1':
            raise ValueError('Completed vision image retention expired; its result is still available by job ID')
        if row['codec'] != 'zlib-json-v1':
            raise ValueError('Unsupported vision payload archive codec')
        raw = zlib.decompress(row['payload'])
        if len(raw) != row['original_bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('Vision payload archive integrity check failed')
        return json.loads(raw)

    def archive(self, db, identity):
        job = db.execute('SELECT body FROM jobs WHERE id=?', (identity,)).fetchone()
        attempts = (db.execute('SELECT number,body FROM vision_attempts WHERE job_id=?', (identity,)).fetchall()
                    if self.has_attempts(db) else [])
        raw = json.dumps({'job': job['body'], 'attempts': {str(a['number']): a['body'] for a in attempts}},
                         ensure_ascii=False, separators=(',', ':')).encode()
        if shutil.disk_usage(self.ledger.path.parent).free < DISK_RESERVE + len(raw)*2:
            return False
        packed = zlib.compress(raw, level=1)
        db.execute('INSERT INTO vision_payload_archive VALUES (?,?,?,?,?,?)',
                   (identity, 'zlib-json-v1', hashlib.sha256(raw).hexdigest(), len(raw), packed, self.ledger.clock()))
        # Verify the persisted bytes before replacing the active copy. Failure
        # rolls back BOTH statements, including on power loss (SQLite/WAL).
        if self.hydrate(db, identity)['job'] != job['body']:
            raise ValueError('Vision payload archive verification failed')
        db.execute("UPDATE jobs SET body='' WHERE id=?", (identity,))
        if attempts:
            db.execute("UPDATE vision_attempts SET body='' WHERE job_id=?", (identity,))
        self.cached = None
        return True

    def make_room(self, db, incoming=0, new_job=False):
        self.maintain(db)
        current = self.usage(db)
        def fits():
            return (current['charged_bytes'] + incoming <= HOT_LIMIT
                    and current['hot_jobs'] + int(new_job) <= HOT_JOBS)
        if fits():
            return shutil.disk_usage(self.ledger.path.parent).free >= DISK_RESERVE
        rows = self.rows(db)
        for row in self.eligible(db):
            if not self.archive(db, row['id']):
                return False
            current['charged_bytes'] -= rows[row['id']]['used']
            current['hot_jobs'] -= 1
            if fits():
                return True
        return False

    def status(self, db):
        now = self.ledger.clock()
        if self.cached is not None and 0 <= now-self.cached['measured_at'] < 10:
            return self.cached
        usage = self.usage(db)
        candidates = {r['id'] for r in self.eligible(db)}
        reclaimable = sum(r['used'] for r in self.rows(db).values() if r['id'] in candidates)
        count, raw, packed = db.execute('''SELECT count(*),coalesce(sum(original_bytes),0),
            coalesce(sum(length(payload)),0) FROM vision_payload_archive WHERE codec<>'expired-v1' ''').fetchone()
        expired = db.execute("SELECT count(*) FROM vision_payload_archive WHERE codec='expired-v1'").fetchone()[0]
        free = shutil.disk_usage(self.ledger.path.parent).free
        minimum = self.charge(2)
        self.cached = {**usage, 'limit_bytes': HOT_LIMIT, 'limit_hot_jobs': HOT_JOBS,
                       'reclaimable_bytes': reclaimable, 'archived_jobs': count,
                       'archived_original_bytes': raw, 'archived_compressed_bytes': packed,
                       'expired_payload_jobs': expired,
                       'disk_free_bytes': free, 'disk_reserve_bytes': DISK_RESERVE,
                       'measured_at': now,
                       'payload_retention_seconds': PAYLOAD_TTL, 'result_retention_seconds': RESULT_TTL,
                       'available': free >= DISK_RESERVE and usage['charged_bytes']-reclaimable+minimum <= HOT_LIMIT
                           and usage['hot_jobs']-len(candidates) < HOT_JOBS}
        return self.cached
