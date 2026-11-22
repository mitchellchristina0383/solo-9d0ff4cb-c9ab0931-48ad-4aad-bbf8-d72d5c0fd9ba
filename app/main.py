"""混合精度训练 · 单作业协调 API.

协调 N 个训练进程的梯度/溢出上报:
- 作业登记: 进程数、初始权重向量、学习率; 缩放倍率从 16 起步, 步号从 0 起步
- 每进程每步只采纳首报 (reports 表 PRIMARY KEY(step, rank) + INSERT OR IGNORE)
- 梯度齐全 -> 取平均、除以本步倍率、按学习率做一次 SGD 更新
- 任一进程报溢出 -> 立即跳过更新, 倍率减半 (最低 1), 不等其余进程
- 两种结果都进入下一步; 旧步迟报/重试只返回该步已记录的决议
- 全部状态存 SQLite, 服务重启后保留已受理上报、权重、倍率和步决议
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import asynccontextmanager, contextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

DB_PATH = os.environ.get("DB_PATH", "/data/coordinator.db")
INITIAL_SCALE = 16.0
MIN_SCALE = 1.0

# 进程内锁: 串行化写路径。跨进程的正确性由 SQLite 的
# BEGIN IMMEDIATE 事务 + 唯一约束兜底, 因此单 worker 即可安全运行。
_lock = threading.RLock()


# ---------------------------------------------------------------------------
# 存储层
# ---------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@contextmanager
def write_tx():
    """BEGIN IMMEDIATE 写事务: 读-改-写整体串行化, 异常即回滚."""
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


@contextmanager
def read_conn():
    conn = _connect()
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    directory = os.path.dirname(DB_PATH)
    if directory:
        os.makedirs(directory, exist_ok=True)
    # executescript 自行管理事务, 不走 write_tx
    conn = _connect()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS job (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                num_processes INTEGER NOT NULL,
                lr REAL NOT NULL,
                weights TEXT NOT NULL,        -- JSON 数组, 当前权重
                scale REAL NOT NULL,          -- 当前缩放倍率
                step INTEGER NOT NULL         -- 当前步号
            );
            CREATE TABLE IF NOT EXISTS reports (
                step INTEGER NOT NULL,
                rank INTEGER NOT NULL,
                kind TEXT NOT NULL CHECK (kind IN ('grad', 'overflow')),
                grad TEXT,                    -- JSON 数组; overflow 时为 NULL
                received_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (step, rank)      -- 每进程每步只采纳首报
            );
            CREATE TABLE IF NOT EXISTS decisions (
                step INTEGER PRIMARY KEY,     -- 每步只有一份决议
                outcome TEXT NOT NULL CHECK (outcome IN ('updated', 'skipped')),
                scale REAL NOT NULL,          -- 本步使用的倍率
                new_scale REAL NOT NULL,      -- 步结束后的倍率
                weights TEXT NOT NULL,        -- 步结束后的权重 (JSON)
                avg_grad TEXT,                -- 平均梯度 (JSON); 溢出跳过时为 NULL
                decided_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            """
        )
    finally:
        conn.close()


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Mixed-Precision Training Coordinator",
    version="1.0.0",
    description="单作业混合精度训练协调: 梯度汇总 / 溢出跳步 / 动态损失缩放",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------

class JobConfig(BaseModel):
    num_processes: int = Field(gt=0, description="训练进程数")
    weights: list[float] = Field(min_length=1, description="初始权重向量")
    lr: float = Field(gt=0, description="学习率")


class Report(BaseModel):
    step: int = Field(ge=0, description="上报所属步号")
    rank: int = Field(ge=0, description="进程编号, [0, num_processes)")
    kind: Literal["grad", "overflow"]
    grad: Optional[list[float]] = Field(default=None, description="已缩放梯度, kind=grad 时必填")


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def _get_job(conn: sqlite3.Connection) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM job WHERE id = 1").fetchone()
    if row is None:
        raise HTTPException(409, "no job registered; POST /job first")
    return row


def _decision_dict(row: sqlite3.Row) -> dict:
    return {
        "step": row["step"],
        "outcome": row["outcome"],
        "scale": row["scale"],
        "new_scale": row["new_scale"],
        "weights": json.loads(row["weights"]),
        "avg_grad": json.loads(row["avg_grad"]) if row["avg_grad"] else None,
        "decided_at": row["decided_at"],
    }


def _decide_skip(conn: sqlite3.Connection, job: sqlite3.Row) -> dict:
    """溢出: 跳过更新, 倍率减半 (最低 1), 权重不变, 进入下一步."""
    step, scale = job["step"], job["scale"]
    new_scale = max(MIN_SCALE, scale / 2)
    conn.execute(
        "INSERT INTO decisions (step, outcome, scale, new_scale, weights, avg_grad)"
        " VALUES (?, 'skipped', ?, ?, ?, NULL)",
        (step, scale, new_scale, job["weights"]),
    )
    conn.execute("UPDATE job SET step = ?, scale = ? WHERE id = 1", (step + 1, new_scale))
    return {
        "step": step,
        "outcome": "skipped",
        "scale": scale,
        "new_scale": new_scale,
        "weights": json.loads(job["weights"]),
        "avg_grad": None,
    }


def _decide_update(conn: sqlite3.Connection, job: sqlite3.Row) -> dict:
    """梯度齐全: 平均 -> 除以本步倍率 -> 按学习率更新一次权重, 倍率不变."""
    step, scale, lr = job["step"], job["scale"], job["lr"]
    weights = json.loads(job["weights"])
    rows = conn.execute(
        "SELECT grad FROM reports WHERE step = ? ORDER BY rank", (step,)
    ).fetchall()
    grads = [json.loads(r["grad"]) for r in rows]
    n, dim = len(grads), len(weights)
    avg = [sum(g[i] for g in grads) / n for i in range(dim)]
    new_weights = [w - lr * (a / scale) for w, a in zip(weights, avg)]
    conn.execute(
        "INSERT INTO decisions (step, outcome, scale, new_scale, weights, avg_grad)"
        " VALUES (?, 'updated', ?, ?, ?, ?)",
        (step, scale, scale, json.dumps(new_weights), json.dumps(avg)),
    )
    conn.execute(
        "UPDATE job SET step = ?, weights = ? WHERE id = 1",
        (step + 1, json.dumps(new_weights)),
    )
    return {
        "step": step,
        "outcome": "updated",
        "scale": scale,
        "new_scale": scale,
        "weights": new_weights,
        "avg_grad": avg,
    }


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "mixed-precision-training-coordinator",
        "docs": "/docs",
        "endpoints": ["POST /job", "GET /state", "POST /report",
                      "GET /steps/{step}", "DELETE /job", "GET /healthz"],
    }


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/job", status_code=201)
def create_job(cfg: JobConfig):
    """登记作业: 进程数、初始权重、学习率。倍率从 16 起步, 步号从 0 起步。"""
    with _lock, write_tx() as conn:
        if conn.execute("SELECT id FROM job WHERE id = 1").fetchone():
            raise HTTPException(409, "job already registered; DELETE /job to reset")
        conn.execute(
            "INSERT INTO job (id, num_processes, lr, weights, scale, step)"
            " VALUES (1, ?, ?, ?, ?, 0)",
            (cfg.num_processes, cfg.lr, json.dumps(cfg.weights), INITIAL_SCALE),
        )
    return {
        "ok": True,
        "num_processes": cfg.num_processes,
        "weights": cfg.weights,
        "lr": cfg.lr,
        "scale": INITIAL_SCALE,
        "step": 0,
    }


@app.get("/state")
def get_state():
    """进程读取: 当前步号、权重、倍率, 以及本步已受理的上报。"""
    with read_conn() as conn:
        job = _get_job(conn)
        reports = conn.execute(
            "SELECT rank, kind, received_at FROM reports WHERE step = ? ORDER BY rank",
            (job["step"],),
        ).fetchall()
    return {
        "step": job["step"],
        "scale": job["scale"],
        "weights": json.loads(job["weights"]),
        "lr": job["lr"],
        "num_processes": job["num_processes"],
        "reports_received": [dict(r) for r in reports],
    }


@app.post("/report")
def post_report(rep: Report):
    """按步号和进程编号上报已缩放梯度或溢出。

    - 旧步 (step < 当前步): 迟报/重试, 只返回该步决议, 不改状态
    - 未来步 (step > 当前步): 409 拒绝
    - 当前步: 首报入库; 有溢出立即跳步减半倍率; 梯度齐全则平均后更新权重
    """
    with _lock, write_tx() as conn:
        job = _get_job(conn)
        n_proc = job["num_processes"]
        weights = json.loads(job["weights"])
        if rep.rank >= n_proc:
            raise HTTPException(422, f"rank {rep.rank} out of range [0, {n_proc})")
        if rep.kind == "grad":
            if rep.grad is None:
                raise HTTPException(422, "grad is required when kind='grad'")
            if len(rep.grad) != len(weights):
                raise HTTPException(
                    422, f"grad length {len(rep.grad)} != weights length {len(weights)}"
                )
        cur = job["step"]

        # 旧步: 迟报或重试, 只返回该步决议
        if rep.step < cur:
            row = conn.execute(
                "SELECT * FROM decisions WHERE step = ?", (rep.step,)
            ).fetchone()
            return {
                "accepted": False,
                "late": True,
                "duplicate": False,
                "step": rep.step,
                "current_step": cur,
                "scale": job["scale"],
                "decision": _decision_dict(row) if row else None,
            }
        if rep.step > cur:
            raise HTTPException(
                409, f"report for future step {rep.step}; current step is {cur}"
            )

        # 当前步: 每进程每步只采纳首报
        cursor = conn.execute(
            "INSERT OR IGNORE INTO reports (step, rank, kind, grad) VALUES (?, ?, ?, ?)",
            (rep.step, rep.rank, rep.kind,
             json.dumps(rep.grad) if rep.kind == "grad" else None),
        )
        accepted = cursor.rowcount == 1

        decision = None
        overflow = conn.execute(
            "SELECT 1 FROM reports WHERE step = ? AND kind = 'overflow' LIMIT 1",
            (cur,),
        ).fetchone()
        if overflow:
            # 任一进程报溢出: 立即跳过更新, 倍率减半
            decision = _decide_skip(conn, job)
        else:
            cnt = conn.execute(
                "SELECT COUNT(*) AS c FROM reports WHERE step = ?", (cur,)
            ).fetchone()["c"]
            if cnt == n_proc:
                # 梯度齐全: 平均、除倍率、按学习率更新一次
                decision = _decide_update(conn, job)

        new_job = conn.execute("SELECT * FROM job WHERE id = 1").fetchone()
        return {
            "accepted": accepted,
            "late": False,
            "duplicate": not accepted,
            "step": rep.step,
            "current_step": new_job["step"],
            "scale": new_job["scale"],
            "decision": decision,
        }


@app.get("/steps/{step}")
def get_step(step: int):
    """查询某步的决议与已受理上报。"""
    with read_conn() as conn:
        _get_job(conn)
        dec = conn.execute(
            "SELECT * FROM decisions WHERE step = ?", (step,)
        ).fetchone()
        reports = conn.execute(
            "SELECT rank, kind, grad, received_at FROM reports"
            " WHERE step = ? ORDER BY rank",
            (step,),
        ).fetchall()
    return {
        "step": step,
        "decision": _decision_dict(dec) if dec else None,
        "reports": [
            {
                "rank": r["rank"],
                "kind": r["kind"],
                "grad": json.loads(r["grad"]) if r["grad"] else None,
                "received_at": r["received_at"],
            }
            for r in reports
        ],
    }


@app.delete("/job")
def reset_job():
    """清空作业与全部历史 (仅用于演示/测试重置)。"""
    with _lock, write_tx() as conn:
        conn.execute("DELETE FROM reports")
        conn.execute("DELETE FROM decisions")
        conn.execute("DELETE FROM job")
    return {"ok": True}
