#!/usr/bin/env python3
"""协调 API 验证脚本 (仅依赖标准库)。

用法:
    python3 scripts/verify.py phase1 [BASE_URL]   # 主流程: 更新/溢出/迟报/重试/倍率下限
    # ... 重启服务 (同一数据库) ...
    python3 scripts/verify.py phase2 [BASE_URL]   # 重启后: 状态保留 + 重报旧步/当前步

phase1 结束时的预期状态: step=7, scale=1, weights=[0.75, -1.05],
且 rank0 已对 step7 上报梯度 [1,1] (未齐, 无决议)。
"""
import json
import sys
import urllib.error
import urllib.request

PHASE = sys.argv[1] if len(sys.argv) > 1 else "phase1"
BASE = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000"
TOL = 1e-9
failures = []


def req(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(r) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" | {detail}" if detail else ""))
    if not cond:
        failures.append(name)


def vec_close(a, b):
    return a is not None and len(a) == len(b) and all(abs(x - y) < TOL for x, y in zip(a, b))


def phase1():
    # 重置并登记作业: 2 进程, 权重 [1.0, -1.0], lr=0.1
    req("DELETE", "/job")
    st, j = req("POST", "/job", {"num_processes": 2, "weights": [1.0, -1.0], "lr": 0.1})
    check("登记作业 201", st == 201 and j["scale"] == 16.0 and j["step"] == 0, f"status={st}")
    st, _ = req("POST", "/job", {"num_processes": 2, "weights": [1.0], "lr": 0.1})
    check("重复登记被拒 409", st == 409, f"status={st}")

    st, s = req("GET", "/state")
    check("初始状态 step0/scale16", s["step"] == 0 and s["scale"] == 16.0
          and vec_close(s["weights"], [1.0, -1.0]), json.dumps(s))

    # 参数校验
    st, _ = req("POST", "/report", {"step": 0, "rank": 5, "kind": "overflow"})
    check("rank 越界 422", st == 422, f"status={st}")
    st, _ = req("POST", "/report", {"step": 0, "rank": 0, "kind": "grad", "grad": [1.0]})
    check("梯度维度不符 422", st == 422, f"status={st}")
    st, _ = req("POST", "/report", {"step": 9, "rank": 0, "kind": "overflow"})
    check("未来步被拒 409", st == 409, f"status={st}")

    # --- step 0: 正常更新 ---
    st, r = req("POST", "/report", {"step": 0, "rank": 0, "kind": "grad", "grad": [16.0, 32.0]})
    check("s0 rank0 首报受理", st == 200 and r["accepted"] and r["decision"] is None)
    st, r = req("POST", "/report", {"step": 0, "rank": 0, "kind": "grad", "grad": [999.0, 999.0]})
    check("s0 rank0 重报被拒(首报唯一)", r["accepted"] is False and r["duplicate"] is True)
    st, r = req("POST", "/report", {"step": 0, "rank": 1, "kind": "grad", "grad": [32.0, -16.0]})
    # avg=[24,8], /16=[1.5,0.5], w=[1-0.15, -1-0.05]=[0.85,-1.05]
    check("s0 齐全触发更新", r["decision"] and r["decision"]["outcome"] == "updated")
    check("s0 更新后权重正确", vec_close(r["decision"]["weights"], [0.85, -1.05]),
          json.dumps(r["decision"]["weights"]))
    check("s0 倍率不变", r["decision"]["new_scale"] == 16.0 and r["current_step"] == 1)

    # --- step 1: 溢出立即跳步 ---
    st, r = req("POST", "/report", {"step": 1, "rank": 0, "kind": "overflow"})
    check("s1 溢出立即跳过", r["decision"]["outcome"] == "skipped" and r["current_step"] == 2)
    check("s1 倍率减半 16->8", r["decision"]["new_scale"] == 8.0)
    check("s1 权重不变", vec_close(r["decision"]["weights"], [0.85, -1.05]))

    # --- 溢出后的迟报: 只返回该步决议 ---
    st, r = req("POST", "/report", {"step": 1, "rank": 1, "kind": "grad", "grad": [1.0, 1.0]})
    check("s1 迟报标记 late", r["late"] is True and r["accepted"] is False)
    check("s1 迟报返回该步决议(skipped)", r["decision"] and r["decision"]["outcome"] == "skipped")
    st, s = req("GET", "/state")
    check("迟报不改状态", s["step"] == 2 and s["scale"] == 8.0
          and vec_close(s["weights"], [0.85, -1.05]))

    # --- 旧步重试: 返回该步决议 ---
    st, r = req("POST", "/report", {"step": 0, "rank": 0, "kind": "grad", "grad": [0.0, 0.0]})
    check("s0 重试返回 updated 决议", r["late"] and r["decision"]["outcome"] == "updated"
          and vec_close(r["decision"]["weights"], [0.85, -1.05]))

    # --- step 2: 首报为梯度, 之后的溢出重报不生效 ---
    st, r = req("POST", "/report", {"step": 2, "rank": 0, "kind": "grad", "grad": [8.0, 8.0]})
    check("s2 rank0 梯度首报受理", r["accepted"] is True)
    st, r = req("POST", "/report", {"step": 2, "rank": 0, "kind": "overflow"})
    check("s2 rank0 溢出重报被拒(首报唯一)", r["duplicate"] is True and r["decision"] is None)
    st, r = req("POST", "/report", {"step": 2, "rank": 1, "kind": "grad", "grad": [8.0, -8.0]})
    # avg=[8,0], /8=[1,0], w=[0.85-0.1, -1.05]=[0.75,-1.05]
    check("s2 按首报更新", r["decision"]["outcome"] == "updated"
          and vec_close(r["decision"]["weights"], [0.75, -1.05]))
    check("s2 倍率仍为 8", r["decision"]["new_scale"] == 8.0)

    # --- step 3..6: 连续溢出, 倍率 8->4->2->1->1(下限) ---
    expect = [(3, 4.0), (4, 2.0), (5, 1.0), (6, 1.0)]
    for step, new_scale in expect:
        st, r = req("POST", "/report", {"step": step, "rank": 0, "kind": "overflow"})
        check(f"s{step} 溢出, 倍率->{new_scale}",
              r["decision"]["outcome"] == "skipped"
              and r["decision"]["new_scale"] == new_scale
              and r["current_step"] == step + 1)
    st, s = req("GET", "/state")
    check("倍率下限 1, 权重未被溢出改动", s["scale"] == 1.0
          and vec_close(s["weights"], [0.75, -1.05]) and s["step"] == 7)

    # --- 步历史查询 ---
    st, h = req("GET", "/steps/1")
    kinds = [(x["rank"], x["kind"]) for x in h["reports"]]
    check("steps/1 决议为 skipped", h["decision"]["outcome"] == "skipped")
    check("steps/1 仅含已受理的溢出上报(迟报不入库)", kinds == [(0, "overflow")], json.dumps(kinds))

    # --- 留一个挂起上报, 供重启后验证 reports 表持久化 ---
    st, r = req("POST", "/report", {"step": 7, "rank": 0, "kind": "grad", "grad": [1.0, 1.0]})
    check("s7 rank0 首报受理(挂起)", r["accepted"] is True and r["decision"] is None)

    st, s = req("GET", "/state")
    check("phase1 终态 step7/scale1/w[0.75,-1.05]",
          s["step"] == 7 and s["scale"] == 1.0 and vec_close(s["weights"], [0.75, -1.05]))


def phase2():
    # 重启后: 作业状态保留
    st, s = req("GET", "/state")
    check("重启后状态保留 step7/scale1", s["step"] == 7 and s["scale"] == 1.0
          and vec_close(s["weights"], [0.75, -1.05]), json.dumps(s))
    check("重启后已受理上报保留(rank0 挂起)",
          any(r["rank"] == 0 and r["kind"] == "grad" for r in s["reports_received"]),
          json.dumps(s["reports_received"]))

    # 重启后重报旧步: 返回已持久化的决议
    st, r = req("POST", "/report", {"step": 1, "rank": 1, "kind": "grad", "grad": [1.0, 1.0]})
    check("重启后重报 s1 -> skipped 决议", r["late"] and r["decision"]["outcome"] == "skipped")
    st, r = req("POST", "/report", {"step": 0, "rank": 0, "kind": "grad", "grad": [16.0, 32.0]})
    check("重启后重报 s0 -> updated 决议", r["late"] and r["decision"]["outcome"] == "updated"
          and vec_close(r["decision"]["weights"], [0.85, -1.05]))

    # 重启后重报当前步同 rank: 首报(重启前受理)仍然唯一, 溢出重报不生效
    st, r = req("POST", "/report", {"step": 7, "rank": 0, "kind": "overflow"})
    check("重启后 s7 rank0 溢出重报被拒(首报跨重启唯一)",
          r["duplicate"] is True and r["decision"] is None)

    # 补齐梯度 -> 按 scale=1 更新: avg=[2,2]/1, w=[0.75-0.2, -1.05-0.2]=[0.55,-1.25]
    st, r = req("POST", "/report", {"step": 7, "rank": 1, "kind": "grad", "grad": [3.0, 3.0]})
    check("重启后 s7 齐全触发更新", r["decision"] and r["decision"]["outcome"] == "updated")
    check("s7 更新权重正确(scale=1)", vec_close(r["decision"]["weights"], [0.55, -1.25]),
          json.dumps(r["decision"]["weights"]))

    st, h = req("GET", "/steps/7")
    check("steps/7 决议已持久化", h["decision"]["outcome"] == "updated"
          and vec_close(h["decision"]["weights"], [0.55, -1.25]))
    st, h = req("GET", "/steps/0")
    check("steps/0 旧决议仍在", h["decision"]["outcome"] == "updated")


print(f"== verify {PHASE} against {BASE} ==")
if PHASE == "phase1":
    phase1()
elif PHASE == "phase2":
    phase2()
else:
    print(f"unknown phase: {PHASE}")
    sys.exit(2)

print(f"== {'ALL PASS' if not failures else f'{len(failures)} FAILURES: {failures}'} ==")
sys.exit(1 if failures else 0)
