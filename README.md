# 混合精度训练 · 单作业协调 API

FastAPI + SQLite 实现的单作业混合精度训练协调服务。任一进程梯度溢出即跳过整步，
动态损失缩放倍率从 16 起步、溢出时减半（最低 1）。

## 运行

```bash
docker compose up --build
# 打开 http://localhost:8000/docs 进行交互演示
```

本地开发（不用 Docker）：

```bash
pip install -r requirements.txt
DB_PATH=./coordinator.db uvicorn app.main:app --port 8000
```

SQLite 数据库落在 Compose 命名卷 `coordinator-data`（容器内 `/data/coordinator.db`），
服务/容器重启后，已受理的上报、权重、倍率和步决议全部保留。

## 实际实现

### 数据模型（SQLite，3 张表）

| 表 | 内容 | 关键约束 |
|---|---|---|
| `job` | 单行：进程数、学习率、当前权重(JSON)、当前倍率、当前步号 | `id = 1` 单例 |
| `reports` | 已受理的上报：步号、rank、类型(grad/overflow)、梯度(JSON)、受理时间 | `PRIMARY KEY(step, rank)` —— 每进程每步只采纳首报 |
| `decisions` | 每步决议：updated/skipped、本步倍率、新倍率、步后权重、平均梯度、决议时间 | `PRIMARY KEY(step)` —— 每步只有一份决议 |

应用本身无内存状态，SQLite 是唯一事实来源，因此重启天然保留全部状态。

### API

| 端点 | 说明 |
|---|---|
| `POST /job` | 登记作业：进程数、初始权重向量、学习率。倍率初始化为 16，步号从 0 起步。重复登记返回 409 |
| `GET /state` | 进程读取：当前步号、权重、倍率、学习率，及本步已受理的上报列表 |
| `POST /report` | 按 `{step, rank, kind, grad?}` 上报已缩放梯度或溢出（语义见下） |
| `GET /steps/{step}` | 查询某步的决议与已受理上报 |
| `DELETE /job` | 清空全部状态（仅演示/测试用） |
| `GET /healthz` | 健康检查 |

### `POST /report` 语义

- **旧步（step < 当前步）**：迟报/重试。不入库、不改状态，只返回 `decisions` 表中该步已记录的决议（`late: true`）。
- **未来步（step > 当前步）**：409 拒绝。
- **当前步**：`INSERT OR IGNORE` 写入 `reports`——已有同 `(step, rank)` 记录则视为重报，拒绝且首报内容不变（`duplicate: true`）。随后在同一事务内检查：
  - 本步存在任一 overflow → **立即**决议 `skipped`：跳过更新、倍率减半（`max(1, scale/2)`）、权重不变，进入下一步。不等其余进程。
  - 否则本步上报数 == 进程数 → 决议 `updated`：梯度取平均、除以本步倍率、`w -= lr * grad` 更新一次，倍率不变，进入下一步。
- 决议与步进在同一 `BEGIN IMMEDIATE` 事务中提交；`decisions.step` 主键保证每步至多一份决议，权重至多更新一次。
- 参数校验：rank 越界、梯度维度与权重不符、`kind=grad` 缺梯度均返回 422。
- 迟报/重报**不写入** `reports` 表（`GET /steps/{step}` 只反映已受理的首报）。

### 并发与一致性

- 写路径：进程内 `threading.Lock` + SQLite `BEGIN IMMEDIATE` 事务，读-改-写整体串行。
- 首报唯一由 `reports` 主键 + `INSERT OR IGNORE` 保证；每步单决议由 `decisions` 主键保证。
- 已用 20 并发请求（4 rank × 5 次重复）验证：恰好 4 条受理、1 次更新、权重正确。

## 验证结果

本环境无 Docker，Compose 配置未实际构建；服务在本地以 uvicorn 直接运行，
用 `scripts/verify.py`（仅标准库）完成两阶段验证，**全部通过**：

```bash
python3 scripts/verify.py phase1   # 主流程（31 项断言全 PASS）
# 杀掉服务进程，用同一数据库重启
python3 scripts/verify.py phase2   # 重启后（9 项断言全 PASS）
```

phase1 覆盖（作业：2 进程、权重 `[1.0, -1.0]`、lr=0.1）：

- 登记/重复登记 409；初始状态 step=0、scale=16；rank 越界/维度不符/未来步的 4xx 校验
- step0：两进程梯度 `[16,32]`、`[32,-16]` → 平均 `[24,8]` ÷16 ×0.1 → 权重更新为 `[0.85, -1.05]`，倍率不变
- 同 `(step, rank)` 重报被拒，首报内容不变
- step1：rank0 报溢出 → **立即** skipped，倍率 16→8，权重不变，进入 step2
- **溢出后的迟报**：rank1 对 step1 补报梯度 → 只返回该步 skipped 决议，状态不变
- 旧步重试：重报 step0 → 返回该步 updated 决议
- step2：rank0 首报梯度后，其溢出重报被拒（首报唯一），按梯度正常更新（倍率 8）
- step3–6 连续溢出：倍率 8→4→2→1→**1（下限）**，权重始终不变
- `GET /steps/1`：决议为 skipped，且只含已受理的溢出上报（迟报不入库）

phase2（杀掉进程、用同一 SQLite 文件重启后）覆盖：

- **重启后状态保留**：step=7、scale=1、权重 `[0.75, -1.05]`，重启前受理的挂起上报仍在
- **重启后重报旧步**：重报 step1/step0 → 分别返回已持久化的 skipped/updated 决议
- **重启后重报当前步同 rank**：首报跨重启唯一，溢出重报被拒、不触发跳步
- 补齐梯度后按 scale=1 正确更新为 `[0.55, -1.25]`；`GET /steps/{0,7}` 决议均可查

另做并发测试：4 rank × 5 次重复共 20 个并发上报 → 恰好 4 条受理、1 次更新决议、
权重只更新一次（`[-0.1, -0.1]`），late 响应均带回该步决议。全 PASS。

`/docs`（Swagger UI）与 `/openapi.json` 均验证返回 200，可直接在页面演示上述流程。

## 未完成项 / 已知限制

- **Docker 未实测**：本环境无 Docker，`Dockerfile`/`docker-compose.yml` 经过静态检查（YAML 合法、
  依赖版本与本地实测一致）但未实际构建运行；本地等效命令（uvicorn + 同一 DB 路径）已全量验证。
- **单作业、单实例**：只支持一个作业；多副本水平扩展未做（SQLite 单写者，跨进程正确性依赖
  `BEGIN IMMEDIATE` 串行化，多实例需换 PostgreSQL 等外部库）。
- **无鉴权**：所有端点（含 `DELETE /job` 重置）无认证授权，仅限可信内网/演示。
- **无进程存活跟踪**：进程掉线不上报时会永远等待，无超时/心跳/占位跳过机制。
- **倍率只降不升**：按需求只实现溢出减半（下限 1），未实现连续成功后升倍率的策略。
- **步号从 0 起步**、时间戳为 UTC（`datetime('now')`），如需 1 起步或本地时区可调整。
- 权重/梯度以 JSON 数组存 SQLite，适合演示与小向量；大规模向量需换二进制/外部存储。
