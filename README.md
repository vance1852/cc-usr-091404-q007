# 生物制剂效价复测会审系统

细胞法效价检测的**会审（quality review）系统**：保存实验方法版本、板图、
标准品批号、稀释序列、原始读数与拟合参数；执行确定性四参数逻辑（4PL）
拟合；按**当时生效的规则集**判定板有效性；对排除孔完整留痕；复测须
多角色审批；批次结论按预置组合策略纳入谱系内**全部**有效测定，杜绝
只挑选有利结果。

纯 Python 标准库实现（无 numpy / 无第三方依赖），SQLite 持久化。

## 它解决的问题

> 一块读数离散的细胞法检测板，第一次计算恰低于放行限，更换算法参数/
> 排除孔后结果却"合格"。主管担心选择性复测掩盖真实失效。

系统的回答：

1. **原始读数不可修改**（SQLite 触发器禁止 UPDATE/DELETE），结论可一路
   追溯到未经改动的原始值与导入载荷（SHA-256 内容哈希）。
2. **分析版本只追加**：首算失败与排孔后"合格"两个版本都保留，可逐参数
   比较曲线、效价、平行性与规则命中差异。
3. **每个排除孔**记录操作者、理由，以及该孔排除前/后的效价与百分比差异。
4. **规则集带生效日期**：无效判定遵循分析当时生效的版本（v1.0 / v1.1），
   历史结论永远可按旧规则复算。
5. **首轮结论锁定**后才允许申请复测；复测须 **SUPERVISOR + QA 两个不同
   角色**分别批准，且申请人不能自批。
6. **批次组合结论**强制纳入同一谱系的全部锁定板（含失败的首轮板），
   待审批复测申请必须先了结，禁止挑数据。
7. **重复导入识别**：规范化内容哈希，即使改板号/文件名再次上传也返回
   409 并指回既有板。

## 目录结构

```
potency/
  fitting.py    确定性 4PL/LM 拟合、相对效价、平行性 F 检验、精密度、有效范围
  rules.py      带生效日期的规则集（v1.0 / v1.1）与纯函数评估
  db.py         SQLite schema、只读触发器、内置种子数据
  service.py    业务工作流（导入/版本/排孔/锁定/复测审批/批次结论/追溯）
  api.py        JSON HTTP API（标准库 http.server）
tests/          数值、规则、服务工作流、HTTP 端到端、落盘持久化测试
scripts/
  seed_demo.py  端到端演示（离散板 → 翻转 → 锁定 → 复测 → 组合结论）
```

## 快速开始

```bash
python3 scripts/seed_demo.py           # 端到端演示，生成 demo_potency.db
python3 -m potency.api --db demo_potency.db --port 8080
python3 -m unittest discover -s tests  # 运行全部测试（60 个）
```

演示输出的关键数字：首轮 77.15%（低于 80% 放行限）→ 排除签认离群孔
E09 后 80.85%（差异 +4.80% 已留痕）→ 首轮仍以 v1 锁定 → 双人批准复测
板 94.39% → 只挑好板被 `LINEAGE_INCOMPLETE` 拒绝 → 纳入两板算术平均
85.77% 放行。

## 数值模块（`potency/fitting.py`）

* 模型：`y = A + (D-A)/(1+(x/C)^B)`（A 底、D 顶、C EC50、B Hill 斜率）
* 优化：自实现 Levenberg–Marquardt（高斯消元 + 差分雅可比，固定初值、
  固定迭代策略），**无随机数、无哈希序、无线程**
* 确定性：输入点按 `(剂量, 响应)` 规范化排序，重复调用逐位一致，且与
  调用方给出的孔序无关
* 相对效价：标准品/样品**共斜率**联合拟合，`Pr = EC50_std/EC50_smp ×
  标示效价`；delta 法给出对数尺度 95% 置信区间
* 平行性：共斜率（约束）与自由斜率模型的 **F 检验**
* 精密度：剂量级重复 SD/CV、合并 CV；有效范围（标准品实测剂量跨度，
  样品 EC50 须落于其内）；反算浓度与回收率%
* 内置正则化不完全 Beta，提供 F / t 分布 CDF（已对照标准 F/t 表测试）

## 规则集（`potency/rules.py`）

| 规则 | 级别 | 含义 |
|---|---|---|
| `FIT_CONVERGED` | 无效 | 4PL 必须收敛 |
| `PARALLELISM_P` | 无效 | 共斜率/自由斜率 F 检验 p ≥ 0.05 |
| `MIN_DOSE_LEVELS` | 无效 | ≥4 个剂量水平 |
| `MIN_REPLICATES` | 无效 | 每剂量 ≥2 重复孔 |
| `STD/SMP_REPLICATE_CV` | 无效 | 重复 CV ≤ 20% |
| `QUANT_RANGE` | 无效 | 样品 EC50 落在标准品剂量范围内 |
| `STD/SMP_BACKCALC_RECOVERY` | 警告 | 反算回收率 80–120% |
| `POTENCY_CI_WIDTH` | v1.0 警告 / **v1.1 无效** | 95% CI 半宽 ≤25% / ≤20% |

放行限：相对效价占标示效价 80–125%。评估是纯函数
`evaluate_plate(fit_dict, ruleset)`，可对任意历史版本重放。

## API 一览

鉴权：请求头 `X-User: <username>`（内置 `analyst.li` /
`supervisor.wang` / `qa.zhao` / `reviewer.chen`）。

```
POST /plates                          导板（409 DUPLICATE_IMPORT 识别重复）
GET  /plates/{id}
POST /plates/{id}/analyses            新增只追加分析版本（body 可带 exclusions）
GET  /plates/{id}/analyses
GET  /analyses/{id}
GET  /analyses/{a}/compare/{b}        曲线/效价/规则命中/排孔差异
POST /plates/{id}/lock                SUPERVISOR 锁定首轮结论
POST /plates/{id}/retest-requests     锁定后才能申请
POST /retest-requests/{id}/decision   SUPERVISOR 与 QA 分别 approved/rejected
POST /batches                         创建批次（指定组合策略）
POST /batches/{id}/plates
POST /batches/{id}/conclude           QA 结论（反挑选校验）
GET  /batches/{id}/trace              结论 → 每板原始读数 + 内容哈希
GET  /rulesets                        查看规则集版本与生效日期
GET  /audit?entity=&id=               审计日志
```

## 组合策略与反挑选规则

`conclude_batch` 在结论前强制执行：

1. **谱系完整性**：纳入批次的任一板，其同 `lineage_key` 的所有板都必须
   纳入，否则 `LINEAGE_INCOMPLETE`（即不能只留有利的复测板）。
2. **全部锁定**：谱系内每块板必须已锁定首轮结论。
3. **无悬挂复测**：谱系不能存在 `pending` 复测申请（`PENDING_RETEST`）。
4. **已批准复测板必须纳入**（`RETEST_MISSING`）。
5. 策略 `MEAN_ALL_VALID` / `GMEAN_ALL_VALID` 对全部板做算术/几何平均；
   任一板无效或组合值超出 80–125% 即不放行。

## 测试

```bash
python3 -m unittest discover -s tests -v
```

覆盖：4PL 参数恢复、相对效价、CI、平行性阳性/阴性、F/t 分布表值、
确定性与孔序无关、重复导入、原始数据触发器、版本只追加、排孔前后
差异、规则时点选择与历史复算、锁定冻结、双人审批/自批禁止/拒绝闭环、
谱系反挑选、算术/几何组合、HTTP 全流程、落盘重开。
