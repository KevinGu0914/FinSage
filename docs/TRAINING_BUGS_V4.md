# MARFT V4 训练问题追踪

**创建时间**: 2025-12-18 23:46 UTC
**训练状态**: Epoch 1/3 进行中
**服务器**: 3台 (Aggressive, Balanced, Adaptive)

---

## 🐛 待修复问题列表

### 问题 1: SharedModelExpertManager 缺少 create_completion 方法
**严重程度**: ⚠️ 警告 (不阻断训练)
**影响**: ManagerCoordinator 无法使用 LLM 选择 sizing/hedging 策略
**出现位置**: 所有3个服务器
**日志示例**:
```
[WARNING] Sizing method selection failed: 'SharedModelExpertManager' object has no attribute 'create_completion'
[WARNING] Hedging strategy selection failed: 'SharedModelExpertManager' object has no attribute 'create_completion'
```

**根因分析**:
- `ManagerCoordinator` 调用 `llm_provider.create_completion()`
- 但 `SharedModelExpertManager` 没有实现这个方法
- 系统回退到默认的 `risk_parity` 策略

**修复方案**:
在 `finsage/rl/shared_expert_manager.py` 添加 `create_completion` 方法:
```python
def create_completion(self, prompt: str, **kwargs) -> str:
    """兼容 LLMProvider 接口的简单文本生成"""
    messages = [{"role": "user", "content": prompt}]
    response = self.generate_batch([messages], **kwargs)
    if response and len(response) > 0:
        return response[0].get("content", "")
    return ""
```

---

### 问题 2: expert_reports 传入字典而非 ExpertReport 对象
**严重程度**: ⚠️ 警告 (不阻断训练)
**影响**: ManagerCoordinator 无法访问 overall_view 属性
**出现位置**: 所有3个服务器
**日志示例**:
```
[WARNING] Manager coordination failed: 'dict' object has no attribute 'overall_view'
```

**根因分析**:
- `train_with_real_data_v4.py` 第1984-1990行构建 `expert_reports` 时使用字典
- 但 `ManagerCoordinator._synthesize_expert_reports()` 期望 `ExpertReport` 对象
- 需要访问 `.overall_view` 属性

**修复方案**:
在 `scripts/train_with_real_data_v4.py` 修改 expert_reports 构建:
```python
from dataclasses import dataclass

@dataclass
class ExpertReport:
    action: str
    confidence: float
    reasoning: str
    overall_view: str = ""

# 构建时使用:
expert_reports = {}
for role, action in all_actions.items():
    expert_reports[role] = ExpertReport(
        action=action.get("action", "HOLD"),
        confidence=action.get("confidence", 0.5),
        reasoning=action.get("reasoning", ""),
        overall_view=action.get("reasoning", "")  # 用 reasoning 作为 overall_view
    )
```

---

### 问题 3: JSON 格式不匹配 - Missing 'action' field
**严重程度**: 🟡 低 (已有 fallback，偶发)
**影响**: 个别步骤 LLM 返回格式不符预期
**出现位置**: S2 (Balanced) - 仅1次
**日志示例**:
```
[WARNING] [PARSE] Missing 'action' field in JSON: {
    "GOOGL": {
        "action": "BUY_25%",
        ...
```

**根因分析**:
- LLM 返回了 per-stock 格式 `{"GOOGL": {"action": ...}}`
- 但解析代码期望顶层 `{"action": ...}` 格式
- 系统已有 fallback 处理，回退到默认 action

**修复方案**:
在解析逻辑中增加对 per-stock 格式的支持（低优先级）

---

## ✅ 已修复问题

### 问题 A: JSON 截断错误 (已修复)
**修复时间**: 2025-12-18 23:30 UTC
**问题**: `max_new_tokens=256` 导致 JSON 输出被截断
**修复**: 改为 `max_new_tokens=512`
**文件**: `finsage/rl/shared_expert_manager.py` 第458行

### 问题 B: ManagerCoordinator 初始化失败 (已修复)
**修复时间**: 2025-12-18 23:28 UTC
**问题**: `PortfolioManager.__init__() missing 1 required positional argument: 'hedging_toolkit'`
**修复**: 添加 `hedging_toolkit = HedgingToolkit()` 参数
**文件**: `scripts/train_with_real_data_v4.py`

---

## 📊 训练状态监控

### 最新检查 (2025-12-18 23:56 UTC)

| 服务器 | 策略 | 显存使用 | GPU利用率 | 日志行数 | 状态 |
|--------|------|----------|-----------|----------|------|
| S1 (174.78.228.101:40726) | Aggressive | 54GB | 58% | 948行 | ✅ 运行中 |
| S2 (49.213.134.9:18109) | Balanced | 51GB | 60% | ~900行 | ✅ 运行中 |
| S3 (173.207.82.240:40038) | Adaptive | 33GB | 46% | ~900行 | ✅ 运行中 |

### 训练进度
- **当前**: Epoch 1/3, 多个 trading days 完成
- **资产数**: 30 assets (factor-driven)
- **交易执行**: 正常 (10-21 trades per step)
- **奖励计算**: 正常 (Alpha 和 Team Reward 计算中)

### Bug 统计
| Bug | S1 | S2 | S3 | 总计 |
|-----|----|----|----|----|
| create_completion missing | 多次 | 多次 | 多次 | 每步3次 |
| overall_view missing | 多次 | 多次 | 多次 | 每步1次 |
| JSON 格式不匹配 | 0 | 1 | 0 | 1次 |

---

## 📝 下一轮训练待办

1. [ ] 修复 `create_completion` 方法
2. [ ] 修复 `expert_reports` 使用 ExpertReport 对象
3. [ ] 考虑增加 `max_new_tokens` 到 768 (预防更长 JSON)
4. [ ] 添加更详细的 epoch 进度日志

---

**最后更新**: 2025-12-19 00:47 UTC

---

## ✅ 2025-12-19 00:47 - 所有 Bug 已修复并重启

### 修复的问题:
1. `create_completion` - 支持 `messages` 参数格式
2. `overall_view` - 使用 SimpleNamespace 支持属性访问
3. `recommendations` - 添加空列表默认值
4. JSON 格式 - 支持 per-stock 嵌套格式

### 当前训练状态:
- S1 (Aggressive): 79GB GPU, 运行中, 0个原 bug 警告
- S2 (Balanced): 60GB GPU, 运行中, 0个原 bug 警告
- S3 (Adaptive): 33GB GPU, 运行中, 0个原 bug 警告

---

## ✅ 2025-12-19 01:55 - DataFrame Bug 修复并重启

### 发现的新问题:
**问题**: "The truth value of a DataFrame is ambiguous"
**严重程度**: 🔴 致命 (导致训练崩溃)
**根因**: `len(returns_window) > 0` 对 pandas DataFrame 产生歧义
**日志示例**:
```
ValueError: The truth value of a DataFrame is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
```

### 修复方案:
在 `scripts/train_with_real_data_v4.py` 修改:
```python
# 修复前 (错误):
"returns": returns_window if len(returns_window) > 0 else pd.DataFrame()

# 修复后 (正确):
"returns": returns_window if not returns_window.empty else pd.DataFrame()
```

### 修复后重启状态 (2025-12-19 01:55 UTC):
| 服务器 | 策略 | 显存使用 | GPU利用率 | 当前交易日 | 状态 |
|--------|------|----------|-----------|------------|------|
| S1 (174.78.228.101:40726) | Aggressive | 32GB | 61% | 2023-03-09 | ✅ 运行中 |
| S2 (49.213.134.9:18109) | Balanced | 32GB | 62% | 2023-02-23 | ✅ 运行中 |
| S3 (173.207.82.240:40038) | Adaptive | 31GB | 49% | 2023-02-15 | ✅ 运行中 |

### 训练特征:
- 无 DataFrame 歧义错误
- 正常执行交易 (10-20 trades per step)
- Alpha 和 Team Reward 正常计算
- 30 assets factor-driven universe
