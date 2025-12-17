# Git 协作规则 - 老师优先

## 黄金法则
**每次修改代码前，必须先拉取老师的最新修改！**

## 日常工作流程

```bash
# 1. 开始工作前，先获取老师的最新修改
git pull

# 2. 修改代码...

# 3. 提交你的修改
git add .
git commit -m "描述你的改动"

# 4. 推送前再次检查老师有没有新提交
git pull

# 5. 推送到 GitHub
git push
```

## 遇到冲突时（你和老师改了同一处）

### 方法1：采用老师的版本（推荐）
```bash
git checkout --theirs <冲突文件>
git add <冲突文件>
git commit -m "采用老师的修改"
git push
```

### 方法2：完全重置为老师的版本
```bash
git fetch origin
git reset --hard origin/main
```

## 查看状态命令

```bash
# 查看本地有哪些修改
git status

# 查看与远程的差异
git diff origin/main

# 查看提交历史
git log --oneline -10
```

## 注意事项

1. **永远不要** 在没有 `git pull` 的情况下直接 `git push`
2. **遇到冲突**时，优先保留老师的修改
3. **不确定时**，先备份你的修改，再拉取老师的版本
4. `.env` 文件不会同步（已在 .gitignore 中排除）

## 快速命令（复制粘贴即可）

```bash
# 同步并推送（一步完成）
git pull && git add . && git commit -m "更新" && git push
```
