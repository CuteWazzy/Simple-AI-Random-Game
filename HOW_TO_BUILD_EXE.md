# 如何生成 Windows .exe 文件

本文档提供两种方案，任选其一即可生成 `ai_game.exe`。

---

## 方案 A：本地一键打包（推荐）

### 准备工作

1. **安装 Python 3.10+**
   - 下载: https://www.python.org/downloads/
   - 安装时**务必勾选** "Add Python to PATH"

2. **下载源码包**
   - 解压 `ai_game_local.zip` 到任意目录（例如 `D:\ai_game`）

### 打包步骤

1. 打开解压后的目录（应能看到 `ai_game.py`、`build_windows.bat` 等）

2. **双击 `build_windows.bat`**

3. 等待 5-10 分钟（会自动安装 PyTorch + PyInstaller 并打包）

4. 完成后，在 `dist\` 目录下会生成 `ai_game.exe`（约 500MB）

### 使用方法

```cmd
dist\ai_game.exe              :: 交互式菜单
dist\ai_game.exe stats        :: 查看模型统计
dist\ai_game.exe watch        :: 观战 AI 自对战
dist\ai_game.exe watch 0 0.5 80  :: 指定种子0、温度0.5、80回合
dist\ai_game.exe human        :: 人机对弈
dist\ai_game.exe human 0      :: 你坐 P1 位置
```

### 手动打包（如果 bat 脚本失败）

```cmd
:: 1. 创建虚拟环境
python -m venv venv
venv\Scripts\activate

:: 2. 安装依赖
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy matplotlib pyinstaller

:: 3. 打包
pyinstaller --onefile --name ai_game ^
    --add-data "scripts;scripts" ^
    --add-data "models\model.pt;." ^
    --add-data "models\genetic_model.pt;." ^
    --hidden-import torch ^
    --collect-all torch ^
    --collect-all numpy ^
    ai_game.py

:: 4. 生成物在 dist\ai_game.exe
```

---

## 方案 B：GitHub Actions 云端打包（无需本地环境）

如果你不想本地装 Python，可以用 GitHub 的免费 CI 在云端打包。

### 步骤

1. **注册 GitHub 账号**（如已有可跳过）
   - https://github.com/signup

2. **创建新仓库**
   - https://github.com/new
   - 名称随意，例如 `ai-game`
   - 设为 Public 或 Private 均可

3. **上传源码**
   
   方法一：网页上传
   - 在新仓库页面点击 "uploading an existing file"
   - 把 `ai_game_local.zip` 解压后的所有文件拖进去
   - 注意要包含 `.github` 隐藏文件夹（ workflows 在里面）
   - Commit changes

   方法二：Git 命令
   ```bash
   cd ai_game_local
   git init
   git add .
   git commit -m "init"
   git branch -M main
   git remote add origin https://github.com/你的用户名/ai-game.git
   git push -u origin main
   ```

4. **手动触发打包**
   - 进入仓库页面
   - 点击 "Actions" 标签
   - 左侧选择 "Build Windows EXE"
   - 右侧点击 "Run workflow" → "Run workflow"
   - 等待约 10-15 分钟

5. **下载 exe**
   - 打包完成后，点击对应的 workflow run
   - 滚动到页面底部 "Artifacts" 区域
   - 点击 `ai_game-windows-exe` 下载 zip
   - 解压得到 `ai_game.exe`

6. **（可选）发布 Release**
   ```bash
   git tag v1.0
   git push origin v1.0
   ```
   打 tag 后会自动创建 Release 并附加 exe

---

## 两种方案对比

| 维度 | 方案 A（本地） | 方案 B（GitHub Actions） |
|------|---------------|-------------------------|
| 需要 Python | ✓ | ✗ |
| 需要联网 | ✓（装依赖） | ✓ |
| 打包时间 | 5-10 分钟 | 10-15 分钟 |
| 产物大小 | ~500MB | ~500MB |
| 可重复 | ✓ | ✓ |
| 免费 | ✓ | ✓（GitHub 公开仓库免费） |
| 适合场景 | 本地有 Python 环境 | 不想装 Python / 想自动化 |

---

## 常见问题

### Q: 打包后 exe 体积太大怎么办？
A: PyTorch 本身就有 500MB+。如果只要更小的包，可以：
- 不打包 torch，改用 onnxruntime 推理（需要改代码）
- 用 `--exclude-module` 排除不需要的模块

### Q: exe 启动慢（首次 5-10 秒）？
A: 这是 PyInstaller 单文件模式的正常现象，因为要解压内部依赖到临时目录。改用目录模式（去掉 `--onefile`）可加快启动。

### Q: 杀毒软件报毒？
A: PyInstaller 打包的 exe 经常被误报。可以：
- 添加到杀毒软件白名单
- 用方案 B 在 GitHub Actions 打包（云端打包更不容易被误报）
- 给 exe 签名（需要代码签名证书）

### Q: 想要 Mac 版怎么办？
A: 把 workflow 里的 `runs-on: windows-latest` 改成 `macos-latest` 即可。

### Q: 打包失败怎么办？
A: 检查：
1. Python 版本是否 3.10+
2. 是否在虚拟环境里
3. 网络是否能访问 pypi.org
4. 查看 `build\ai_game\warn-ai_game.txt` 警告信息

---

## 当前模型信息

```
代数: 430
最佳种群: C_tanh_conservative
历史最佳胜率: 61.1%
参数量: 52252
```

打包完成后，`ai_game.exe` 会自动加载内置模型，无需额外配置。
