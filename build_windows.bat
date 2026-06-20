@echo off
REM ============================================================
REM   AI 数字对战 - Windows 一键打包脚本
REM   双击运行即可生成 ai_game.exe
REM ============================================================

setlocal enabledelayedexpansion
chcp 65001 >nul

echo ============================================================
echo   AI 数字对战 - Windows 打包脚本
echo ============================================================
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.10+
    echo 下载地址: https://www.python.org/downloads/
    echo 安装时请勾选 "Add Python to PATH"
    pause
    exit /b 1
)

echo [1/5] 创建虚拟环境...
python -m venv venv
call venv\Scripts\activate.bat

echo [2/5] 安装依赖...
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy matplotlib pyinstaller

echo [3/5] 验证脚本可运行...
python ai_game.py stats
if errorlevel 1 (
    echo [错误] 脚本运行失败，请检查错误信息
    pause
    exit /b 1
)

echo [4/5] 打包成 exe（可能需要 5-10 分钟）...
pyinstaller --onefile --name ai_game ^
    --add-data "scripts;scripts" ^
    --add-data "models\model.pt;." ^
    --add-data "models\genetic_model.pt;." ^
    --hidden-import torch ^
    --collect-all torch ^
    --collect-all numpy ^
    ai_game.py

if errorlevel 1 (
    echo [错误] 打包失败
    pause
    exit /b 1
)

echo [5/5] 打包完成！
echo.
echo ============================================================
echo   成功！可执行文件位于:
echo   dist\ai_game.exe
echo ============================================================
echo.
echo 使用方法:
echo   dist\ai_game.exe              进入交互式菜单
echo   dist\ai_game.exe stats        查看模型统计
echo   dist\ai_game.exe watch        观战 AI 自对战
echo   dist\ai_game.exe human        人机对弈
echo.
pause
