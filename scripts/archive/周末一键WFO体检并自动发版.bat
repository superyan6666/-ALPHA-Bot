@echo off
chcp 65001 >nul
echo =======================================================
echo     🤖 AI 量化周末体检与云端同步程序 (WFO Evaluator)
echo =======================================================
echo.

echo [1/4] 正在与云端 Hermes 同步最新状态...
git pull origin main

echo.
echo [2/4] 正在启动 i5 本地狂暴算力，执行 WFO 滚动衰减评估...
python agent_auto_eval.py

echo.
echo [3/4] 正在生成下周生产环境的最新量化模型...
python train_prod_model.py

echo.
echo [4/4] 正在执行自动防退化阻断检查...
python auto_gate.py
IF ERRORLEVEL 1 (
    echo.
    echo ❌ 门控未通过！模型存在过拟合或退化，已终止向甲骨文云端发布！
    echo 请检查 eval_results_wfo.log 查看详情。
    pause
    EXIT /B
)

echo.
echo ✅ 模型质检通过！准备打包最新高胜率模型推送到云端...
git add .quantbot_data/*.pth .quantbot_data/*.json auto_gate.py eval_results*.log
git commit -m "[bot] Weekend automated model upgrade"
git push origin main

echo.
echo 🎉 发布成功！甲骨文 Hermes 已准备好在下周一使用新模型战斗！
pause
