#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR" || exit 1
PORT=7866
export MFLUX_NO_IDLE=1
export PYTHONUNBUFFERED=1
echo "=================================================="
echo "🎨 mflux-paint WebUI 已就绪"
echo "🌐 请在浏览器中访问: http://127.0.0.1:${PORT}"
echo "✨ 已集成模型与功能:"
echo "   - Qwen-Image-2.1 · Turbo Edit (v0.2.1 r128 / r256, 6 或 8 步)"
echo "   - Qwen-Image-2.1 · Quality Edit (40 steps)"
echo "   - Qwen-Image-2.1 · Turbo Text-to-Image (v0.2.1 r128 / r256, 6 或 8 步)"
echo "   - Qwen-Image-2.1 · Quality Text-to-Image"
echo "   - ✨ AI 提示词重写 (已打通本地 oMLX Qwen3.6-35B)"
echo "   - 🎨 原生透明图像生成 (RGBA 4通道支持已打通)"
echo "   - ⏱️ 实时与最终耗时统计展示 (精确到秒)"
echo "=================================================="
exec /Users/vanch/mflux-pr736/.venv/bin/python -u server.py
