@echo off
cd /d C:\Users\Javohir\Desktop\binance_bot
call .venv\Scripts\activate.bat

:loop
echo [%date% %time%] Bot starting...
python -u main.py
echo [%date% %time%] Bot stopped/crashed. Restarting in 10 seconds...
timeout /t 10 /nobreak
goto loop