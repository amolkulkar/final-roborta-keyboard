@echo off
setlocal
pushd "%~dp0"
if not exist logs mkdir logs
"C:\Users\asokr\OneDrive\Desktop\Roberta Final year\final-roborta-keyboard\.venv\Scripts\python.exe" -B src\main.py >> logs\run.log 2>&1
popd
endlocal
