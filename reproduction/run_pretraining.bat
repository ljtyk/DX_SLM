@echo off
rem Run the original, unmodified GPT2_model.py with reproduction\work as the working
rem directory, so its relative paths (cache\, checkpoints\, runs\) stay inside work\.
rem Resumes from work\checkpoints\latest.pt if it exists.
rem
rem model.compile() builds Triton kernels, which on Windows needs the MSVC compiler.
rem Set VCVARS if Visual Studio Build Tools are installed somewhere else, and PY to
rem pick a specific Python (default: python on PATH).

if not defined PY set PY=python
if not defined VCVARS set VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat
if exist "%VCVARS%" call "%VCVARS%" >nul
set CC=cl
set PYTHONUNBUFFERED=1
rem GPT2_model.py prints emoji; without this, redirected output can fail on cp1252.
set PYTHONUTF8=1

cd /d "%~dp0work"
copy /y "..\..\bpe_tokenizer.json" . >nul
"%PY%" "..\..\GPT2_model.py" >> train.log 2>&1
