@echo off
rem Full pretraining re-run. Each stage logs to reproduction\work\*.log and is safe
rem to re-run: finished downloads, cleaning and validation splits are skipped.
rem   1. English Gutenberg texts: unpack txt-files.tar.zip if it is in work\
rem      (fast; https://www.gutenberg.org/cache/epub/feeds/txt-files.tar.zip),
rem      then fetch anything missing book by book       -> extract.log, download.log
rem   2. clean + validation split (train_tokenizer.py) -> prepare.log
rem   3. score the original checkpoints                -> eval.log, eval_results.json
rem   4. pretraining with GPT2_model.py                -> train.log, runs\, checkpoints\
rem Optional while training (from work\): python ..\snapshot_checkpoints.py 1
rem                                       python ..\stop_at_target.py
rem Set PY to pick a specific Python (default: python on PATH).

if not defined PY set PY=python
set PYTHONUNBUFFERED=1
set PYTHONUTF8=1

if not exist "%~dp0work" mkdir "%~dp0work"
cd /d "%~dp0work"
copy /y "..\..\bpe_tokenizer.json" . >nul

echo [%date% %time%] download >> pipeline.log
if not exist pg_catalog.csv curl -s -o pg_catalog.csv https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv
if exist txt-files.tar.zip "%PY%" ..\extract_archive.py txt-files.tar.zip >> extract.log 2>&1 || goto :fail
"%PY%" ..\download_gutenberg.py >> download.log 2>&1 || goto :fail

echo [%date% %time%] prepare >> pipeline.log
"%PY%" ..\prepare_data.py >> prepare.log 2>&1 || goto :fail

echo [%date% %time%] eval original checkpoints >> pipeline.log
if exist "..\..\checkpoints\final.pt" "%PY%" ..\eval_checkpoints.py "original_jun17=..\..\checkpoints\final - Copy.pt" "original_final=..\..\checkpoints\final.pt" "original_latest=..\..\checkpoints\latest.pt" >> eval.log 2>&1 || goto :fail

echo [%date% %time%] pretraining >> pipeline.log
call "%~dp0run_pretraining.bat"
echo [%date% %time%] done (exit %errorlevel%) >> pipeline.log
exit /b 0

:fail
echo [%date% %time%] FAILED (exit %errorlevel%) >> pipeline.log
exit /b 1
