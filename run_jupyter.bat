@echo off
title Jupyter Lab Auto Launcher

:: 3. 환경 활성화
call conda activate env_aias_test

:: 4. 주피터 랩 실행
jupyter lab

pause