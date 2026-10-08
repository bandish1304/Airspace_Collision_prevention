@echo off
REM One OpenSky polling session, for Windows Task Scheduler.
REM Scheduled six times daily at four-hour spacing to cover the full
REM 24-hour traffic cycle. 30 minutes at 10s spacing is ~180 credits
REM per session, ~1080 per day against a 4000/day allowance.

cd /d "C:\Users\bandish1304\Desktop\Airspace collision prevention"
".venv\Scripts\python.exe" -m src.data.poll_opensky --session-minutes 30 --interval-seconds 10 >> "data\raw\sessions\poll.log" 2>&1
