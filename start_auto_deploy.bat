@echo off
echo Installing auto-deployment dependencies...
pip install -r auto_deploy_requirements.txt

echo Starting auto-deployment monitor...
python auto_deploy.py

pause
