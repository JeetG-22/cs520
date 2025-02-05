#!/bin/bash
python3 -m venv venv

# Detect Windows (PowerShell or Command Prompt)
if [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "win32"* ]]; then
    # For Windows
    if [ -n "$WINDIR" ]; then
        # PowerShell: Activate with PowerShell script
        . .\venv\Scripts\Activate.ps1
    else
        # Command Prompt (or Git Bash): Activate with batch script
        . .\venv\Scripts\activate
    fi
else
    # For Linux and macOS
    source venv/bin/activate
fi

pip install -r requirements.txt