@echo off
rem * ===================
echo Install Python venv
rem * ===================
pushd %~dp0
echo %CD%
echo.=================================================================================
echo.Create pywin32 folder and downloand python-3.9.13-embed-win32 from https://www.python.org/ftp/python/3.9.13/python-3.9.13-embed-win32.zip
echo.Create virtualenv
call python.exe -m pip install virtualenv
call python.exe -m virtualenv env
echo.-----------------------------------------------------------------------------------
echo.Activate virtualenv
call env\Scripts\activate
echo %CD%
echo.-----------------------------------------------------------------------------------
echo.Install durguestprofile
call CD..
echo %CD%
call python.exe -m pip install --trusted-host pypi.python.org --trusted-host pypi.org --trusted-host files.pythonhosted.org --no-warn-script-location durguestprofile
call pip freeze
echo.-----------------------------------------------------------------------------------
echo.vscode will be downloaded......
call explorer "https://code.visualstudio.com/docs/?dv=winzip"
popd
pause