@ECHO OFF

pushd %~dp0

REM Minimal Sphinx make.bat for Windows.

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=.
set BUILDDIR=_build

if "%1" == "" goto help

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Install pybmodes with the
	echo.[docs] extra: pip install -e ".[docs]"
	echo.
	exit /b 1
)

if "%1" == "strict" (
	%SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR% -W --keep-going %SPHINXOPTS%
	goto end
)

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%

:end
popd
