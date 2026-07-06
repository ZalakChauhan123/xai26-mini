@echo off
setlocal EnableDelayedExpansion

REM =============================================================================
REM model.bat
REM -----------------------------------------------------------------------------
REM Runs the whole XAI-on-RDF pipeline in one go:
REM   1. Data analysis                (notebooks\1__analysis.py)
REM   2. Graph pre-processing         (notebooks\2__Graph_PreProcessing.py)
REM   3. R-GCN model training         (notebooks\3__model_train.py)
REM   4. GNNExplainer explainability  (notebooks\4__Explainability.py)
REM   5. Captum explainability + report (notebooks\5__Explainability_Captus.py)
REM
REM All outputs are written under  outputs\  exactly like the repo:
REM   outputs\plots\Analysis\                              (3 analysis plots)
REM   outputs\plots\Model_Explainability\GNN_Explainer\    (2 GNNExplainer plots)
REM   outputs\plots\Model_Explainability\Captus_Explainer\ (2 Captum plots)
REM   outputs\reports\Evaluate_Explaination.txt            (explanation report)
REM
REM The generated plots are opened at the end so they are shown to the user.
REM =============================================================================

REM ----------------------------------------------------------------------------
REM Resolve the repo root from this script's location (CWD-independent)
REM ----------------------------------------------------------------------------
set "ROOT=%~dp0"
REM Strip trailing backslash
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
cd /d "%ROOT%"

REM Python interpreter (has torch / torch-geometric / captum / rdflib installed)
REM Adjust this path to match your Windows Python install if needed
set "PY=python"

REM src\ holds the shared library modules imported by stages 3-5
set "PYTHONPATH=%ROOT%\src"
REM Force a non-interactive matplotlib backend so plots render to files headlessly
set "MPLBACKEND=Agg"

REM ----------------------------------------------------------------------------
REM Recreate the output folders (exact repo structure) even if they were deleted
REM ----------------------------------------------------------------------------
if not exist "%ROOT%\outputs\plots\Analysis" mkdir "%ROOT%\outputs\plots\Analysis"
if not exist "%ROOT%\outputs\plots\Model_Explainability\GNN_Explainer" mkdir "%ROOT%\outputs\plots\Model_Explainability\GNN_Explainer"
if not exist "%ROOT%\outputs\plots\Model_Explainability\Captus_Explainer" mkdir "%ROOT%\outputs\plots\Model_Explainability\Captus_Explainer"
if not exist "%ROOT%\outputs\reports" mkdir "%ROOT%\outputs\reports"

REM ----------------------------------------------------------------------------
REM Run the pipeline
REM ----------------------------------------------------------------------------
echo ==^> [1/5] Data Analysis
"%PY%" "%ROOT%\notebooks\1__analysis.py"
if errorlevel 1 goto :error

echo ==^> [2/5] Graph Pre-Processing
"%PY%" "%ROOT%\notebooks\2__Graph_PreProcessing.py"
if errorlevel 1 goto :error

echo ==^> [3/5] Model Training (R-GCN)
"%PY%" "%ROOT%\src\model_train.py"
if errorlevel 1 goto :error

echo ==^> [4/5] Explainability (GNNExplainer)
"%PY%" "%ROOT%\notebooks\4__Explainability.py"
if errorlevel 1 goto :error

echo ==^> [5/5] Explainability (Captum) + Report
"%PY%" "%ROOT%\notebooks\5__Explainability_Captus.py"
if errorlevel 1 goto :error

REM ----------------------------------------------------------------------------
REM Show the generated plots (analysis, GNN explainer, Captum explainer)
REM ----------------------------------------------------------------------------
echo.
echo ==^> Opening generated plots ...

for %%F in ("%ROOT%\outputs\plots\Analysis\*.png") do (
    if exist "%%F" start "" "%%F"
)
for %%F in ("%ROOT%\outputs\plots\Model_Explainability\GNN_Explainer\*.png") do (
    if exist "%%F" start "" "%%F"
)
for %%F in ("%ROOT%\outputs\plots\Model_Explainability\Captus_Explainer\*.png") do (
    if exist "%%F" start "" "%%F"
)

echo.
echo ============================================================
echo  Pipeline complete. Outputs are in: %ROOT%\outputs
echo    - Analysis plots        : outputs\plots\Analysis\
echo    - GNN explainer plots    : outputs\plots\Model_Explainability\GNN_Explainer\
echo    - Captum explainer plots : outputs\plots\Model_Explainability\Captus_Explainer\
echo    - Explanation report     : outputs\reports\Evaluate_Explaination.txt
echo ============================================================
goto :eof

:error
echo.
echo ============================================================
echo  Pipeline FAILED. See error output above.
echo ============================================================
exit /b 1