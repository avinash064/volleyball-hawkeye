@echo off
echo ============================================================
echo   GitHub Repository Setup for Volleyball Hawk-Eye
echo ============================================================
echo.

REM Check if git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git is not installed!
    echo Please install Git from https://git-scm.com/download/win
    pause
    exit /b 1
)

echo [Step 1] Setting up git repository...
echo.

REM Configure git (update with your details)
echo Please enter your GitHub username:
set /p GITHUB_USER=Username: 

echo.
echo Please enter your GitHub email:
set /p GITHUB_EMAIL=Email: 

echo.
echo Please enter repository name (default: volleyball-hawkeye):
set /p REPO_NAME=Repository name: 
if "%REPO_NAME%"=="" set REPO_NAME=volleyball-hawkeye

echo.
echo Configuring git...
git config user.name "%GITHUB_USER%"
git config user.email "%GITHUB_EMAIL%"

echo.
echo [Step 2] Creating initial commit...
git add .
git commit -m "Initial commit: Volleyball Hawk-Eye Tactical Intelligence System"

echo.
echo ============================================================
echo   IMPORTANT: Create GitHub Repository
echo ============================================================
echo.
echo 1. Open your browser and go to:
echo    https://github.com/new
echo.
echo 2. Repository name: %REPO_NAME%
echo.
echo 3. Description: Production-ready volleyball tactical intelligence system
echo.
echo 4. Keep it Public or Private (your choice)
echo.
echo 5. DO NOT initialize with README (we have one)
echo.
echo 6. Click "Create repository"
echo.
echo Press any key after you've created the repository on GitHub...
pause >nul

echo.
echo [Step 3] Connecting to GitHub...
echo.
set REPO_URL=https://github.com/%GITHUB_USER%/%REPO_NAME%.git
echo Repository URL: %REPO_URL%
echo.

git branch -M main
git remote add origin %REPO_URL%

echo.
echo [Step 4] Pushing to GitHub...
echo.
echo This will open a browser window for authentication.
echo Sign in with your GitHub account.
echo.
pause

git push -u origin main

echo.
echo ============================================================
echo   SUCCESS!
echo ============================================================
echo.
echo Your repository is now on GitHub:
echo https://github.com/%GITHUB_USER%/%REPO_NAME%
echo.
echo Model weights link is in WEIGHTS.md
echo.
pause
