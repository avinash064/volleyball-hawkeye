# GitHub Setup Guide

## Quick Start

Follow these steps to push the Volleyball Hawk-Eye project to GitHub:

### 1. Create GitHub Repository

1. Go to [GitHub](https://github.com)
2. Click "New repository"
3. Repository name: `volleyball-hawkeye` (or your choice)
4. Description: "Production-ready volleyball tactical intelligence system using RT-DETR and ByteTrack"
5. Keep it **Public** or **Private**
6. **DO NOT** initialize with README (we already have one)
7. Click "Create repository"

### 2. Configure Git (First Time Only)

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### 3. Initial Commit

```bash
cd C:\Users\xghostrider\Downloads\NEw_ProJect\Volleyball

# Check status
git status

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Volleyball Hawk-Eye Tactical Intelligence System"
```

### 4. Connect to GitHub

Replace `yourusername` with your GitHub username:

```bash
git remote add origin https://github.com/yourusername/volleyball-hawkeye.git
git branch -M main
git push -u origin main
```

### 5. Verify Upload

Go to your GitHub repository URL:
```
https://github.com/yourusername/volleyball-hawkeye
```

You should see all files uploaded!

## What Gets Pushed

✅ **Included:**
- Source code (`.py` files)
- Documentation (`.md` files)
- Configuration files
- Scripts
- `.gitignore`
- `requirements.txt`
- Sample configs

❌ **Excluded (via .gitignore):**
- Model weights (`*.pt` files)
- Output videos (`*.mp4`, `*.avi`)
- Virtual environment (`venv/`)
- Cache files (`__pycache__/`)
- Large datasets

## Model Weights

Since model weights are large (>100MB), they're **NOT** pushed to GitHub.

Instead:
1. We provide the **Google Drive download link** in `WEIGHTS.md`
2. Users download weights separately
3. Link is always available in `README.md`

**Google Drive Link**: https://drive.google.com/file/d/1uASPCHAk6kDuVV8eZnTzEUx-zccD4-ik/view?usp=sharing

## Repository Structure on GitHub

```
volleyball-hawkeye/
├── README.md                         ← Main documentation
├── WEIGHTS.md                        ← Model download instructions
├── GITHUB_SETUP.md                   ← This file
├── requirements.txt                  ← Dependencies
├── .gitignore                        ← Ignore rules
├── hawkeye_complete.py               ← Main system
├── simple_player_tracker.py          ← Simplified tracker
├── batch_process.py                  ← Batch processing
├── volleyball_hawkeye.py             ← Original version
├── input_videos/                     ← Input folder
├── scripts/                          ← Utility scripts
├── src/                              ← Source modules
└── configs/                          ← Configuration files
```

## Updating Repository

After making changes:

```bash
# Check what changed
git status

# Add changed files
git add .

# Commit with message
git commit -m "Description of changes"

# Push to GitHub
git push
```

## Branching Strategy (Optional)

For collaborative development:

```bash
# Create feature branch
git checkout -b feature/new-feature

# Make changes...

# Commit
git add .
git commit -m "Add new feature"

# Push branch
git push -u origin feature/new-feature

# Create Pull Request on GitHub
```

## Troubleshooting

### Large Files Error

If you get "file too large" error:

```bash
# Find large files
git ls-files -s | awk '$4 > 50000000 {print $4}' 

# Remove from git (if accidentally added)
git rm --cached filename.pt
git commit -m "Remove large file"
```

Then add to `.gitignore`.

### Authentication Issues

If prompted for password:

1. **Use Personal Access Token** instead of password
2. Go to GitHub → Settings → Developer settings → Personal access tokens
3. Generate new token with `repo` scope
4. Use token as password

Or set up SSH:
```bash
ssh-keygen -t ed25519 -C "your.email@example.com"
cat ~/.ssh/id_ed25519.pub
# Add to GitHub → Settings → SSH keys
```

### Merge Conflicts

If you have conflicts:

```bash
git pull
# Resolve conflicts in files
git add .
git commit -m "Resolve merge conflicts"
git push
```

## GitHub Features to Enable

### 1. Releases

Create a release for v1.0:
1. Go to "Releases" on GitHub
2. Click "Create a new release"
3. Tag: `v1.0.0`
4. Title: "Volleyball Hawk-Eye v1.0"
5. Description: Release notes
6. Attach model weights (optional)

### 2. Issues

Enable GitHub Issues for bug reports and feature requests.

### 3. Wiki

Enable Wiki for extended documentation.

### 4. GitHub Actions (CI/CD)

Optional: Add `.github/workflows/test.yml` for automated testing.

## Sharing Your Project

Your repository URL will be:
```
https://github.com/yourusername/volleyball-hawkeye
```

Share with:
- Model weights link
- Installation instructions
- Demo videos (if available)

## License

Consider adding a LICENSE file (MIT, Apache 2.0, etc.)

```bash
# Add MIT License
curl https://raw.githubusercontent.com/licenses/license-templates/master/templates/mit.txt > LICENSE

git add LICENSE
git commit -m "Add MIT License"
git push
```

## Next Steps

1. ✅ Repository created on GitHub
2. ✅ Code pushed
3. ⬜ Add demo GIF/video to README
4. ⬜ Create GitHub Release
5. ⬜ Share project link

---

**Congratulations! Your project is now on GitHub! 🎉**
