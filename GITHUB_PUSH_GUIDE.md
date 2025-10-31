# Push to GitHub - Step by Step Guide

## Prerequisites
- Git installed
- GitHub account
- Repository already initialized locally (run `.\setup_github.ps1` first)

## Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Fill in:
   - **Repository name**: `NCKH-FITHANU`
   - **Description**: `AI-powered Speech Emotion Recognition for Online Classrooms`
   - **Visibility**: Public (or Private if needed)
3. **IMPORTANT**: Do NOT check:
   - ❌ Add a README file
   - ❌ Add .gitignore
   - ❌ Choose a license
4. Click "Create repository"

## Step 2: Link Local Repository to GitHub

```powershell
# Set main branch
git branch -M main

# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/NCKH-FITHANU.git

# Verify remote
git remote -v
```

## Step 3: Push to GitHub

```powershell
# Push to main branch
git push -u origin main
```

If you get authentication error, you may need to:

**Option A: Use Personal Access Token**
1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token with `repo` scope
3. Use token as password when pushing

**Option B: Use GitHub CLI**
```powershell
# Install GitHub CLI
winget install GitHub.cli

# Authenticate
gh auth login

# Push
git push -u origin main
```

## Step 4: Verify Upload

Visit: https://github.com/YOUR_USERNAME/NCKH-FITHANU

You should see:
- ✅ All files uploaded
- ✅ README displayed on homepage
- ✅ 26 files, ~2,274 lines of code

## Step 5: Configure Repository Settings (Optional)

### Add Topics
Settings → General → Topics:
- `speech-emotion-recognition`
- `ai`
- `machine-learning`
- `transformer`
- `wav2vec2`
- `fastapi`
- `python`
- `nlp`
- `audio-processing`

### Add Description
`AI-powered emotion recognition combining Wav2Vec2 and DistilRoBERTa for analyzing student emotions in online classroom environments`

### Configure Pages (if you want web demo hosted)
Settings → Pages → Source: `main` branch → `/web` folder

### Add Collaborators
Settings → Collaborators → Add people

## Common Issues

### Authentication Failed
```powershell
# Use credential manager
git config --global credential.helper manager

# Or use SSH instead
git remote set-url origin git@github.com:YOUR_USERNAME/NCKH-FITHANU.git
```

### Large Files (if you get warnings)
```powershell
# Check file sizes
git ls-files | xargs ls -lh

# If needed, use Git LFS for model files
git lfs install
git lfs track "*.bin"
```

### Already Exists
If repository name is taken, use alternative:
- `NCKH-FITHANU-SER`
- `SER-NCKH-2025`
- `speech-emotion-recognition-classroom`

## Next Steps After Upload

1. **Add badges** (copy from `GITHUB_README.md`)
2. **Create release**: Releases → Draft new release → v0.1.0
3. **Enable Issues**: Settings → General → Features → Issues
4. **Add Wiki**: Settings → General → Features → Wikis
5. **Protect main branch**: Settings → Branches → Add rule

## Updating Repository

After making changes:

```powershell
# Check status
git status

# Stage changes
git add .

# Commit
git commit -m "Description of changes"

# Push
git push
```

## Clone on Another Machine

```powershell
git clone https://github.com/YOUR_USERNAME/NCKH-FITHANU.git
cd NCKH-FITHANU
```

Then follow QUICKSTART.md to set up environment.

---

**Need help?** Check GitHub documentation: https://docs.github.com/
