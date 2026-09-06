# CI/CD Pipeline Documentation

**Aurik 10.0.0 Continuous Integration & Deployment**  
Letzte Aktualisierung: 16. Februar 2026

> **Normativer Hinweis (Release-Scope):** Aurik ist als Desktop-Produkt releasefaehig
> (Linux AppImage, Windows .exe). Historische CI-Hinweise zu Docker/Server gelten nur,
> wenn sie explizit nicht den Desktop-Releasepfad betreffen (`LEGACY_NON_RELEASE`).

---

## 📋 Overview

Aurik 10.0.0 nutzt GitHub Actions für Continuous Integration, Testing, Security Audits und automatisierte Releases.

### Available Workflows

| Workflow | Trigger | Purpose | Status |
| --- | --- | --- | --- |
| `ci_enhanced.yml` | Push/PR | Umfassende CI/CD Pipeline | ✅ Active |
| `ci.yml` | Push/PR | Basis CI mit Tests | ✅ Active |
| `release.yml` | Git Tags | Multi-Platform Build & Release | ✅ Active |
| `validate_musical_goals.yml` | Push/PR | Musical Quality Validation | ✅ Active |

---

## 🔄 CI/CD Pipeline (`ci_enhanced.yml`)

Der Hauptworkflow für kontinuierliche Integration mit mehreren Jobs.

### Jobs

#### 1. **Quality Gate** 🛡️

Führt Code-Quality-Checks durch, bevor Tests laufen.

```yaml
- Black formatting check (--line-length 120)
- Flake8 linting (complexity, style)
- Mypy type checking
- Bandit security scan
- Upload: security-reports artifact
```

**Zweck:** Stellt sicher, dass Code-Standards eingehalten werden und keine offensichtlichen Sicherheitslücken vorhanden sind.

#### 2. **Test Suite** 🧪

Führt vollständige Testsuite mit Coverage aus.

```yaml
- Install system dependencies (libsndfile1, ffmpeg, portaudio19-dev)
- Install Python dependencies
- Run pytest with coverage (--cov, -n auto)
- Upload: coverage-reports artifact
- Post coverage comments on PRs
```

**Features:**

- Parallele Testausführung (`-n auto`)
- Coverage-Reporting (XML, HTML, Terminal)
- Automatische Coverage-Comments auf Pull Requests

#### 3. **Dependency Audit** 🔒

Prüft Dependencies auf Sicherheitslücken.

```yaml
- Safety check (vulnerability scan)
- pip-audit (Python package audit)
- Upload: dependency-audit-reports artifact
```

**Zweck:** Erkennt bekannte Sicherheitslücken in Dependencies frühzeitig.

#### 4. **Docker Build** 🐳

Baut Docker Image und scannt auf Vulnerabilities.

```yaml
- Build mit Docker Buildx
- Trivy vulnerability scanner
- Cache management (type=gha)
- Upload: SARIF security results
```

**Trigger:** Nur bei Push-Events (nicht bei PRs)

#### 5. **Performance Benchmark** ⚡

Führt Performance-Benchmarks aus.

```yaml
- Run pytest with --benchmark-only
- Upload: benchmark_results.json artifact
```

**Trigger:** Nur bei Push-Events (nicht bei PRs)

#### 6. **CI Summary** 📊

Fasst alle Ergebnisse zusammen.

```yaml
- Check results of all previous jobs
- Report status
```

---

## 🚀 Release Pipeline (`release.yml`)

Automatisierte Multi-Platform Builds und Release-Erstellung.

### Trigger

1. **Git Tags:** Pushe einen Tag mit Format `v*.*.*` (z.B. `v10.0.0`)

   ```bash
   git tag v10.0.0
   git push origin v10.0.0
   ```

2. **Manual Dispatch:** Über GitHub Actions UI
   - Go to: Actions → Build and Release → Run workflow
   - Input: Version number (z.B., `9.0.1`)

### Jobs

#### 1. **Create Release** 📝

```yaml
- Generate release notes from commits
- Extract changelog from CHANGELOG.md
- Create GitHub Release
- Output: upload_url for artifact uploads
```

#### 2. **Build Windows** 🪟

```yaml
- Build with PyInstaller (aurik_90.spec)
- Create ZIP archive
- Upload to GitHub Release
- Asset: Aurik_90_Windows_x64.zip
```

#### 3. **Build Linux** 🐧

```yaml
- Install system dependencies
- Build with PyInstaller
- Create tar.gz archive
- Upload to GitHub Release
- Asset: Aurik_90_Linux_x64.tar.gz
```

#### 4. **Build macOS** 🍎

```yaml
- Install portaudio via brew
- Build with PyInstaller
- Create ZIP of .app bundle
- Upload to GitHub Release
- Asset: Aurik_90_macOS_x64.zip
```

#### 5. **Docker Release** 🐳

```yaml
- Build Docker image
- Tag with semver (latest, major.minor, major)
- Push to Docker Hub (optional)
```

**Note:** Requires `DOCKER_USERNAME` and `DOCKER_PASSWORD` secrets in repository settings.

---

## 🎯 Usage Guide

### For Developers

#### Running CI Locally

**Quality Gate:**

```bash
# Formatting check
black --check --line-length 120 .

# Linting
flake8 . --count --max-line-length=120

# Type checking
mypy . --config-file=mypy.ini

# Security scan
bandit -r . -ll -x './.venv_*,./build,./dist,./models'
```

**Test Suite:**

```bash
# Run all tests with coverage
pytest tests/ --cov=. --cov-report=term-missing -n auto

# Run only unit tests
pytest tests/unit/ --maxfail=1

# Run benchmarks
pytest benchmarks/ --benchmark-only
```

**Dependency Audit:**

```bash
# Safety check
pip install safety
safety check

# pip-audit
pip install pip-audit
pip-audit
```

#### Creating a Release

1. **Update CHANGELOG.md**

   ```markdown
   ## [9.0.1] - 2026-02-17
   ### Added
   - New feature X
   ### Fixed
   - Bug Y
   ```

2. **Commit and Tag**

   ```bash
   git add CHANGELOG.md
   git commit -m "Release 9.0.1"
   git tag v10.0.0
   git push origin main
   git push origin v10.0.0
   ```

3. **Monitor Release**
   - Go to: Actions → Build and Release
   - Wait for all builds to complete (~15-25 min)
   - Check: Releases page for artifacts

4. **Verify Assets**

   ```
   ✅ Aurik_90_Windows_x64.zip
   ✅ Aurik_90_Linux_x64.tar.gz
   ✅ Aurik_90_macOS_x64.zip
   ✅ Release notes generated
   ```

### For CI/CD Maintainers

#### Workflow Configuration

**Environment Variables:**

```yaml
env:
  PYTHON_VERSION: "3.10"
  APP_NAME: "Aurik"
  APP_VERSION: "9.0"
```

**Secrets Required:**

- `GITHUB_TOKEN` (automatic, no setup needed)
- `DOCKER_USERNAME` (optional, for Docker Hub push)
- `DOCKER_PASSWORD` (optional, for Docker Hub push)

#### Caching Strategy

**pip cache:**

```yaml
- uses: actions/setup-python@v5
  with:
    python-version: ${{ env.PYTHON_VERSION }}
    cache: 'pip'
```

**Docker cache:**

```yaml
- uses: docker/build-push-action@v5
  with:
    cache-from: type=gha
    cache-to: type=gha,mode=max
```

#### Artifact Management

**Artifacts Retention:**

- Coverage reports: 30 days
- Benchmark results: 30 days
- Security reports: 90 days
- Release assets: Permanent (attached to release)

---

## 📊 Monitoring & Reporting

### GitHub Actions UI

**View Workflow Runs:**

- Repository → Actions tab
- Filter by workflow, branch, status
- Click run for detailed logs

**Artifacts:**

- Run details → Artifacts section
- Download: coverage-reports, benchmark-results, security-reports

### Status Badges

**In README.md:**

```markdown
[![CI/CD Pipeline](https://github.com/YOUR_USERNAME/Aurik_Standalone/actions/workflows/ci_enhanced.yml/badge.svg)](...)
[![Release Build](https://github.com/YOUR_USERNAME/Aurik_Standalone/actions/workflows/release.yml/badge.svg)](...)
```

**Badge Colors:**

- 🟢 Green: All checks passed
- 🔴 Red: Failures detected
- 🟡 Yellow: Running or pending

### Notifications

**Default Notifications:**

- Email on workflow failures (repository contributors)
- In-app GitHub notifications

**Custom Notifications:**

- Extend workflows with Slack/Discord webhooks
- Add notification jobs to workflows

---

## 🔧 Troubleshooting

### Common Issues

#### 1. **Test Failures**

```
Problem: Tests fail in CI but pass locally
Solution:
- Check Python version matches (3.10)
- Verify system dependencies installed
- Review environment variables
```

#### 2. **PyInstaller Build Errors**

```
Problem: PyInstaller fails to create executable
Solution:
- Update aurik_90.spec with missing imports
- Check for dynamic imports (use hiddenimports)
- Verify all resources included (datas=[...])
```

#### 3. **Cache Issues**

```
Problem: Stale cache causing inconsistent builds
Solution:
- Manually clear cache: Actions → Caches → Delete
- Bump CACHE_VERSION in workflow
```

#### 4. **Release Upload Failures**

```
Problem: Assets fail to upload to release
Solution:
- Verify tag format correctness (v*.*.*)
- Check upload_url propagation from create-release job
- Review GITHUB_TOKEN permissions (should be automatic)
```

### Debug Workflows

**Enable debug logging:**

```bash
# In repository → Settings → Secrets and variables → Actions
# Add repository variable:
ACTIONS_STEP_DEBUG = true
ACTIONS_RUNNER_DEBUG = true
```

**Re-run failed jobs:**

- Workflow run → Re-run failed jobs button
- Or: Re-run all jobs (if needed)

---

## 🚦 Best Practices

### Commit Messages

```
✅ Good: "Fix crash when processing 32-bit audio files"
✅ Good: "Add support for FLAC batch processing"
❌ Bad: "fix bug"
❌ Bad: "updates"
```

### Pull Request Workflow

1. Create feature branch
2. Make changes with tests
3. Push and create PR
4. CI runs automatically
5. Review coverage report comment
6. Merge when all checks pass

### Release Versioning

```
Major.Minor.Patch (Semantic Versioning)
10.0.20 → Patch release (bug fixes)
10.1.0 → Minor release (new features)
11.0.0 → Major release (breaking changes)
```

### Pre-Release Testing

```bash
# Before creating release tag
./run_all_tests.sh
pytest tests/ --maxfail=1 --disable-warnings
pytest benchmarks/ --benchmark-only
```

---

## 📈 Performance Metrics

### CI Pipeline Speed

| Job                   | Typical Duration | Timeout    |
|-----------------------|------------------|------------|
| Quality Gate          | 2-3 min          | 10 min     |
| Test Suite            | 5-8 min          | 30 min     |
| Dependency Audit      | 1-2 min          | 10 min     |
| Docker Build          | 3-5 min          | 20 min     |
| Performance Benchmark | 2-4 min          | 15 min     |
| **Total Pipeline**    | **~15-20 min**   | **60 min** |

### Release Build Speed

| Platform  | Build Time     | Size        |
|-----------|----------------|-------------|
| Windows   | 8-12 min       | ~200 MB     |
| Linux     | 6-10 min       | ~180 MB     |
| macOS     | 10-15 min      | ~220 MB     |
| **Total** | **~25-35 min** | **~600 MB** |

---

## 🔐 Security

### Secrets Management

- Never commit secrets to repository
- Use GitHub Secrets for sensitive data
- Rotate Docker credentials regularly

### Dependency Security

- Automated scanning via Safety and pip-audit
- Review security reports in Artifacts
- Update vulnerable dependencies promptly

### Docker Security

- Trivy scans for vulnerabilities
- SARIF results uploaded to Security tab
- Base image updates in Dockerfile

---

## 🎓 Further Reading

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [PyInstaller Manual](https://pyinstaller.org/en/stable/)
- [Semantic Versioning](https://semver.org/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

---

## 📞 Support

**Issues:** GitHub Issues  
**Discussions:** GitHub Discussions  
**Maintainers:** See CONTRIBUTING.md

---

**Last Updated:** 16. Februar 2026  
**Version:** 1.0  
**Status:** ✅ Production Ready
