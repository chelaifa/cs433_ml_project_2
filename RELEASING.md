# Release Process

This document describes how to create a new release of the RAG Pipeline.

## Overview

Releases are automated via GitHub Actions. When you push a version tag, the workflow:

1. ✅ Builds API PEX executable (portable, ~100MB)
2. ✅ Fetches latest research report PDF from [project-2-rag-report](https://github.com/eliemada/project-2-rag-report)
3. ✅ Creates GitHub release with changelog
4. ✅ Uploads release assets (pex, pdf, checksums)
5. ✅ Publishes Docker images to GitHub Container Registry

## Creating a Release

### 1. Update Version Numbers

Update version in all `pyproject.toml` files:

```bash
# Update versions to match (e.g., 1.2.0)
vim pyproject.toml                          # Root workspace
vim packages/shared/pyproject.toml          # rag_pipeline
vim packages/api/pyproject.toml             # API
vim packages/worker/pyproject.toml          # Worker
```

### 2. Update CHANGELOG

Add release notes to `CHANGELOG.md` (create if doesn't exist):

```markdown
## [1.2.0] - 2025-01-15

### Added
- New feature X
- Support for Y

### Changed
- Improved performance of Z

### Fixed
- Bug in retrieval pipeline
```

### 3. Commit and Tag

```bash
# Commit version updates
git add pyproject.toml packages/*/pyproject.toml CHANGELOG.md
git commit -m "chore: Bump version to 1.2.0"

# Create and push tag
git tag -a v1.2.0 -m "Release v1.2.0"
git push origin main
git push origin v1.2.0
```

### 4. Wait for CI

The release workflow will automatically:
- Build artifacts (~5 minutes)
- Create GitHub release (~2 minutes)
- Publish Docker images (~15 minutes)

Monitor progress: `https://github.com/YOUR_ORG/project-2-rag/actions`

### 5. Verify Release

Check the release page: `https://github.com/YOUR_ORG/project-2-rag/releases`

Verify assets are present:
- ✅ `api-linux-amd64.pex` (~100MB)
- ✅ `research-report.pdf` (from report repo)
- ✅ `SHA256SUMS` (checksums for verification)

### 6. Test Release

Download and test the PEX:

```bash
# Download
wget https://github.com/YOUR_ORG/project-2-rag/releases/download/v1.2.0/api-linux-amd64.pex

# Verify checksum
wget https://github.com/YOUR_ORG/project-2-rag/releases/download/v1.2.0/SHA256SUMS
sha256sum -c SHA256SUMS

# Test run
chmod +x api-linux-amd64.pex
./api-linux-amd64.pex api.main:app --host 0.0.0.0 --port 8000
```

Test Docker image:

```bash
docker pull ghcr.io/YOUR_ORG/project-2-rag/rag-api:v1.2.0
docker run -p 8000:8000 ghcr.io/YOUR_ORG/project-2-rag/rag-api:v1.2.0
```

## Manual Release (If Needed)

If you need to create a release manually:

```bash
# Trigger workflow manually via GitHub UI
# Go to: Actions → Release → Run workflow
# Enter tag: v1.2.0
```

Or use GitHub CLI:

```bash
gh workflow run release.yml -f tag=v1.2.0
```

## Release Assets

### PEX Executable

- **File**: `api-linux-amd64.pex`
- **Size**: ~100MB (no GPU dependencies)
- **Platform**: Linux x86_64 (manylinux_2_28)
- **Python**: 3.11+ required
- **Use case**: Quick deployments, testing, lightweight servers

**Why no worker.pex?**
Worker package includes GPU dependencies (PyTorch, Transformers) which would result in a 4-5GB pex file. For worker deployment, use Docker instead:

```bash
docker pull ghcr.io/YOUR_ORG/project-2-rag/pdf-worker:v1.2.0
```

### Research Report PDF

Automatically fetched from the latest release of [project-2-rag-report](https://github.com/eliemada/project-2-rag-report/releases).

If the report repo hasn't been updated, the workflow will attempt to download `main.pdf` from the main branch as a fallback.

### Docker Images

Published to GitHub Container Registry:

- `ghcr.io/YOUR_ORG/project-2-rag/rag-api:v1.2.0` (800MB)
- `ghcr.io/YOUR_ORG/project-2-rag/pdf-worker:v1.2.0` (4.5GB)

Tags created:
- `v1.2.0` (exact version)
- `1.2` (minor version)
- `1` (major version)
- `latest` (latest release)

## Semantic Versioning

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (v1.0.0 → v2.0.0): Breaking changes
- **MINOR** (v1.0.0 → v1.1.0): New features, backward compatible
- **PATCH** (v1.0.0 → v1.0.1): Bug fixes, backward compatible

Examples:
- `v1.0.0` → `v1.0.1`: Fixed bug in retrieval
- `v1.0.0` → `v1.1.0`: Added new LLM model support
- `v1.0.0` → `v2.0.0`: Changed API schema (breaking)

## Troubleshooting

### Build Fails

If PEX build fails:

```bash
# Test build locally
./scripts/ci/build_pex.sh api x86_64-manylinux_2_28

# Check dependencies
uv sync --all-packages
uv pip list
```

### PDF Not Found

If research report PDF isn't found:

1. Check [report repo releases](https://github.com/eliemada/project-2-rag-report/releases)
2. Ensure latest release has a PDF asset
3. Or manually upload PDF to this release

### Docker Push Fails

If Docker image push fails:

1. Check GitHub Container Registry permissions
2. Ensure `GITHUB_TOKEN` has `packages:write` permission
3. Verify Docker Buildx setup

### Release Already Exists

If tag already exists:

```bash
# Delete tag locally and remotely
git tag -d v1.2.0
git push origin :refs/tags/v1.2.0

# Recreate tag
git tag -a v1.2.0 -m "Release v1.2.0"
git push origin v1.2.0
```

## Post-Release

After creating a release:

1. ✅ Announce in relevant channels (Slack, email, etc.)
2. ✅ Update deployment documentation if needed
3. ✅ Monitor issue tracker for bug reports
4. ✅ Update deployment servers to new version

## Example Release Workflow

Complete example:

```bash
# 1. Update versions
vim pyproject.toml packages/*/pyproject.toml
# Change version = "0.1.0" to version = "1.0.0"

# 2. Update changelog
vim CHANGELOG.md
# Add release notes

# 3. Commit and tag
git add .
git commit -m "chore: Release v1.0.0"
git tag -a v1.0.0 -m "Release v1.0.0 - Initial production release"
git push origin main
git push origin v1.0.0

# 4. Wait for workflow (check Actions tab)
# 5. Verify release: https://github.com/YOUR_ORG/project-2-rag/releases/tag/v1.0.0
# 6. Test deployment
# 7. Announce release
```

Done! 🎉
