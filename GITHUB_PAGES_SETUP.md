# GitHub Pages and Streamlit CI/CD Setup

## What GitHub Actions now does

This repository now uses two separate deployment/validation tracks:

### 1. GitHub Pages for the static site
Workflow:
- `.github/workflows/pages.yml`

What it does:
- installs Node dependencies
- builds the Vite site
- publishes the generated `docs/` output to GitHub Pages

This serves:
- homepage
- docs hub
- blog pages
- walkthrough mirror

Expected live URLs after Pages is enabled:
- `https://sgmoorthy.github.io/DERMA-Agent/`
- `https://sgmoorthy.github.io/DERMA-Agent/docs/index.html`
- `https://sgmoorthy.github.io/DERMA-Agent/blog/index.html`
- `https://sgmoorthy.github.io/DERMA-Agent/walkthrough.md`

### 2. Streamlit startup validation in CI
Workflow:
- `.github/workflows/streamlit-smoke.yml`

What it does:
- installs Python dependencies
- boots `derma_agent/app.py` in headless mode
- boots `app_enhanced.py` in headless mode
- fails CI if either app cannot start cleanly

This is the correct GitHub Actions role for Streamlit in this repository.

---

## Important limitation

GitHub Pages cannot host Streamlit.

Why:
- GitHub Pages serves static files only
- Streamlit requires a live Python process and runtime

So the right deployment split is:
- **GitHub Pages** → static site/docs/blog
- **Streamlit Community Cloud or another app host** → interactive dashboards

---

## How to enable GitHub Pages

1. Go to repository settings
2. Open **Pages**
3. Set **Source** to **GitHub Actions**
4. Push to `main`
5. Wait for the `Deploy to GitHub Pages` workflow to finish

---

## Recommended Streamlit deployment target

For the interactive dashboards, deploy one of these entrypoints to Streamlit Community Cloud:
- `derma_agent/app.py`
- `app_enhanced.py`

Secrets should be added in the relevant platform settings:
- **GitHub Actions**: `Settings → Secrets and variables → Actions`
- **Streamlit Community Cloud**: `App settings → Secrets`

Suggested secret names:
- `OPENAI_API_KEY`
- `GOOGLE_API_KEY`

Once deployed, add the returned Streamlit URL to the README live links section.

See also:
- `STREAMLIT_DEPLOYMENT.md`
- `.streamlit/secrets.toml.example`
- `README.md`
