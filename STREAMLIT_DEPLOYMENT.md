# Streamlit Deployment Guide

## Important distinction

GitHub Actions can:
- run tests,
- run Streamlit smoke checks,
- build and deploy the static Vite/docs site to GitHub Pages.

GitHub Actions **cannot by itself host a long-running Streamlit server** on GitHub Pages.

GitHub Pages is for static content only. A real Streamlit deployment requires a runtime host.

---

## Recommended deployment split

### Static site / docs
Use GitHub Pages for:
- landing page
- docs hub
- blog pages
- walkthrough mirror

This repo now does that via:
- `.github/workflows/pages.yml`

### Interactive dashboards
Use a runtime host for:
- `derma_agent/app.py`
- `app_enhanced.py`

Recommended options:
- **Streamlit Community Cloud**
- Render
- Railway
- Azure Web App / Container App
- any Docker-capable host

---

## What CI/CD now covers

### 1. Static-site deployment
`pages.yml` now builds the Vite app and deploys the generated `docs/` directory to GitHub Pages.

### 2. Streamlit smoke testing
`streamlit-smoke.yml` verifies that both Streamlit entrypoints can boot in headless mode in CI.

This catches:
- broken imports
- startup-time syntax errors
- missing dependency wiring
- obvious boot regressions

It does **not** publish a permanent Streamlit server.

---

## Recommended live setup

### Option A — GitHub Pages + Streamlit Community Cloud
This is the simplest setup.

- GitHub Pages:
  - static docs and blog
  - URL pattern: `https://sgmoorthy.github.io/DERMA-Agent/`
- Streamlit Community Cloud:
  - interactive app runtime
  - URL pattern: `https://<your-app-name>.streamlit.app`

### Suggested app entrypoint
If you want a single public Streamlit URL, deploy one of:
- `derma_agent/app.py` for the main scientific dashboard
- `app_enhanced.py` for the enhanced multi-tab dashboard

---

## Deploying to Streamlit Community Cloud

1. Push this repository to GitHub
2. Open https://share.streamlit.io/
3. Create a new app
4. Select the repository and branch
5. Set the main file path to one of:
   - `derma_agent/app.py`
   - `app_enhanced.py`
6. Add required secrets such as:
   - `OPENAI_API_KEY` for OpenAI-backed discovery generation
   - `GOOGLE_API_KEY` for the Gemini-backed research assistant
7. Deploy

You can paste them directly into **App settings → Secrets** in Streamlit Community Cloud using the same shape as `.streamlit/secrets.toml.example`.

Once the app is deployed, add the returned URL to `README.md` under the live links section.

---

## GitHub repository settings for CI

If you want CI runs to boot with real secrets available, add them here:

1. Open the repository on GitHub
2. Go to **Settings → Secrets and variables → Actions**
3. Add repository secrets:
   - `OPENAI_API_KEY`
   - `GOOGLE_API_KEY`

The workflow `.github/workflows/streamlit-smoke.yml` is already wired to consume those secrets if present.

---

## Recommended README live links

- GitHub Pages home: `https://sgmoorthy.github.io/DERMA-Agent/`
- Docs hub: `https://sgmoorthy.github.io/DERMA-Agent/docs/index.html`
- Blog index: `https://sgmoorthy.github.io/DERMA-Agent/blog/index.html`
- Streamlit app: add after deployment on Streamlit Community Cloud

---

## Why this split is correct

This mirrors the technical reality of the stack:
- Vite output is static and ideal for Pages
- Streamlit is a Python web application that needs a live process

Using the right host for each part keeps deployment simpler and avoids pretending that GitHub Pages can run Streamlit.
