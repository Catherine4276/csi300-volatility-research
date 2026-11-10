# Publishing and deployment

## 1. Create the public GitHub repository

Install the GitHub CLI, authenticate, and create a public repository:

```powershell
winget install --id GitHub.cli
gh auth login
gh repo create csi300-volatility-research --public --source . --remote origin --push
```

Alternative without `gh`:

1. Create an empty public repository on GitHub.
2. Add its URL as `origin`.
3. Push the `main` branch.

```powershell
git remote add origin https://github.com/YOUR_NAME/csi300-volatility-research.git
git push -u origin main
```

Raw provider CSV files are excluded by `.gitignore`. The download script and
data manifests preserve reproducibility without publicly redistributing them.

## 2. Deploy the dashboard with Streamlit Community Cloud

1. Sign in to Streamlit Community Cloud with GitHub.
2. Choose **New app**.
3. Select the public repository and `main` branch.
4. Set the main file path to `streamlit_app.py`.
5. Select Python 3.12.
6. Deploy.

The included `reports/` and `experiments/` artifacts let the dashboard render
immediately. The app does not need raw source data or secrets.

If the app is updated on GitHub, Streamlit Cloud can redeploy automatically.

## 3. Deploy with Docker

```powershell
docker build -t csi300-volatility-research .
docker run --rm -p 8501:8501 csi300-volatility-research
```

Open `http://localhost:8501`.

## 4. Refresh research artifacts

The deployed app displays committed research artifacts. To update forecasts:

```powershell
.\run_research.cmd
git add reports experiments data/raw/*_manifest.json data/processed/*_manifest.json
git commit -m "Update research artifacts"
git push
```

Verify the tests and research flow before publishing updates.

## Public-link checklist

- The repository is public and contains the MIT license.
- Raw provider data is not redistributed.
- No `.streamlit/secrets.toml`, token, password, or private file is committed.
- `streamlit_app.py` is the configured entry point.
- Automated tests pass in GitHub Actions.
- The public dashboard states that forecasts are not investment advice.

