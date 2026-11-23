# Releasing to PyPI

The package is published as `hmm-regime` (`pip install hmm-regime`, `import hmm_regime`).
`.github/workflows/publish.yml` builds it, tests the built wheel and uploads it.
It uses PyPI trusted publishing, so no API token is stored anywhere.

## One-time setup

1. **Accounts.** Create an account on [pypi.org](https://pypi.org) and one on [test.pypi.org](https://test.pypi.org) (separate sites, separate accounts). Turn on two-factor authentication; PyPI requires it.
2. **Pending publisher on PyPI.** pypi.org → Account settings → Publishing → *Add a new pending publisher* → GitHub:
   - PyPI project name: `hmm-regime`
   - Owner: `mehmetoaltekin`
   - Repository name: `hmm-regime-detection`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`

   A pending publisher reserves the name for this workflow; the project is created by the first upload.
3. **The same on TestPyPI**, with environment name `testpypi`.
4. **GitHub environments.** In the repository: Settings → Environments → *New environment*, create `pypi` and `testpypi`. Optional but recommended for `pypi`: *Required reviewers* = yourself, so every upload waits for your click.

## Every release

1. Set the version in `hmm_regime/__init__.py` (`__version__ = "X.Y.Z"`) and move the `Unreleased` entries of `CHANGELOG.md` under `## X.Y.Z`. Merge to `main`; CI must be green.
2. Rehearsal (optional): Actions → *Publish to PyPI* → *Run workflow*. This uploads to TestPyPI. Check it with
   `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ hmm-regime`.
   TestPyPI accepts each version only once as well.
3. Tag and push:
   ```bash
   git checkout main && git pull
   git tag -a vX.Y.Z -m "hmm-regime X.Y.Z"
   git push origin vX.Y.Z
   ```
   The workflow refuses a tag that does not match `__version__`.
4. A version number can never be uploaded twice, even after deleting it on PyPI. A mistake means a new version (e.g. 0.2.1).
