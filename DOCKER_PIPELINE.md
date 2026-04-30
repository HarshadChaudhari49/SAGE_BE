# Docker Pipeline

This project does not require building a Docker image on your local machine.
The deployment pipeline is:

```text
GitHub push -> GitHub Actions -> Docker image -> GHCR -> Deployment platform
```

## Stage 1: Push Code to GitHub

```powershell
git add Dockerfile .dockerignore .github/workflows/docker-image.yml simulation_server.py DOCKER_PIPELINE.md
git commit -m "Add GitHub Docker deployment pipeline"
git push origin main
```

## Stage 2: GitHub Validates the Server

The workflow imports `simulation_server.py` using Python 3.11:

```text
python -B -c "import simulation_server"
```

This catches syntax/import errors before building the image.

## Stage 3: GitHub Builds the Docker Image

GitHub Actions builds the image from `Dockerfile`.

The Dockerfile runs the dashboard server:

```text
CMD ["python", "simulation_server.py"]
```

The container exposes:

```text
PORT=7860
EXPOSE 7860
```

## Stage 4: GitHub Pushes Image to GHCR

The image is pushed to GitHub Container Registry:

```text
ghcr.io/harshadchaudhari49/sage_be:latest
```

It also creates a commit-specific image tag:

```text
ghcr.io/harshadchaudhari49/sage_be:<commit-sha>
```

## Stage 5: Deployment Trigger

The workflow has a deployment stage.

To enable it, create this GitHub repository secret:

```text
DEPLOY_HOOK_URL
```

Put your hosting provider's deploy hook URL in that secret.

Common flow:

1. Create a web service on Render, Koyeb, Railway, or another Docker host.
2. Set the service image to:

```text
ghcr.io/harshadchaudhari49/sage_be:latest
```

3. Add the provider's deploy hook URL as `DEPLOY_HOOK_URL`.
4. Push to `main`.

If `DEPLOY_HOOK_URL` is not configured, the workflow still builds and pushes the Docker image, but skips deployment.

## Stage 6: Hugging Face Spaces Deployment

The workflow can also deploy the project source to a Hugging Face Docker Space.

Create a Hugging Face Space with:

```text
SDK: Docker
Visibility: Public
```

Then create these GitHub repository secrets:

```text
HF_TOKEN
HF_SPACE
```

`HF_TOKEN` is your Hugging Face access token.

`HF_SPACE` should be:

```text
your-huggingface-username/your-space-name
```

Example:

```text
harshadchaudhari49/sage-be
```

After every push to `main`, GitHub Actions pushes the repository to the Hugging Face Space. Hugging Face then builds the Dockerfile and runs:

```text
python simulation_server.py
```

The Space uses the README metadata:

```yaml
sdk: docker
app_port: 7860
```

## Server Requirements Inside Docker

The Docker container must have:

- Python 3.11
- dependencies from `requirements.txt`
- `simulation_server.py`
- `templates/dashboard.html`
- saved model files under `models/`

The Flask app must bind to:

```text
0.0.0.0
```

and use the platform port:

```python
port = int(os.environ.get("PORT", 5000))
```
