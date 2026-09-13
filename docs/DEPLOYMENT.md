# Deploy the playlist demo

## Current public demo: Vercel + Docker on this laptop

The frontend is deployed at **https://playlist-lab-ruddy.vercel.app** in the Vercel Hobby project `xharra/playlist-lab`. Share this address, not localhost or the temporary backend address. Deployment is from the local `web/` folder using Vercel CLI; GitHub is not connected and a Git commit is not required to update this deployment.

Vercel's server-side `RECOMMENDER_API_URL` points to a Cloudflare Quick Tunnel into the Docker API. The three trained models and catalog files stay in the local `models/` folder, mounted read-only. Only the API's health, search and recommendation endpoints are served by that backend; model files are not downloadable through it. The Vercel upload contains the frontend source, not the model bundle.

**Availability:** this is a demonstration setup. The laptop must stay awake and connected to the internet, with Docker and the API/tunnel containers running. Vercel continues to serve the page if the laptop is off, but search and recommendations are unavailable. A Quick Tunnel has no uptime guarantee and receives a new URL when restarted ([Cloudflare documentation](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/)). Do not change sleep settings without considering this dependency.

### Reconnect after restarting Windows or Docker

Open Docker Desktop and wait for the engine to run. From the project folder, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_vercel_demo.ps1
```

The script starts the API, local web server and tunnel, reads the current tunnel URL, checks all three models, updates the Vercel production environment variable, and deploys the current frontend source. It reuses the existing Vercel project, so the public frontend address stays the same. It requires this laptop's project-local Node/Vercel tools and the Vercel CLI login established during setup. If sign-in expires, run `.tools/node-v22.23.2-win-x64/node.exe .tools/vercel-cli/node_modules/vercel/dist/vc.js login` from the project folder and approve Vercel's device login, then rerun the script.

For a read-only API connection check without deploying:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_vercel_demo.ps1 -CheckOnly
```

To stop public access to the laptop API while leaving the local app running:

```powershell
docker compose -f compose.yaml -f compose.tunnel.yaml stop tunnel
```

### Deployment changes and checks (13 September 2026)

- Built and ran the Docker web/API images; all three models passed local recommendation requests.
- Installed Vercel CLI into the ignored `.tools/vercel-cli/` folder and completed the user-approved device login.
- Created `xharra/playlist-lab`, set the production backend URL, and deployed only `web/`.
- Added `compose.tunnel.yaml` and the reconnection script. No new inbound laptop firewall port is required.
- Added frontend upload exclusions, bounded API request timeouts, a 120-second recommendation function duration, and service-offline messages suitable for the public site.
- Made Next.js standalone output conditional: enabled for Docker, disabled for Vercel. This resolved the observed Next.js 16.3/Vercel adapter build failure ([upstream issue](https://github.com/vercel/next.js/issues/96646)).
- The local production build and ESLint passed. The corrected Vercel production build completed successfully. Unauthenticated requests to the public homepage, health and search succeeded; LightGCN, SIGN and Residual SIGN each returned three recommendations through the Vercel URL. The restart script's `-CheckOnly` path also passed against the running tunnel.

If you connect GitHub later, set Vercel's **Root Directory to `web`** before enabling automatic deployments. Commit/push the source and deployment files; keep the ignored model files, `.env*`, `.vercel/`, `.tools/` and generated transfer bundle out of Git. GitHub connection alone does not host the models or remove the laptop dependency.

For an always-online service independent of this laptop, move the Docker API/models to a server and change `RECOMMENDER_API_URL` to its HTTPS endpoint, then redeploy Vercel. The existing free Vercel frontend address can be kept. The following alternative hosts both services together on a VPS and uses a separately owned domain.

## Alternative: host everything on a Linux VPS

This setup runs the Next.js interface and Python recommender as separate containers. The three trained checkpoints and shared catalogs remain in `models/`; Compose mounts that folder read-only into the API. The public option adds Caddy, which serves one HTTPS address and keeps the API private. GitHub is not required.

## Current local test status (13 September 2026)

WSL 2 and Docker Desktop are installed and working. After the user freed disk space (about 69 GB free on C: at the last pre-build check), both production images built successfully. `docker compose up -d` started the API and web containers; the API passed Docker's health check. `http://localhost:3001/api/health` returned `ready` and listed LightGCN, SIGN and Residual SIGN. Search returned tracks, and a recommendation request succeeded for each of the three models. The local demo was left running. The private `deployment-bundle/` transfer folder was also created successfully. No Docker data cleanup is needed.

## What has been prepared

- `web/Dockerfile` builds the production Next.js server with Node 22.
- `docker/api.Dockerfile` installs only the Python inference dependencies on CPU.
- `compose.yaml` starts both services and checks that all three models are ready. The local demo uses `http://localhost:3001` so it does not conflict with the existing port-3000 development server.
- `compose.public.yaml` and `docker/Caddyfile` add the HTTPS entry point for a server with a domain name.
- `docker/check_models.py` checks the required ignored model files before the API starts.
- `scripts/prepare_deployment_bundle.ps1` copies only deployment source and the required private model artifacts into a transfer folder; it excludes raw datasets, notebooks and training files.

## 1. Start Docker Desktop and test locally (optional)

WSL 2 and Docker Desktop are already installed. Open Docker Desktop and wait until it says **Running**. Reopen the VS Code terminal if the `docker` command is not found. On this laptop the CLI is also at `C:\Users\artix\AppData\Local\Programs\DockerDesktop\resources\bin\docker.exe`.

Free at least **10 GB on C:** before building the images; PyTorch and Docker layers need room. Do not delete the project's `models` folder. If local disk space is inconvenient, skip this optional test and build on the Linux VPS in step 4. The VPS does not use this laptop's WSL installation.

From the repository root in a PowerShell terminal:

```powershell
docker version
python docker/check_models.py
docker compose up --build -d
docker compose ps
```

If `python` is not on PATH, use the environment recorded in `.tools/environment.json` or `C:\Users\artix\.venvs\music\Scripts\python.exe` for the artifact check. The first image build can take a while. When both services are healthy, open `http://localhost:3001`. Check `http://localhost:3001/api/health`: it should say `ready` and list LightGCN, SIGN and Residual SIGN. Then try search and one recommendation with each model in the page.

Logs and stop commands:

```powershell
docker compose logs --tail=100 api
docker compose logs --tail=100 web
docker compose down
```

`docker compose down` stops the demo; it does not remove the original model files.

## 2. Get a public server and domain

Choose a Linux VPS with approximately **8 GB RAM and at least 30 GB storage** for this catalog and the large checkpoints. Install Docker Engine with the Compose plugin using the provider's instructions or the [official Docker guide](https://docs.docker.com/engine/install/). Set a domain or subdomain A record to the VPS public IP. Allow inbound TCP ports **80** and **443** in its firewall. Caddy can then obtain and renew HTTPS certificates when DNS and these ports are ready ([Caddy requirements](https://caddyserver.com/docs/automatic-https)).

These are the account, payment, DNS and server-creation steps you must do manually. Do not deploy from this laptop as a public server; the VPS should stay online independently.

## 3. Transfer the application without GitHub

The transfer folder has already been created at `deployment-bundle/`. If you need to regenerate it later, move or rename the existing folder first; the script deliberately refuses to overwrite an existing destination. To create it again:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/prepare_deployment_bundle.ps1
```

This creates `deployment-bundle` in the project. You can instead pass `-Destination 'D:\music-demo-upload'` if another drive is available and C: is tight. The destination must not already exist. The bundle folder is ignored by Git and Docker builds. It contains about 1.5 GB of model/catalog artifacts, so transfer it privately with an SFTP client such as WinSCP into a folder such as `~/music-demo` on the VPS. The original source files remain untouched. Do not put the models in a public repository.

## 4. Start the public demo

On the VPS, change into the uploaded `music-demo` folder. Create a file named `.env` beside `compose.yaml` containing your real domain, for example:

```dotenv
DEMO_DOMAIN=music.example.com
```

Then run:

```bash
docker compose -f compose.yaml -f compose.public.yaml config
docker compose -f compose.yaml -f compose.public.yaml up --build -d
docker compose -f compose.yaml -f compose.public.yaml ps
```

Open `https://music.example.com/api/health` and confirm all three models are listed. Open the main address, search for a track and test recommendations under each model. You can now share `https://music.example.com` with your professor and others. The Python API is reachable only through the web application; port 8000 is not published.

If the page is unavailable, check `docker compose -f compose.yaml -f compose.public.yaml logs --tail=100` on the VPS. For DNS or certificate trouble, confirm the A record points to the VPS and ports 80/443 are open. To update the demo, transfer changed source files, then rerun the public `up --build -d` command. Transfer changed model files too only if you retrain or regenerate them.
