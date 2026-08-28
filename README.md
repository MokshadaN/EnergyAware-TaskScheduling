# Energy-Aware Cloud Task Scheduling

An interactive portfolio project for comparing cloud task-placement policies across energy consumption, latency, deadline success, and virtual-machine utilization.

## What you can explore

- Adjustable workload and cluster size
- Energy-aware, least-utilized, and round-robin scheduling policies
- Cumulative energy and VM-utilization visualizations
- A self-contained browser simulation with no API keys or backend

The original Python reinforcement-learning code remains in the repository for experimentation. The public dashboard uses a transparent, deterministic heuristic so visitors can immediately compare scheduling trade-offs.

## Run locally

Open `index.html` in a modern browser, or serve the repository with any static-file server.

## Deploy to Vercel

1. Import this GitHub repository in [Vercel](https://vercel.com/new).
2. Select the `main` branch after merging the pull request.
3. Vercel automatically detects and serves the static site—leave the build command and output directory blank.
4. Click **Deploy**.

`vercel.json` adds clean URLs and a small set of standard security headers.

## Project structure

```text
index.html              # Vercel portfolio site
assets/                 # Browser simulation and styles
vercel.json             # Static-site deployment configuration
core/                   # Cloud-environment simulation
agents/                 # DQN scheduler implementation
main.py                 # Training and benchmark workflow
```

## Notes

This project uses synthetic workload data. Its visual metrics are for simulated policy comparisons and are not measurements from a production data center.

## License

Apache-2.0
