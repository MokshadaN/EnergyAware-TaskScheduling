# Energy-Aware Cloud Task Scheduling

An interactive cloud-scheduling simulator that compares task-placement policies across energy consumption, latency, deadline success, and VM utilization.

## Live dashboard

Deploy `app.py` on Streamlit Community Cloud to publish the interactive demo.

## What it demonstrates

- Interactive workload and cluster controls
- Energy-aware, least-utilized, and round-robin scheduling policies
- Per-VM energy and utilization charts
- Reproducible scenarios using a fixed random seed
- CSV export for each simulation run

The original reinforcement-learning training code remains in the repository for experimentation. The dashboard is intentionally lightweight and uses a transparent heuristic so visitors can explore the scheduling trade-offs without a model checkpoint or lengthy training job.

## Run locally

```bash
git clone https://github.com/MokshadaN/EnergyAware-TaskScheduling.git
cd EnergyAware-TaskScheduling
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

1. Sign in at [Streamlit Community Cloud](https://share.streamlit.io).
2. Create an app from this repository and select the `main` branch.
3. Set the entry point to `app.py`.
4. Deploy. Streamlit installs packages from `requirements.txt` automatically.

## Project structure

```text
app.py                  # Interactive portfolio dashboard
core/                   # Cloud-environment simulation
agents/                 # DQN scheduler implementation
utils/                  # Metrics and visualizations
main.py                 # Training and benchmark workflow
```

## Notes

This project uses synthetic workload data. Its visual metrics are suitable for comparing simulated policies; they should not be interpreted as measurements from a production cloud environment.

## License

Apache-2.0
