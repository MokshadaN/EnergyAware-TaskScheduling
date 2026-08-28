const $ = (selector) => document.querySelector(selector);
const seeded = (seed) => () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296;

function simulate(taskCount, vmCount, policy) {
  const random = seeded(taskCount * 31 + vmCount * 17 + policy.length);
  const machines = Array.from({ length: vmCount }, (_, index) => ({
    name: "VM-" + (index + 1),
    capacity: 550 + random() * 1350,
    rate: 0.12 + random() * 0.36,
    use: 0,
  }));

  const points = [];
  const outcomes = [];
  let cumulative = 0;

  for (let task = 0; task < taskCount; task++) {
    const demand = 120 + random() * 830;
    let index;

    if (policy === "round") {
      index = task % vmCount;
    } else if (policy === "load") {
      index = machines.reduce((best, vm, i) => (vm.use < machines[best].use ? i : best), 0);
    } else if (policy === "edf") {
      index = machines.reduce((best, vm, i) => {
        const score = vm.use * 0.7 + (1 - vm.capacity / 2000) * 0.3;
        const bestScore = machines[best].use * 0.7 + (1 - machines[best].capacity / 2000) * 0.3;
        return score < bestScore ? i : best;
      }, 0);
    } else {
      // Energy-Aware DQN heuristic score
      index = machines.reduce((best, vm, i) => {
        const score = vm.rate * 0.55 + vm.use * 0.30 + (1 - vm.capacity / 2000) * 0.15;
        const bestScore =
          machines[best].rate * 0.55 +
          machines[best].use * 0.30 +
          (1 - machines[best].capacity / 2000) * 0.15;
        return score < bestScore ? i : best;
      }, 0);
    }

    const vm = machines[index];
    const response = (demand / vm.capacity) * (1 + vm.use * 0.45);
    const energy = (vm.rate * demand * response) / 3600;
    vm.use = Math.min(1, vm.use * 0.9 + (demand / vm.capacity) * 0.1);
    cumulative += energy;
    points.push(cumulative);
    outcomes.push({ response: response, deadline: 8 + random() * 47 });
  }

  const average = outcomes.reduce((sum, row) => sum + row.response, 0) / taskCount;
  const success = outcomes.filter((row) => row.response <= row.deadline).length / taskCount;
  const utilization = machines.map((vm) => vm.use);
  const mean = utilization.reduce((sum, value) => sum + value, 0) / vmCount;
  const spread = Math.sqrt(
    utilization.reduce((sum, value) => sum + (value - mean) ** 2, 0) / vmCount
  );

  return {
    machines: machines,
    points: points,
    energy: cumulative,
    average: average,
    success: success,
    balance: Math.max(0, 1 - spread),
  };
}

function drawChart(points) {
  const svg = $("#lineChart");
  if (!svg) return;
  const width = 800;
  const height = 260;
  const pad = 12;
  const max = Math.max(...points, 1);

  const sample = points.filter(
    (_, index) => index % Math.ceil(points.length / 80) === 0 || index === points.length - 1
  );

  const path = sample
    .map(
      (value, index) =>
        (index ? "L " : "M ") +
        (pad + (index / (sample.length - 1)) * (width - pad * 2)) +
        " " +
        (height - pad - (value / max) * (height - pad * 2))
    )
    .join(" ");

  svg.innerHTML =
    '<defs><linearGradient id="gradient" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#abf700"/><stop offset="1" stop-color="#abf700" stop-opacity="0"/></linearGradient></defs><line class="chart-grid" x1="' +
    pad +
    '" x2="' +
    (width - pad) +
    '" y1="65" y2="65"/><line class="chart-grid" x1="' +
    pad +
    '" x2="' +
    (width - pad) +
    '" y1="130" y2="130"/><line class="chart-grid" x1="' +
    pad +
    '" x2="' +
    (width - pad) +
    '" y1="195" y2="195"/><path class="chart-fill" d="' +
    path +
    " L " +
    (width - pad) +
    " " +
    (height - pad) +
    " L " +
    pad +
    " " +
    (height - pad) +
    ' Z"/><path class="chart-line" d="' +
    path +
    '"/>';
}

function render() {
  const tasks = +$("#tasks").value;
  const vms = +$("#vms").value;
  const policy = $("#policy").value;
  const result = simulate(tasks, vms, policy);

  if ($("#taskValue")) $("#taskValue").textContent = tasks;
  if ($("#vmValue")) $("#vmValue").textContent = vms;
  if ($("#energy")) $("#energy").textContent = result.energy.toFixed(3);
  if ($("#success")) $("#success").textContent = (result.success * 100).toFixed(1) + "%";
  if ($("#response")) $("#response").textContent = result.average.toFixed(2);
  if ($("#balance")) $("#balance").textContent = (result.balance * 100).toFixed(0) + "%";
  if ($("#policyBadge")) $("#policyBadge").textContent = $("#policy").selectedOptions[0].textContent;

  if ($("#vmBars")) {
    $("#vmBars").innerHTML = result.machines
      .map(
        (vm) =>
          '<div class="bar-row"><span>' +
          vm.name +
          '</span><div class="bar"><i style="width:' +
          (vm.use * 100).toFixed(0) +
          '%"></i></div><span>' +
          (vm.use * 100).toFixed(0) +
          "%</span></div>"
      )
      .join("");
  }

  drawChart(result.points);
}

function updateReadyStatus() {
  const tasksVal = $("#tasks") ? $("#tasks").value : "250";
  const vmsVal = $("#vms") ? $("#vms").value : "10";

  // Update output badge labels dynamically on input scroll/drag
  if ($("#taskValue")) $("#taskValue").textContent = tasksVal;
  if ($("#vmValue")) $("#vmValue").textContent = vmsVal;

  if ($("#simulationStatus")) {
    $("#simulationStatus").textContent =
      "Ready to simulate " + tasksVal + " tasks across " + vmsVal + " virtual machines.";
  }
}

// Bind both 'input' and 'change' events so slider scroll/drag updates numbers instantaneously
["tasks", "vms", "policy"].forEach((id) => {
  const el = $("#" + id);
  if (el) {
    el.addEventListener("input", updateReadyStatus);
    el.addEventListener("change", updateReadyStatus);
  }
});

const runBtn = $("#run");
if (runBtn) {
  runBtn.addEventListener("click", () => {
    const label = runBtn.querySelector(".button-label");
    const status = $("#simulationStatus");

    runBtn.disabled = true;
    runBtn.classList.add("running");
    if (label) label.textContent = "Simulating";
    if (status) {
      status.textContent = "Simulation running: evaluating VM allocation and calculating energy traces...";
      status.classList.add("running");
    }

    window.setTimeout(() => {
      render();
      runBtn.disabled = false;
      runBtn.classList.remove("running");
      if (label) label.textContent = "Run simulation";
      if (status) {
        status.textContent = "Simulation complete: metrics updated for " + $("#tasks").value + " tasks.";
        status.classList.remove("running");
      }
    }, 600);
  });
}

render();
updateReadyStatus();