const $ = (selector) => document.querySelector(selector);
const seeded = (seed) => () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296;
function simulate(taskCount, vmCount, policy) {
  const random = seeded(taskCount * 31 + vmCount * 17 + policy.length);
  const machines = Array.from({ length: vmCount }, (_, index) => ({name: `VM-${index + 1}`, capacity: 550 + random() * 1350, rate: .12 + random() * .36, use: 0, jobs: 0}));
  const points = [], outcomes = []; let cumulative = 0;
  for (let task = 0; task < taskCount; task++) {
    const demand = 120 + random() * 830;
    let index;
    if (policy === 'round') index = task % vmCount;
    else if (policy === 'load') index = machines.reduce((best, vm, i) => vm.use < machines[best].use ? i : best, 0);
    else index = machines.reduce((best, vm, i) => {
      const score = vm.rate * .55 + vm.use * .30 + (1 - vm.capacity / 2000) * .15;
      const bestScore = machines[best].rate * .55 + machines[best].use * .30 + (1 - machines[best].capacity / 2000) * .15;
      return score < bestScore ? i : best;
    }, 0);
    const vm = machines[index], response = demand / vm.capacity * (1 + vm.use * .45), energy = vm.rate * demand * response / 3600;
    vm.use = Math.min(1, vm.use * .9 + demand / vm.capacity * .1); vm.jobs++; cumulative += energy;
    points.push(cumulative); outcomes.push({ response, deadline: 8 + random() * 47 });
  }
  const average = outcomes.reduce((sum, row) => sum + row.response, 0) / taskCount;
  const success = outcomes.filter((row) => row.response <= row.deadline).length / taskCount;
  const utilization = machines.map((vm) => vm.use), mean = utilization.reduce((sum, value) => sum + value, 0) / vmCount;
  const spread = Math.sqrt(utilization.reduce((sum, value) => sum + (value - mean) ** 2, 0) / vmCount);
  return { machines, points, energy: cumulative, average, success, balance: Math.max(0, 1 - spread) };
}
function drawChart(points) {
  const svg = $('#lineChart'), width = 800, height = 260, pad = 12, max = Math.max(...points);
  const sample = points.filter((_, index) => index % Math.ceil(points.length / 80) === 0 || index === points.length - 1);
  const path = sample.map((value, index) => `${index ? 'L' : 'M'} ${pad + index / (sample.length - 1) * (width - pad * 2)} ${height - pad - value / max * (height - pad * 2)}`).join(' ');
  svg.innerHTML = `<defs><linearGradient id="gradient" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#abf700"/><stop offset="1" stop-color="#abf700" stop-opacity="0"/></linearGradient></defs><line class="chart-grid" x1="${pad}" x2="${width - pad}" y1="65" y2="65"/><line class="chart-grid" x1="${pad}" x2="${width - pad}" y1="130" y2="130"/><line class="chart-grid" x1="${pad}" x2="${width - pad}" y1="195" y2="195"/><path class="chart-fill" d="${path} L ${width - pad} ${height - pad} L ${pad} ${height - pad} Z"/><path class="chart-line" d="${path}"/><text x="${pad}" y="${height - 1}" fill="#8e9b96" font-size="10">START</text><text x="${width - 40}" y="${height - 1}" fill="#8e9b96" font-size="10">END</text>`;
}
function render() {
  const tasks = +$('#tasks').value, vms = +$('#vms').value, policy = $('#policy').value;
  $('#taskValue').textContent = tasks; $('#vmValue').textContent = vms;
  const result = simulate(tasks, vms, policy);
  $('#energy').textContent = result.energy.toFixed(3); $('#success').textContent = `${(result.success * 100).toFixed(1)}%`;
  $('#response').textContent = result.average.toFixed(2); $('#balance').textContent = `${(result.balance * 100).toFixed(0)}%`;
  $('#policyBadge').textContent = $('#policy').selectedOptions[0].textContent;
  $('#vmBars').innerHTML = result.machines.map((vm) => `<div class="bar-row"><span>${vm.name}</span><div class="bar"><i style="width:${vm.use * 100}%"></i></div><span>${(vm.use * 100).toFixed(0)}%</span></div>`).join('');
  drawChart(result.points);
}
['tasks', 'vms', 'policy'].forEach((id) => $(`#${id}`).addEventListener('input', render));
$('#run').addEventListener('click', render); render();