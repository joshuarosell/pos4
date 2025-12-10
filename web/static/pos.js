const itemsEl = document.getElementById('items');
const totalEl = document.getElementById('total');
const sessionEl = document.getElementById('session-state');

let total = 0;

const ws = new WebSocket(`ws://${location.host}/ws`);

ws.addEventListener('open', () => {
  sessionEl.textContent = 'Session: connecting';
});

ws.addEventListener('message', (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === 'session') {
    sessionEl.textContent = `Session: ${msg.active ? 'active' : 'inactive'}`;
    itemsEl.innerHTML = '';
    total = 0;
    for (const [name, price] of msg.items || []) {
      addItem(name, price);
    }
    updateTotal();
  } else if (msg.type === 'session_start') {
    sessionEl.textContent = 'Session: active';
    itemsEl.innerHTML = '';
    total = 0;
    updateTotal();
    playBeep();
  } else if (msg.type === 'session_end') {
    sessionEl.textContent = 'Session: inactive';
    total = msg.total || total;
    updateTotal();
    playBeep(2);
  } else if (msg.type === 'item') {
    addItem(msg.name, msg.price);
    playBeep();
  }
});

ws.addEventListener('close', () => {
  sessionEl.textContent = 'Session: disconnected';
});

function addItem(name, price) {
  const li = document.createElement('li');
  li.innerHTML = `<span>${name}</span><span>$${Number(price).toFixed(2)}</span>`;
  itemsEl.appendChild(li);
  total += Number(price);
  updateTotal();
}

function updateTotal() {
  totalEl.textContent = `Total: $${total.toFixed(2)}`;
}

function playBeep(times = 1) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const duration = 0.08;
  const gap = 0.1;
  for (let i = 0; i < times; i++) {
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'square';
    osc.frequency.setValueAtTime(1200, ctx.currentTime);
    gain.gain.setValueAtTime(0.15, ctx.currentTime);
    osc.connect(gain).connect(ctx.destination);
    osc.start(ctx.currentTime + i * (duration + gap));
    osc.stop(ctx.currentTime + i * (duration + gap) + duration);
  }
}
