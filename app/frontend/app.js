/* ==========================================================================
   OSTutor - Student-Centric Interactive Application Logic
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const chatHistory = document.getElementById('chatHistory');
  const chatInput = document.getElementById('chatInput');
  const sendBtn = document.getElementById('sendBtn');
  const navTabs = document.querySelectorAll('.tab-btn');
  const workspaceViews = document.querySelectorAll('.workspace-view');

  const algoSelect = document.getElementById('algoSelect');
  const quantumInput = document.getElementById('quantumInput');
  const runSimBtn = document.getElementById('runSimBtn');
  const addProcBtn = document.getElementById('addProcBtn');
  const procTableBody = document.getElementById('procTableBody');
  const ganttBarRow = document.getElementById('ganttBarRow');
  const simMetricsBox = document.getElementById('simMetricsBox');

  const ragResultsList = document.getElementById('ragResultsList');

  let activeTopicId = 'proc';

  // Process list state for CPU Scheduler
  let processList = [
    { id: 'P1', arrival: 0, burst: 6, priority: 2 },
    { id: 'P2', arrival: 1, burst: 3, priority: 1 },
    { id: 'P3', arrival: 2, burst: 8, priority: 3 },
    { id: 'P4', arrival: 3, burst: 4, priority: 2 }
  ];

  // Render Process Input Table
  function renderProcessTable() {
    procTableBody.innerHTML = '';
    processList.forEach((proc, index) => {
      const tr = document.createElement('tr');
      tr.style.borderBottom = '1px solid var(--border-light)';
      tr.innerHTML = `
        <td style="padding: 8px;">
          <input type="text" value="${proc.id}" data-index="${index}" data-field="id" class="proc-field" style="width:60px;">
        </td>
        <td style="padding: 8px;">
          <input type="number" value="${proc.arrival}" min="0" data-index="${index}" data-field="arrival" class="proc-field" style="width:70px;">
        </td>
        <td style="padding: 8px;">
          <input type="number" value="${proc.burst}" min="1" data-index="${index}" data-field="burst" class="proc-field" style="width:70px;">
        </td>
        <td style="padding: 8px;">
          <input type="number" value="${proc.priority}" min="1" data-index="${index}" data-field="priority" class="proc-field" style="width:70px;">
        </td>
        <td style="padding: 8px; text-align: center;">
          <button class="delete-proc-btn" data-index="${index}" style="background:var(--accent-pink-light); color:var(--accent-pink); border:none; padding:6px 12px; border-radius:8px; font-weight:700; cursor:pointer; font-size:12px;">Delete</button>
        </td>
      `;
      procTableBody.appendChild(tr);
    });

    // Attach Input Event Listeners for live recalculation
    document.querySelectorAll('.proc-field').forEach(input => {
      const handleInput = (e) => {
        const idx = parseInt(e.target.getAttribute('data-index'));
        const field = e.target.getAttribute('data-field');
        const val = e.target.value;
        if (field === 'id') processList[idx].id = val;
        else processList[idx][field] = parseInt(val) || 0;
        runScheduler();
      };
      input.addEventListener('input', handleInput);
      input.addEventListener('change', handleInput);
    });

    // Attach Delete Event Listeners
    document.querySelectorAll('.delete-proc-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const idx = parseInt(e.target.getAttribute('data-index'));
        processList.splice(idx, 1);
        renderProcessTable();
        runScheduler();
      });
    });
  }

  renderProcessTable();

  // Add Process Button
  addProcBtn.addEventListener('click', () => {
    const newNum = processList.length + 1;
    processList.push({
      id: `P${newNum}`,
      arrival: processList.length,
      burst: 4,
      priority: 2
    });
    renderProcessTable();
    runScheduler();
  });

  // 1. Topic Selector Global Callback
  window.selectTopic = function(topicId, topicTitle) {
    activeTopicId = topicId;

    document.querySelectorAll('.side-topic-btn').forEach(btn => {
      if (btn.getAttribute('data-id') === topicId) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });

    chatInput.value = `Explain ${topicTitle} in detail with step-by-step kernel trace`;
    chatInput.focus();

    fetchRagInspector(topicTitle);
  };

  // 2. Workspace Tab Switcher
  navTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetTab = tab.getAttribute('data-tab');

      navTabs.forEach(t => t.classList.remove('active'));
      workspaceViews.forEach(v => v.classList.remove('active'));

      tab.classList.add('active');
      const targetView = document.getElementById(`view-${targetTab}`);
      if (targetView) targetView.classList.add('active');

      if (targetTab === 'simulator') runScheduler();
    });
  });

  // 3. AI Tutor Chat Execution
  async function sendChatMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    const userItem = document.createElement('div');
    userItem.className = 'message-item user';
    userItem.innerHTML = `
      <div class="message-bubble">${escapeHtml(text)}</div>
    `;
    chatHistory.appendChild(userItem);
    chatInput.value = '';
    chatHistory.scrollTop = chatHistory.scrollHeight;

    try {
      const res = await fetch('/api/tutor/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, topic: activeTopicId, use_rag: true })
      });

      const data = await res.json();

      let formattedText = data.response
        .replace(/### (.*?)\n/g, '<h3 style="color:var(--brand-purple); font-size:16px; margin: 10px 0 6px;">$1</h3>')
        .replace(/#### (.*?)\n/g, '<h4 style="color:var(--text-dark); font-size:14px; margin: 8px 0 4px;">$1</h4>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/`([^`]+)`/g, '<code style="background:var(--brand-purple-light); color:var(--brand-purple); padding:2px 6px; border-radius:4px; font-family:var(--font-mono); font-size:13px;">$1</code>');

      let traceHtml = '';
      if (data.kernel_trace && data.kernel_trace.length > 0) {
        const traceLines = data.kernel_trace.map(t => `<div>&gt; ${escapeHtml(t)}</div>`).join('');
        traceHtml = `
          <div class="kernel-trace-box">
            <div class="kernel-trace-title">⚡ Step-by-Step Kernel Execution Trace</div>
            ${traceLines}
          </div>
        `;
      }

      const tutorItem = document.createElement('div');
      tutorItem.className = 'message-item tutor';
      tutorItem.innerHTML = `
        <div class="message-bubble">
          ${formattedText}
          ${traceHtml}
        </div>
      `;
      chatHistory.appendChild(tutorItem);
      chatHistory.scrollTop = chatHistory.scrollHeight;

      if (data.rag_citations && data.rag_citations.length > 0) {
        renderRagCards(data.rag_citations);
      }
    } catch (err) {
      console.error('Chat error:', err);
    }
  }

  sendBtn.addEventListener('click', sendChatMessage);
  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendChatMessage();
  });

  // 4. RAG Textbook Inspector with Clickable Links
  async function fetchRagInspector(query = 'Process Management Virtual Memory') {
    try {
      const res = await fetch('/api/rag/retrieve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query, top_k: 4 })
      });
      if (!res.ok) return;
      const data = await res.json();
      renderRagCards(data.results);
    } catch (err) {
      console.error('RAG fetch error:', err);
    }
  }

  function renderRagCards(docs) {
    ragResultsList.innerHTML = '';
    docs.forEach(doc => {
      const card = document.createElement('div');
      card.className = 'rag-card';
      const scorePct = Math.round((doc.similarity_score || doc.score || 0.88) * 100);
      const docUrl = doc.url || 'https://www.os-book.com/OS10/';

      card.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span class="rag-topic">📖 ${doc.topic || 'OS Reference'}</span>
          <span style="font-size:11px; font-weight:700; background:var(--brand-purple-light); color:var(--brand-purple); padding:2px 6px; border-radius:6px;">${scorePct}% Match</span>
        </div>
        <div style="font-size:13px; margin-top:4px;">
          <a href="${docUrl}" target="_blank" rel="noopener noreferrer" style="color:var(--brand-purple); font-weight:700; text-decoration:underline;" onclick="event.stopPropagation();">
            📘 ${escapeHtml(doc.source)} ↗
          </a>
        </div>
        <div class="rag-text" style="margin-top:6px;">${escapeHtml(doc.content)}</div>
      `;

      card.addEventListener('click', () => {
        window.open(docUrl, '_blank');
      });

      ragResultsList.appendChild(card);
    });
  }

  fetchRagInspector();

  // 5. CPU Scheduler Gantt Simulator with Custom Process List Input
  async function runScheduler() {
    const algo = algoSelect.value;
    const quantum = parseInt(quantumInput.value) || 2;

    try {
      const res = await fetch('/api/simulator/schedule', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ algorithm: algo, processes: processList, quantum: quantum })
      });

      if (!res.ok) return;
      const data = await res.json();

      ganttBarRow.innerHTML = '';
      const totalTime = data.total_time || 1;
      const palette = ['#7c3aed', '#2563eb', '#f43f5e', '#f59e0b', '#10b981', '#06b6d4'];

      data.gantt.forEach((block, i) => {
        const pct = (block.duration / totalTime) * 100;
        const div = document.createElement('div');
        div.className = 'gantt-block';
        div.style.width = `${pct}%`;
        const colorIdx = (parseInt(block.pid.replace(/\D/g, '')) - 1) % palette.length;
        div.style.backgroundColor = palette[colorIdx >= 0 ? colorIdx : i % palette.length];
        div.textContent = `${block.pid} (${block.duration}ms)`;
        ganttBarRow.appendChild(div);
      });

      simMetricsBox.innerHTML = `
        <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
          <span>Algorithm: <strong>${data.algorithm} ${data.algorithm === 'RR' ? '(Q=' + quantum + 'ms)' : ''}</strong></span>
          <span>Total Execution Time: <strong>${data.total_time} ms</strong></span>
        </div>
        <div style="display:flex; justify-content:space-between;">
          <span>Avg Waiting Time: <strong style="color:var(--brand-purple);">${data.avg_waiting_time} ms</strong></span>
          <span>Avg Turnaround Time: <strong style="color:var(--accent-blue);">${data.avg_turnaround_time} ms</strong></span>
        </div>
      `;
    } catch (err) {
      console.error('Scheduler error:', err);
    }
  }

  runSimBtn.addEventListener('click', runScheduler);

  function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
});
