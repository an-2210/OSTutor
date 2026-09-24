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
  const runSimBtn = document.getElementById('runSimBtn');
  const ganttBarRow = document.getElementById('ganttBarRow');
  const simMetricsBox = document.getElementById('simMetricsBox');

  const shapBarsContainer = document.getElementById('shapBarsContainer');
  const ragResultsList = document.getElementById('ragResultsList');

  let activeTopicId = 'proc';

  // 1. Topic Selector Global Callback
  window.selectTopic = function(topicId, topicTitle) {
    activeTopicId = topicId;

    // Update sidebar active buttons
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
      if (targetTab === 'shap') fetchShapData();
    });
  });

  // 3. AI Tutor Chat Execution
  async function sendChatMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    // Append Student Query Bubble
    const userItem = document.createElement('div');
    userItem.className = 'message-item user';
    userItem.innerHTML = `
      <div class="message-bubble">${escapeHtml(text)}</div>
    `;
    chatHistory.appendChild(userItem);
    chatInput.value = '';
    chatHistory.scrollTop = chatHistory.scrollHeight;

    // Fetch Tutor Response from FastAPI Backend
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

  // 4. RAG Textbook Inspector
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
      card.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span class="rag-topic">📘 ${doc.topic || 'OS Reference'}</span>
          <span style="font-size:11px; font-weight:700; background:var(--brand-purple-light); color:var(--brand-purple); padding:2px 6px; border-radius:6px;">${scorePct}% Match</span>
        </div>
        <div style="font-size:12px; font-weight:600; color:var(--text-dark);">${doc.source}</div>
        <div class="rag-text">${doc.content}</div>
      `;
      ragResultsList.appendChild(card);
    });
  }

  fetchRagInspector();

  // 5. CPU Scheduler Gantt Simulator
  async function runScheduler() {
    const algo = algoSelect.value;
    try {
      const res = await fetch('/api/simulator/schedule', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ algorithm: algo, processes: [], quantum: 2 })
      });

      if (!res.ok) return;
      const data = await res.json();

      ganttBarRow.innerHTML = '';
      const totalTime = data.total_time || 1;
      const colors = {
        'P1': '#7c3aed',
        'P2': '#2563eb',
        'P3': '#f43f5e',
        'P4': '#f59e0b'
      };

      data.gantt.forEach(block => {
        const pct = (block.duration / totalTime) * 100;
        const div = document.createElement('div');
        div.className = 'gantt-block';
        div.style.width = `${pct}%`;
        div.style.backgroundColor = colors[block.pid] || '#6d28d9';
        div.textContent = `${block.pid} (${block.duration}ms)`;
        ganttBarRow.appendChild(div);
      });

      simMetricsBox.innerHTML = `
        <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
          <span>Algorithm: <strong>${data.algorithm}</strong></span>
          <span>Total Execution: <strong>${data.total_time} ms</strong></span>
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

  // 6. SHAP Feature Telemetry
  async function fetchShapData() {
    try {
      const res = await fetch('/api/explain/shap', { method: 'POST' });
      if (!res.ok) return;
      const data = await res.json();

      shapBarsContainer.innerHTML = '';
      data.features.forEach(feat => {
        const absVal = Math.abs(feat.shap_value);
        const pct = Math.min(100, Math.round((absVal / 0.5) * 100));

        const row = document.createElement('div');
        row.className = 'shap-row';
        row.innerHTML = `
          <div class="shap-label">
            <span>${feat.name} [Value: ${feat.feature_value}]</span>
            <span style="font-weight:700; color: ${feat.shap_value >= 0 ? 'var(--brand-purple)' : 'var(--accent-pink)'}">
              ${feat.shap_value >= 0 ? '+' : ''}${feat.shap_value}
            </span>
          </div>
          <div class="shap-bar-bg">
            <div class="shap-bar-fill" style="width: ${pct}%; background: ${feat.shap_value >= 0 ? 'linear-gradient(90deg, var(--brand-purple), var(--accent-blue))' : 'var(--accent-pink)'};"></div>
          </div>
        `;
        shapBarsContainer.appendChild(row);
      });
    } catch (err) {
      console.error('SHAP error:', err);
    }
  }

  function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
});
