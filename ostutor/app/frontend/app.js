/* ==========================================================================
   OSTutor - Student-Centric Interactive Application Logic
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  // Navigation & Workspace Elements
  const chatHistory = document.getElementById('chatHistory');
  const chatInput = document.getElementById('chatInput');
  const sendBtn = document.getElementById('sendBtn');
  const navTabs = document.querySelectorAll('.tab-btn');
  const workspaceViews = document.querySelectorAll('.workspace-view');
  const navEbooksLink = document.getElementById('navEbooksLink');

  // CPU Simulator Elements
  const algoSelect = document.getElementById('algoSelect');
  const quantumFieldGroup = document.getElementById('quantumFieldGroup');
  const quantumInput = document.getElementById('quantumInput');
  const runSimBtn = document.getElementById('runSimBtn');
  const procTableBody = document.getElementById('procTableBody');
  const resetProcsBtn = document.getElementById('resetProcsBtn');

  const newPidInput = document.getElementById('newPid');
  const newArrivalInput = document.getElementById('newArrival');
  const newBurstInput = document.getElementById('newBurst');
  const newPriorityInput = document.getElementById('newPriority');
  const addProcBtn = document.getElementById('addProcBtn');

  const ganttBarRow = document.getElementById('ganttBarRow');
  const ganttLegend = document.getElementById('ganttLegend');
  const ganttTimeAxis = document.getElementById('ganttTimeAxis');
  const simMetricsBox = document.getElementById('simMetricsBox');

  // Presets
  const presetConvoy = document.getElementById('presetConvoy');
  const presetSjf = document.getElementById('presetSjf');
  const presetRr = document.getElementById('presetRr');
  const presetPriority = document.getElementById('presetPriority');

  // RAG & E-Books Elements
  const ragResultsList = document.getElementById('ragResultsList');
  const ebooksGrid = document.getElementById('ebooksGrid');

  // Quiz Elements
  const quizTopicSelect = document.getElementById('quizTopicSelect');
  const generateQuizBtn = document.getElementById('generateQuizBtn');
  const quizContainer = document.getElementById('quizContainer');

  // Flashcards Elements
  const fcTopicSelect = document.getElementById('fcTopicSelect');
  const flashcardCard = document.getElementById('flashcardCard');
  const fcBadge = document.getElementById('fcBadge');
  const fcTerm = document.getElementById('fcTerm');
  const fcQuestion = document.getElementById('fcQuestion');
  const fcAnswer = document.getElementById('fcAnswer');
  const fcAnalogy = document.getElementById('fcAnalogy');
  const fcCounter = document.getElementById('fcCounter');
  const prevFcBtn = document.getElementById('prevFcBtn');
  const nextFcBtn = document.getElementById('nextFcBtn');

  let activeTopicId = 'proc';

  // Process Queue State
  const defaultProcesses = [
    { id: 'P1', arrival: 0, burst: 6, priority: 2 },
    { id: 'P2', arrival: 1, burst: 3, priority: 1 },
    { id: 'P3', arrival: 2, burst: 8, priority: 3 },
    { id: 'P4', arrival: 3, burst: 4, priority: 2 }
  ];

  let currentProcesses = JSON.parse(JSON.stringify(defaultProcesses));

  const colorMap = {
    'P1': '#7c3aed',
    'P2': '#2563eb',
    'P3': '#f43f5e',
    'P4': '#f59e0b',
    'P5': '#10b981',
    'P6': '#8b5cf6',
    'P7': '#ec4899',
    'P8': '#06b6d4',
    'IDLE': '#94a3b8'
  };

  function getProcessColor(pid) {
    if (pid === 'IDLE') return '#94a3b8';
    if (colorMap[pid]) return colorMap[pid];
    let hash = 0;
    for (let i = 0; i < pid.length; i++) hash = pid.charCodeAt(i) + ((hash << 5) - hash);
    const c = (hash & 0x00FFFFFF).toString(16).toUpperCase();
    return '#' + '00000'.substring(0, 6 - c.length) + c;
  }

  // 1. Workspace Tab Switcher
  function activateTab(targetTab) {
    navTabs.forEach(t => t.classList.remove('active'));
    workspaceViews.forEach(v => v.classList.remove('active'));

    const tabBtn = Array.from(navTabs).find(t => t.getAttribute('data-tab') === targetTab);
    if (tabBtn) tabBtn.classList.add('active');

    const targetView = document.getElementById(`view-${targetTab}`);
    if (targetView) targetView.classList.add('active');

    if (targetTab === 'simulator') runScheduler();
    if (targetTab === 'quiz') fetchQuiz();
    if (targetTab === 'flashcards') fetchFlashcards();
    if (targetTab === 'ebooks') fetchEbooks();
  }

  navTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      activateTab(tab.getAttribute('data-tab'));
    });
  });

  if (navEbooksLink) {
    navEbooksLink.addEventListener('click', (e) => {
      e.preventDefault();
      activateTab('ebooks');
      const ws = document.getElementById('workspace');
      if (ws) ws.scrollIntoView({ behavior: 'smooth' });
    });
  }

  // 2. AI Tutor Chat Execution
  async function sendChatMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    const userItem = document.createElement('div');
    userItem.className = 'message-item user';
    userItem.innerHTML = `<div class="message-bubble">${escapeHtml(text)}</div>`;
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
        const traceLines = data.kernel_trace.map(t => `<div style="margin-bottom:4px;">${escapeHtml(t)}</div>`).join('');
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

  // 3. RAG Textbook Inspector & Citations
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
      const linkHtml = doc.url ? `
        <a href="${doc.url}" target="_blank" rel="noopener noreferrer" style="font-size:12px; font-weight:700; color:var(--brand-purple); text-decoration:none; display:inline-flex; align-items:center; gap:4px; margin-top:4px;">
          📖 Read Textbook Chapter →
        </a>
      ` : '';

      card.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span class="rag-topic">📘 ${doc.topic || 'OS Reference'}</span>
          <span style="font-size:11px; font-weight:700; background:var(--brand-purple-light); color:var(--brand-purple); padding:2px 6px; border-radius:6px;">${scorePct}% Match</span>
        </div>
        <div style="font-size:12px; font-weight:600; color:var(--text-dark);">${doc.source}</div>
        <div class="rag-text">${doc.content}</div>
        ${linkHtml}
      `;
      ragResultsList.appendChild(card);
    });
  }

  fetchRagInspector();

  // 4. CPU Scheduler Gantt Simulator & Process Editor
  function renderProcTable() {
    procTableBody.innerHTML = '';
    currentProcesses.forEach((p, idx) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong style="color:${getProcessColor(p.id)}">${p.id}</strong></td>
        <td><input type="number" class="sim-input-num" value="${p.arrival}" min="0" onchange="updateProc(${idx}, 'arrival', this.value)"></td>
        <td><input type="number" class="sim-input-num" value="${p.burst}" min="1" onchange="updateProc(${idx}, 'burst', this.value)"></td>
        <td><input type="number" class="sim-input-num" value="${p.priority}" min="1" onchange="updateProc(${idx}, 'priority', this.value)"></td>
        <td><button class="btn-delete-sm" onclick="removeProc(${idx})">✕</button></td>
      `;
      procTableBody.appendChild(tr);
    });
  }

  window.updateProc = function(idx, field, val) {
    if (currentProcesses[idx]) {
      currentProcesses[idx][field] = Math.max(field === 'burst' ? 1 : 0, parseInt(val) || 0);
      runScheduler();
    }
  };

  window.removeProc = function(idx) {
    currentProcesses.splice(idx, 1);
    renderProcTable();
    runScheduler();
  };

  if (addProcBtn) {
    addProcBtn.addEventListener('click', () => {
      const pid = newPidInput.value.trim() || `P${currentProcesses.length + 1}`;
      const arrival = Math.max(0, parseInt(newArrivalInput.value) || 0);
      const burst = Math.max(1, parseInt(newBurstInput.value) || 1);
      const priority = Math.max(1, parseInt(newPriorityInput.value) || 1);

      currentProcesses.push({ id: pid, arrival, burst, priority });
      newPidInput.value = '';
      renderProcTable();
      runScheduler();
    });
  }

  if (resetProcsBtn) {
    resetProcsBtn.addEventListener('click', () => {
      currentProcesses = JSON.parse(JSON.stringify(defaultProcesses));
      renderProcTable();
      runScheduler();
    });
  }

  // Presets Handlers
  if (presetConvoy) {
    presetConvoy.addEventListener('click', () => {
      algoSelect.value = 'FCFS';
      currentProcesses = [
        { id: 'P1', arrival: 0, burst: 24, priority: 3 },
        { id: 'P2', arrival: 0, burst: 3, priority: 1 },
        { id: 'P3', arrival: 0, burst: 3, priority: 2 }
      ];
      toggleQuantumGroup();
      renderProcTable();
      runScheduler();
    });
  }

  if (presetSjf) {
    presetSjf.addEventListener('click', () => {
      algoSelect.value = 'SRTF';
      currentProcesses = [
        { id: 'P1', arrival: 0, burst: 8, priority: 2 },
        { id: 'P2', arrival: 1, burst: 4, priority: 1 },
        { id: 'P3', arrival: 2, burst: 1, priority: 3 },
        { id: 'P4', arrival: 3, burst: 3, priority: 2 }
      ];
      toggleQuantumGroup();
      renderProcTable();
      runScheduler();
    });
  }

  if (presetRr) {
    presetRr.addEventListener('click', () => {
      algoSelect.value = 'RR';
      quantumInput.value = '2';
      currentProcesses = [
        { id: 'P1', arrival: 0, burst: 5, priority: 2 },
        { id: 'P2', arrival: 1, burst: 3, priority: 1 },
        { id: 'P3', arrival: 2, burst: 8, priority: 3 },
        { id: 'P4', arrival: 3, burst: 6, priority: 2 }
      ];
      toggleQuantumGroup();
      renderProcTable();
      runScheduler();
    });
  }

  if (presetPriority) {
    presetPriority.addEventListener('click', () => {
      algoSelect.value = 'PRIORITY_P';
      currentProcesses = [
        { id: 'P1', arrival: 0, burst: 10, priority: 3 },
        { id: 'P2', arrival: 0, burst: 1, priority: 1 },
        { id: 'P3', arrival: 2, burst: 2, priority: 4 },
        { id: 'P4', arrival: 3, burst: 1, priority: 2 }
      ];
      toggleQuantumGroup();
      renderProcTable();
      runScheduler();
    });
  }

  function toggleQuantumGroup() {
    if (quantumFieldGroup) {
      quantumFieldGroup.style.display = (algoSelect.value === 'RR') ? 'flex' : 'none';
    }
  }

  algoSelect.addEventListener('change', () => {
    toggleQuantumGroup();
    runScheduler();
  });

  quantumInput.addEventListener('change', runScheduler);

  toggleQuantumGroup();
  renderProcTable();

  async function runScheduler() {
    const algo = algoSelect.value;
    const quantum = parseInt(quantumInput.value) || 2;

    try {
      const res = await fetch('/api/simulator/schedule', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ algorithm: algo, processes: currentProcesses, quantum: quantum })
      });

      if (!res.ok) return;
      const data = await res.json();

      ganttBarRow.innerHTML = '';
      ganttTimeAxis.innerHTML = '';
      ganttLegend.innerHTML = '';

      const totalTime = data.total_time || 1;
      const uniquePids = new Set(data.gantt.map(b => b.pid));

      uniquePids.forEach(pid => {
        const item = document.createElement('div');
        item.className = 'legend-item';
        item.innerHTML = `
          <div class="legend-dot" style="background:${getProcessColor(pid)};"></div>
          <span>${pid === 'IDLE' ? 'IDLE CPU' : pid}</span>
        `;
        ganttLegend.appendChild(item);
      });

      const tickPositions = new Set([0]);

      data.gantt.forEach(block => {
        const pct = (block.duration / totalTime) * 100;
        const div = document.createElement('div');
        div.className = `gantt-block ${block.pid === 'IDLE' ? 'idle' : ''}`;
        div.style.width = `${pct}%`;
        div.style.backgroundColor = getProcessColor(block.pid);
        div.title = `${block.pid}: ${block.start}ms → ${block.end}ms (${block.duration}ms)`;
        div.innerHTML = `<span>${block.pid}</span><span style="font-size:10px; opacity:0.85;">${block.duration}ms</span>`;
        ganttBarRow.appendChild(div);

        tickPositions.add(block.end);
      });

      tickPositions.forEach(timeVal => {
        const pct = (timeVal / totalTime) * 100;
        const tick = document.createElement('div');
        tick.className = 'time-tick';
        tick.style.left = `${pct}%`;
        tick.textContent = `${timeVal}ms`;
        ganttTimeAxis.appendChild(tick);
      });

      let metricsRows = '';
      if (data.process_metrics) {
        metricsRows = data.process_metrics.map(m => `
          <tr>
            <td><strong style="color:${getProcessColor(m.id)}">${m.id}</strong></td>
            <td>${m.arrival} ms</td>
            <td>${m.burst} ms</td>
            <td>${m.priority}</td>
            <td><strong>${m.completion_time} ms</strong></td>
            <td><span style="color:var(--accent-blue); font-weight:700;">${m.turnaround_time} ms</span></td>
            <td><span style="color:var(--brand-purple); font-weight:700;">${m.waiting_time} ms</span></td>
          </tr>
        `).join('');
      }

      simMetricsBox.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; background:var(--surface-purple); padding:12px 16px; border-radius:10px;">
          <span>Algorithm: <strong>${data.algorithm}</strong> ${data.algorithm === 'RR' ? `(Quantum: ${data.quantum}ms)` : ''}</span>
          <span>Total Execution: <strong>${data.total_time} ms</strong></span>
          <span>Avg Wait Time: <strong style="color:var(--brand-purple);">${data.avg_waiting_time} ms</strong></span>
          <span>Avg Turnaround: <strong style="color:var(--accent-blue);">${data.avg_turnaround_time} ms</strong></span>
        </div>

        <div class="table-responsive">
          <table class="proc-metrics-table">
            <thead>
              <tr>
                <th>PID</th>
                <th>Arrival Time</th>
                <th>Burst Time</th>
                <th>Priority</th>
                <th>Completion Time (CT)</th>
                <th>Turnaround Time (TAT)</th>
                <th>Waiting Time (WT)</th>
              </tr>
            </thead>
            <tbody>
              ${metricsRows}
            </tbody>
          </table>
        </div>
      `;
    } catch (err) {
      console.error('Scheduler error:', err);
    }
  }

  runSimBtn.addEventListener('click', runScheduler);

  // 5. Quiz Generator State & Logic
  let currentQuizQuestions = [];

  async function fetchQuiz() {
    const topic = quizTopicSelect ? quizTopicSelect.value : 'All';
    try {
      const res = await fetch('/api/quiz/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic: topic, count: 5 })
      });

      if (!res.ok) return;
      const data = await res.json();
      currentQuizQuestions = data.questions || [];
      renderQuizCards();
    } catch (err) {
      console.error('Quiz fetch error:', err);
    }
  }

  function renderQuizCards() {
    if (!quizContainer) return;
    quizContainer.innerHTML = '';

    if (!currentQuizQuestions || currentQuizQuestions.length === 0) {
      quizContainer.innerHTML = '<div style="text-align:center; padding:20px;">No quiz questions found for selected topic.</div>';
      return;
    }

    currentQuizQuestions.forEach((q, qIdx) => {
      const card = document.createElement('div');
      card.className = 'quiz-q-card';

      const optionsHtml = q.options.map((opt, optIdx) => `
        <button class="quiz-opt-btn" onclick="checkQuizAnswer(${qIdx}, ${optIdx}, this)">
          ${escapeHtml(opt)}
        </button>
      `).join('');

      card.innerHTML = `
        <div class="quiz-q-header">
          <span class="quiz-topic-tag">📌 ${q.topic}</span>
          <span class="quiz-diff-tag">Difficulty: ${q.difficulty}</span>
        </div>
        <div class="quiz-question-text">Q${qIdx + 1}: ${escapeHtml(q.question)}</div>
        <div class="quiz-options-list" id="opts-list-${qIdx}">
          ${optionsHtml}
        </div>
        <div class="quiz-exp-box" id="exp-box-${qIdx}" style="display:none;"></div>
      `;
      quizContainer.appendChild(card);
    });
  }

  window.checkQuizAnswer = function(qIdx, optIdx, btnEl) {
    const q = currentQuizQuestions[qIdx];
    if (!q) return;

    const optsList = document.getElementById(`opts-list-${qIdx}`);
    const expBox = document.getElementById(`exp-box-${qIdx}`);

    const allBtns = optsList.querySelectorAll('.quiz-opt-btn');
    allBtns.forEach((b, idx) => {
      b.disabled = true;
      if (idx === q.answer_index) {
        b.classList.add('correct');
      } else if (idx === optIdx && optIdx !== q.answer_index) {
        b.classList.add('wrong');
      }
    });

    if (expBox) {
      expBox.style.display = 'block';
      const isCorrect = (optIdx === q.answer_index);
      expBox.innerHTML = `
        <div style="font-weight:700; color:${isCorrect ? 'var(--accent-emerald)' : 'var(--accent-pink)'}; margin-bottom:4px;">
          ${isCorrect ? '🎉 Correct Answer!' : '❌ Incorrect Answer'}
        </div>
        <div>${escapeHtml(q.explanation)}</div>
      `;
    }
  };

  if (generateQuizBtn) {
    generateQuizBtn.addEventListener('click', fetchQuiz);
  }

  // 6. OS Flashcards Deck State & Logic
  let currentFlashcards = [];
  let currentFcIndex = 0;
  let isCardFlipped = false;

  async function fetchFlashcards() {
    const topic = fcTopicSelect ? fcTopicSelect.value : 'All';
    try {
      const res = await fetch(`/api/flashcards?topic=${encodeURIComponent(topic)}`);
      if (!res.ok) return;
      const data = await res.json();
      currentFlashcards = data.flashcards || [];
      currentFcIndex = 0;
      isCardFlipped = false;
      renderCurrentFlashcard();
    } catch (err) {
      console.error('Flashcard fetch error:', err);
    }
  }

  function renderCurrentFlashcard() {
    if (!currentFlashcards || currentFlashcards.length === 0) return;

    if (flashcardCard) flashcardCard.classList.remove('is-flipped');
    isCardFlipped = false;

    const fc = currentFlashcards[currentFcIndex];
    if (fcBadge) fcBadge.textContent = fc.topic;
    if (fcTerm) fcTerm.textContent = fc.term;
    if (fcQuestion) fcQuestion.textContent = fc.front;
    if (fcAnswer) fcAnswer.textContent = fc.back;
    if (fcAnalogy) fcAnalogy.textContent = fc.analogy || '';
    if (fcCounter) fcCounter.textContent = `Card ${currentFcIndex + 1} of ${currentFlashcards.length}`;
  }

  if (flashcardCard) {
    flashcardCard.addEventListener('click', () => {
      isCardFlipped = !isCardFlipped;
      flashcardCard.classList.toggle('is-flipped', isCardFlipped);
    });
  }

  if (prevFcBtn) {
    prevFcBtn.addEventListener('click', () => {
      if (currentFlashcards.length === 0) return;
      currentFcIndex = (currentFcIndex - 1 + currentFlashcards.length) % currentFlashcards.length;
      renderCurrentFlashcard();
    });
  }

  if (nextFcBtn) {
    nextFcBtn.addEventListener('click', () => {
      if (currentFlashcards.length === 0) return;
      currentFcIndex = (currentFcIndex + 1) % currentFlashcards.length;
      renderCurrentFlashcard();
    });
  }

  if (fcTopicSelect) {
    fcTopicSelect.addEventListener('change', fetchFlashcards);
  }

  // 7. OS E-Book Library Integration
  async function fetchEbooks() {
    try {
      const res = await fetch('/api/ebooks');
      if (!res.ok) return;
      const data = await res.json();
      renderEbookCards(data.ebooks);
    } catch (err) {
      console.error('E-Books fetch error:', err);
    }
  }

  function renderEbookCards(books) {
    if (!ebooksGrid) return;
    ebooksGrid.innerHTML = '';
    books.forEach(b => {
      const card = document.createElement('div');
      card.className = 'ebook-card';

      const unitsHtml = (b.units || []).map(u => `<span class="unit-tag">${u}</span>`).join('');

      card.innerHTML = `
        <div class="ebook-header">
          <span class="ebook-badge" style="background:${b.cover_color}20; color:${b.cover_color}; border:1px solid ${b.cover_color}40;">${b.badge}</span>
          <h4 class="ebook-title">${b.title}</h4>
          <div class="ebook-authors">✍️ ${b.authors}</div>
        </div>
        <p class="ebook-desc">${b.description}</p>
        <div class="ebook-units">${unitsHtml}</div>
        <a href="${b.url}" target="_blank" rel="noopener noreferrer" class="ebook-link-btn" style="background:${b.cover_color};">
          📖 Read E-Book / PDF →
        </a>
      `;
      ebooksGrid.appendChild(card);
    });
  }

  fetchEbooks();

  function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
});
