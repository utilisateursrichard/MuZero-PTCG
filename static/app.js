/**
 * PTCG MuZero Arena — Frontend Controller & Visualizer
 * Inspired by PTCG_ABCS_Visualizer (hiro094)
 */

const ENERGY = {
  0: ['C', '#b8b8b8'],
  1: ['G', '#4caf50'],
  2: ['R', '#f44336'],
  3: ['W', '#2196f3'],
  4: ['L', '#fbc02d'],
  5: ['P', '#9c27b0'],
  6: ['F', '#c66a2e'],
  7: ['D', '#455a64'],
  8: ['M', '#90a4ae'],
  9: ['N', '#c0a020'],
  10: ['*', '#e0e0e0'],
  11: ['TR', '#8d6e63'],
};

let LANG = 'en';
try {
  LANG = localStorage.getItem('ptcg_lang') || 'en';
} catch (e) {}

function cardImgUrl(id) {
  if (!id) return '';
  return 'assets/cards' + (LANG === 'ja' ? '_jp' : '') + '/' + id + '.jpg';
}

class BattleVisualizerApp {
  constructor() {
    this.state = null;
    this.allDecks = [];
    this.selectedDeckId = 'model_deck';
    this.selectedDeckCards = null;
    this.multiSelectedIndices = new Set();
    this.isSubmitting = false;
    this.zoomPi = null;

    this.initDOM();
    this.initEvents();
    this.loadDecks();
    this.checkActiveBattle();
  }

  initDOM() {
    // Header
    this.turnEl = document.getElementById('turn');
    this.activeEl = document.getElementById('active');
    this.ctxEl = document.getElementById('ctx');
    this.langBtn = document.getElementById('langbtn');
    this.deckBtn = document.getElementById('deckbtn');
    this.resetBtn = document.getElementById('resetbtn');

    this.langBtn.textContent = LANG === 'ja' ? 'EN' : 'JP';

    // Panels
    this.panel0 = document.getElementById('panel0');
    this.panel1 = document.getElementById('panel1');
    this.stadiumEl = document.getElementById('stadium');
    this.stadiumNameEl = document.getElementById('stadiumName');
    this.contextPromptEl = document.getElementById('contextPrompt');

    // AI HUD
    this.aiWinrateEl = document.getElementById('aiWinrate');
    this.aiBarFillEl = document.getElementById('aiBarFill');
    this.aiIntentRowEl = document.getElementById('aiIntentRow');

    // Sidebar
    this.actionsListEl = document.getElementById('actionsList');
    this.actionCountBadgeEl = document.getElementById('actionCountBadge');
    this.multiSelectControlsEl = document.getElementById('multiSelectControls');
    this.btnConfirmMultiEl = document.getElementById('btnConfirmMulti');
    this.logEl = document.getElementById('log');
    this.btnClearLog = document.getElementById('btnClearLog');

    // Preview
    this.previewEl = document.getElementById('preview');
    this.pvImg = document.getElementById('pvimg');
    this.pvCap = document.getElementById('pvcap');
    this.pvEnergy = document.getElementById('pvenergy');
    this.pvTools = document.getElementById('pvtools');
    this.pvId = null;

    // Zoom Discard Modal
    this.zoomEl = document.getElementById('zoom');
    this.zoomTitleEl = document.getElementById('zoom-title');
    this.zoomGridEl = document.getElementById('zoom-grid');
    this.zoomCloseEl = document.getElementById('zoom-close');

    // Deck Modal
    this.deckModal = document.getElementById('deckModal');
    this.btnCloseDeckModal = document.getElementById('btnCloseDeckModal');
    this.btnCancelDeck = document.getElementById('btnCancelDeck');
    this.btnStartGame = document.getElementById('btnStartGame');
    this.btnTriggerGenerate = document.getElementById('btnTriggerGenerate');
    this.deckPreviewGrid = document.getElementById('deckPreviewGrid');
    this.previewTitle = document.getElementById('previewTitle');
    this.prevPokeCount = document.getElementById('prevPokeCount');
    this.prevTrainCount = document.getElementById('prevTrainCount');
    this.prevNrgCount = document.getElementById('prevNrgCount');
    this.deviceSelect = document.getElementById('deviceSelect');
    this.aiModeSelect = document.getElementById('aiModeSelect');

    // Thinking Overlay
    this.thinkingOverlay = document.getElementById('thinkingOverlay');
    this.thinkingSubText = document.getElementById('thinkingSubText');

    // Game Over
    this.gameOverModal = document.getElementById('gameOverModal');
    this.gameOverIcon = document.getElementById('gameOverIcon');
    this.gameOverTitle = document.getElementById('gameOverTitle');
    this.gameOverDesc = document.getElementById('gameOverDesc');
    this.btnRematch = document.getElementById('btnRematch');
    this.btnNewDeck = document.getElementById('btnNewDeck');
  }



  initEvents() {
    this.langBtn.addEventListener('click', () => {
      LANG = LANG === 'ja' ? 'en' : 'ja';
      try {
        localStorage.setItem('ptcg_lang', LANG);
      } catch (e) {}
      this.langBtn.textContent = LANG === 'ja' ? 'EN' : 'JP';
      if (this.state) this.render();
      if (this.zoomPi !== null) this.renderZoom();
    });

    this.deckBtn.addEventListener('click', () => this.openDeckModal());
    this.resetBtn.addEventListener('click', () => this.openDeckModal());

    this.btnCloseDeckModal.addEventListener('click', () => this.closeDeckModal());
    this.btnCancelDeck.addEventListener('click', () => this.closeDeckModal());
    this.btnStartGame.addEventListener('click', () => this.startNewBattle());

    this.btnTriggerGenerate.addEventListener('click', (e) => {
      e.stopPropagation();
      this.generateAIDeck();
    });

    this.btnConfirmMultiEl.addEventListener('click', () => this.submitMultiAction());
    this.btnClearLog.addEventListener('click', () => {
      this.logEl.innerHTML = '<span class="hint">Log cleared.</span>';
    });

    this.zoomCloseEl.addEventListener('click', () => this.closeZoom());
    this.zoomEl.addEventListener('click', (e) => {
      if (e.target === this.zoomEl) this.closeZoom();
    });

    this.btnRematch.addEventListener('click', () => {
      this.gameOverModal.style.display = 'none';
      this.startNewBattle();
    });

    this.btnNewDeck.addEventListener('click', () => {
      this.gameOverModal.style.display = 'none';
      this.openDeckModal();
    });

    // Deck option cards selection
    document.querySelectorAll('.deck-option-card').forEach((card) => {
      card.addEventListener('click', () => {
        document.querySelectorAll('.deck-option-card').forEach((c) => c.classList.remove('selected'));
        card.classList.add('selected');
        this.selectedDeckId = card.getAttribute('data-deck-id');
        this.updateDeckPreview();
      });
    });

    // Hover preview setup
    document.getElementById('wrap').addEventListener('mousemove', (e) => {
      const card = e.target.closest('[data-id]');
      if (card && card.dataset.id) {
        this.showPreview(card);
        this.movePreview(e);
      } else {
        this.hidePreview();
      }
    });
    document.getElementById('wrap').addEventListener('mouseleave', () => this.hidePreview());

    this.zoomGridEl.addEventListener('mousemove', (e) => {
      const card = e.target.closest('[data-id]');
      if (card && card.dataset.id) {
        this.showPreview(card);
        this.movePreview(e);
      } else {
        this.hidePreview();
      }
    });

    // Click discard open
    document.getElementById('wrap').addEventListener('click', (e) => {
      const z = e.target.closest('.zone-open');
      if (z) this.openZoom(+z.dataset.pi);
    });

    // Escape closes zoom
    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') this.closeZoom();
    });
  }

  // ── HOVER PREVIEW (LARGE CARD) ──────────────────────────────────────────────
  energyCardsHtml(ids) {
    if (!ids || !ids.length) return '';
    const order = [];
    const n = {};
    for (const id of ids) {
      if (!id) continue;
      if (!(id in n)) {
        n[id] = 0;
        order.push(id);
      }
      n[id]++;
    }
    return order
      .map((id) => {
        const imgs = Array.from(
          { length: n[id] },
          () => `<img src="${cardImgUrl(id)}" onerror="this.remove()">`
        ).join('');
        const cnt = `<span class="ecount">×${n[id]}</span>`;
        return `<div class="egroup"><span class="estack">${imgs}</span>${cnt}</div>`;
      })
      .join('');
  }

  showPreview(card) {
    const id = card.dataset.id;
    if (id !== this.pvId) {
      this.pvId = id;
      this.pvImg.style.display = 'block';
      this.pvImg.src = cardImgUrl(id);

      const name = card.dataset.name || `Card #${id}`;
      let cap = `<b>${name} (#${id})</b>`;
      if (card.dataset.hp != null && card.dataset.mhp != null) {
        cap += `<div class="sub">HP ${card.dataset.hp}/${card.dataset.mhp}</div>`;
      }
      if (card.dataset.desc) {
        cap += `<div class="sub" style="font-size: 11px; margin-top: 4px;">${card.dataset.desc}</div>`;
      }
      this.pvCap.innerHTML = cap;

      const encards = card.dataset.encards ? card.dataset.encards.split(',').filter(Boolean) : [];
      this.pvEnergy.innerHTML = encards.length ? this.energyCardsHtml(encards) : '';
      this.pvEnergy.style.display = encards.length ? 'flex' : 'none';

      const tools = card.dataset.tools ? card.dataset.tools.split(',').filter(Boolean) : [];
      this.pvTools.innerHTML = tools.length
        ? tools
            .map(
              (t) =>
                `<div class="ptool"><img src="${cardImgUrl(t)}" onerror="this.remove()"><div class="tname">Tool #${t}</div></div>`
            )
            .join('')
        : '';
      this.pvTools.style.display = tools.length ? 'flex' : 'none';
    }
    this.previewEl.style.display = 'block';
  }

  hidePreview() {
    if (this.pvId !== null) {
      this.previewEl.style.display = 'none';
      this.pvId = null;
    }
  }

  movePreview(e) {
    const pad = 16;
    const w = this.previewEl.offsetWidth || 260;
    const h = this.previewEl.offsetHeight || 380;
    let x = e.clientX + pad;
    let y = e.clientY + pad;
    if (x + w > window.innerWidth) x = e.clientX - pad - w;
    if (y + h > window.innerHeight) y = Math.max(6, window.innerHeight - h - 6);
    this.previewEl.style.left = x + 'px';
    this.previewEl.style.top = y + 'px';
  }

  // ── DISCARD PILE MODAL ──────────────────────────────────────────────────────
  openZoom(pi) {
    this.zoomPi = pi;
    this.renderZoom();
    this.zoomEl.classList.remove('hide');
    this.zoomEl.style.display = 'flex';
  }

  closeZoom() {
    this.zoomPi = null;
    this.zoomEl.classList.add('hide');
    this.zoomEl.style.display = 'none';
    this.hidePreview();
  }

  renderZoom() {
    if (this.zoomPi === null || !this.state) return;
    const isHuman = this.zoomPi === 0;
    const targetPlayer = isHuman ? this.state.player : this.state.opponent;
    const discard = targetPlayer?.discard || [];

    this.zoomTitleEl.textContent = `P${this.zoomPi} (${isHuman ? 'Player' : 'MuZero AI'}) — Discard Pile (${discard.length} cards)`;

    if (discard.length === 0) {
      this.zoomGridEl.innerHTML = '<span class="hint">(Discard pile is empty)</span>';
      return;
    }

    this.zoomGridEl.innerHTML = discard
      .map((c) => this.renderPlainCard(c, false))
      .join('');
  }

  // ── CARD RENDERING HELPERS ─────────────────────────────────────────────────
  energyDots(list) {
    if (!list || !list.length) return '';
    return (
      '<div class="energies">' +
      list
        .map((t) => {
          const [lbl, col] = ENERGY[t] || ['?', '#888'];
          return `<span class="en" style="background:${col}" title="${lbl}">${lbl}</span>`;
        })
        .join('') +
      '</div>'
    );
  }

  hpBar(hp, max) {
    const r = max ? Math.max(0, hp / max) : 0;
    const col = r > 0.5 ? 'var(--hpgood)' : r > 0.25 ? 'var(--hpmid)' : 'var(--hpbad)';
    return `<div class="hpbar"><i style="width:${(r * 100).toFixed(0)}%;background:${col}"></i></div>`;
  }

  renderPokemonCard(p, opts = {}) {
    if (!p) {
      return `<div class="card ${opts.active ? 'active' : ''} faceup"><div class="txt"><div class="nm">(none)</div></div></div>`;
    }
    const cls = 'card mon' + (opts.active ? ' active' : '') + ' faceup';
    const url = cardImgUrl(p.id);

    const badges = [];
    for (const s of opts.status || []) badges.push(`<span class="badge sc">${s}</span>`);
    for (const t of p.tools || []) badges.push(`<span class="badge tool" title="Tool #${t.id || t}">Tool</span>`);
    if (p.name && (p.name.toLowerCase().includes(' ex') || p.name.toLowerCase().includes(' vstar'))) {
      badges.push('<span class="badge ex">ex</span>');
    }

    const img = url
      ? `<img src="${url}" onload="this.parentElement.classList.add('hasimg')" onerror="this.remove()">`
      : '';

    const encardIds = (p.energyCards || []).map((e) => (typeof e === 'object' ? e.id : e)).filter(Boolean);
    const toolIds = (p.tools || []).map((t) => (typeof t === 'object' ? t.id : t)).filter(Boolean);

    return `
      <div class="${cls}" data-id="${p.id}" data-name="${p.name || ''}" data-hp="${p.hp}" data-mhp="${p.max_hp || p.maxHp}" data-encards="${encardIds.join(',')}" data-tools="${toolIds.join(',')}" data-desc="${p.description || ''}">
        ${img}
        <div class="badges">${badges.join('')}</div>
        <div class="txt">
          <div class="nm">${p.name || `Card #${p.id}`}</div>
          <div class="stat">
            ${this.energyDots(p.energies)}
            <div class="hp">HP ${p.hp}/${p.max_hp || p.maxHp}</div>
            ${this.hpBar(p.hp, p.max_hp || p.maxHp)}
          </div>
        </div>
      </div>
    `;
  }

  renderPlainCard(c, small = false) {
    if (!c) {
      return `<div class="card ${small ? 'small' : ''} faceup"><div class="txt"><div class="nm">(face down)</div></div></div>`;
    }
    const url = cardImgUrl(c.id);
    const img = url
      ? `<img src="${url}" onload="this.parentElement.classList.add('hasimg')" onerror="this.remove()">`
      : '';
    return `
      <div class="card ${small ? 'small' : ''} faceup" data-id="${c.id}" data-name="${c.name || ''}" data-desc="${c.description || ''}">
        ${img}
        <div class="txt"><div class="nm">${c.name || `Card #${c.id}`}</div></div>
      </div>
    `;
  }

  statusOf(player) {
    if (!player) return [];
    const res = [];
    if (player.poisoned) res.push('PSN');
    if (player.burned) res.push('BRN');
    if (player.asleep) res.push('SLP');
    if (player.paralyzed) res.push('PAR');
    if (player.confused) res.push('CNF');
    return res;
  }

  // ── DECK MANAGEMENT & API ──────────────────────────────────────────────────
  async loadDecks() {
    try {
      const res = await fetch('/api/decks');
      const data = await res.json();
      if (data.status === 'success' && data.decks) {
        this.allDecks = data.decks;
        this.selectedDeckCards = this.allDecks[0]?.cards || null;
        this.updateDeckPreview();
      }
    } catch (e) {
      console.error('Error loading decks:', e);
    }
  }

  async generateAIDeck() {
    this.btnTriggerGenerate.textContent = '⏳ REINFORCE Sampling...';
    this.btnTriggerGenerate.disabled = true;
    try {
      const res = await fetch('/api/generate_deck', { method: 'POST' });
      const data = await res.json();
      if (data.status === 'success') {
        this.selectedDeckCards = data.deck;
        this.renderDeckPreview(data.summary, '✨ RL Sampled AI Deck');
      }
    } catch (e) {
      console.error('Error generating deck:', e);
    } finally {
      this.btnTriggerGenerate.textContent = '🎲 Generate New AI Deck';
      this.btnTriggerGenerate.disabled = false;
    }
  }

  updateDeckPreview() {
    const found = this.allDecks.find((d) => d.id === this.selectedDeckId);
    if (found) {
      this.selectedDeckCards = found.cards;
      this.renderDeckPreview(found.summary, found.name);
    }
  }

  renderDeckPreview(summary, title) {
    if (!summary) return;
    this.previewTitle.textContent = title || 'Deck Composition (60 cards)';
    this.prevPokeCount.textContent = summary.pokemon_count || 0;
    this.prevTrainCount.textContent = summary.trainer_count || 0;
    this.prevNrgCount.textContent = summary.energy_count || 0;

    let html = '';
    const allCards = [
      ...(summary.pokemon || []),
      ...(summary.trainers || []),
      ...(summary.energies || []),
    ];
    allCards.forEach((c) => {
      html += `
        <div class="prev-pill" style="border-left: 3px solid ${c.type_color || '#4aa3ff'};">
          <span><b>${c.count}x</b> ${c.name}</span>
          <small style="color: var(--dim);">#${c.id}</small>
        </div>
      `;
    });
    this.deckPreviewGrid.innerHTML = html;
  }

  openDeckModal() {
    this.deckModal.style.display = 'flex';
    this.updateDeckPreview();
  }

  closeDeckModal() {
    this.deckModal.style.display = 'none';
  }

  async checkActiveBattle() {
    try {
      const res = await fetch('/api/battle/state');
      const data = await res.json();
      if (data.status === 'success' && data.state && data.state.is_started) {
        this.state = data.state;
        this.render();
      } else {
        this.openDeckModal();
      }
    } catch (e) {
      this.openDeckModal();
    }
  }

  setThinking(isThinking) {
    if (!this.thinkingOverlay) return;
    if (isThinking) {
      const isAdvanced = this.state?.ai_mode === 'advanced' || this.state?.ai_mode === 'ismcts' || (this.aiModeSelect && this.aiModeSelect.value === 'advanced');
      if (this.thinkingSubText) {
        this.thinkingSubText.textContent = isAdvanced
          ? 'ISMCTS Simulation (50 iterations & belief)'
          : 'IREE GPU Policy Inference';
      }
      this.thinkingOverlay.style.display = 'flex';
    } else {
      this.thinkingOverlay.style.display = 'none';
    }
  }

  async startNewBattle() {
    this.closeDeckModal();
    const device = this.deviceSelect ? this.deviceSelect.value : 'vulkan';
    const ai_mode = this.aiModeSelect ? this.aiModeSelect.value : 'basic';
    const deck = this.selectedDeckCards;

    this.setThinking(true);
    try {
      const res = await fetch('/api/battle/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          player_deck: deck,
          ai_deck: deck,
          device: device,
          ai_mode: ai_mode,
        }),
      });
      const data = await res.json();
      if (data.status === 'success') {
        this.state = data.state;
        this.render();
      } else {
        this.setThinking(false);
        alert(data.message || 'Error launching battle.');
      }
    } catch (e) {
      this.setThinking(false);
      console.error('Failed to start battle:', e);
      alert('Unable to reach battle server.');
    }
  }

  async submitAction(indices) {
    if (this.isSubmitting) return;
    this.isSubmitting = true;
    this.setThinking(true);
    try {
      const res = await fetch('/api/battle/step', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ selected_indices: indices }),
      });
      const data = await res.json();
      if (data.status === 'success') {
        this.state = data.state;
        this.render();
      } else {
        this.setThinking(false);
        alert(data.message || "Error executing action.");
      }
    } catch (e) {
      this.setThinking(false);
      console.error('Action error:', e);
    } finally {
      this.isSubmitting = false;
    }
  }

  submitMultiAction() {
    const indices = Array.from(this.multiSelectedIndices);
    const ctx = this.state?.select_context || {};
    if (indices.length < ctx.min_count) {
      alert(`Please select at least ${ctx.min_count} option(s).`);
      return;
    }
    this.multiSelectedIndices.clear();
    this.submitAction(indices);
  }

  // ── MAIN RENDER LOOP ───────────────────────────────────────────────────────
  render() {
    if (!this.state) return;
    const s = this.state;

    // Manage thinking overlay
    if (s.is_human_turn || s.is_done) {
      this.setThinking(false);
    } else {
      this.setThinking(true);
    }

    // Header metadata
    this.turnEl.textContent = s.turn || 0;
    const isP0 = s.your_index === 0;
    this.activeEl.textContent = s.turn === 0 ? 'Draw' : (isP0 ? 'P0 (You)' : 'P1 (MuZero AI)');
    this.activeEl.style.color = isP0 ? 'var(--p0)' : 'var(--p1)';
    this.ctxEl.textContent = s.select_context?.prompt ? `Phase #${s.select_context.id}` : 'Play';

    // Engine & AI Mode Badge
    const isAdvanced = s.ai_mode === 'advanced' || s.ai_mode === 'ismcts';
    const engineNameEl = document.getElementById('engineName');
    if (engineNameEl) {
      engineNameEl.textContent = isAdvanced ? 'MuZero ISMCTS (Advanced)' : 'IREE Vulkan (Basic)';
    }

    const hudTitleEl = document.querySelector('.ai-hud-title');
    if (hudTitleEl) {
      hudTitleEl.textContent = isAdvanced ? '🧠 MuZero ISMCTS Search' : '⚡ MuZero IREE Fast Policy';
    }


    // Stadium
    const stadiumObj = s.player?.stadium || s.opponent?.stadium || null;
    if (stadiumObj) {
      this.stadiumNameEl.textContent = stadiumObj.name || `Stadium #${stadiumObj.id}`;
      this.stadiumEl.dataset.id = stadiumObj.id;
    } else {
      this.stadiumNameEl.textContent = '(none)';
      delete this.stadiumEl.dataset.id;
    }

    // Context Banner
    this.contextPromptEl.textContent = s.select_context?.prompt || 'Waiting...';

    // MuZero HUD
    const thoughts = s.ai_thoughts || {};
    const winrate = thoughts.winrate !== undefined ? thoughts.winrate : 50.0;
    this.aiWinrateEl.innerHTML = `AI Win Estimate: <strong>${winrate.toFixed(1)}%</strong>`;
    this.aiBarFillEl.style.width = `${winrate}%`;


    const topActions = thoughts.top_actions || [];
    if (topActions.length > 0) {
      this.aiIntentRowEl.innerHTML = topActions
        .slice(0, 3)
        .map(
          (ta, idx) =>
            `<span>#${idx + 1} <b>${ta.score}%</b> (Opt ${ta.option_index})</span>`
        )
        .join(' &nbsp;•&nbsp; ');
    } else {
      this.aiIntentRowEl.innerHTML = '<span>Waiting for engine...</span>';
    }

    // Render Panels
    this.renderPlayerPanel(this.panel1, s.opponent, 'p1', s.your_index === 1 && !s.is_done);
    this.renderPlayerPanel(this.panel0, s.player, 'p0', s.your_index === 0 && !s.is_done);

    // Actions List
    this.renderActionsList(s.options || [], s.select_context || {});

    // Logs
    this.renderLogs(s.logs || []);

    // Game Over modal
    if (s.is_done && s.result >= 0) {
      this.showGameOverModal(s.result, s.game_over_reason);
    }
  }

  renderPlayerPanel(el, ps, who, isTurn) {
    if (!ps) {
      el.innerHTML = '<div class="hint">Loading board...</div>';
      return;
    }
    const isP0 = who === 'p0';
    const st = this.statusOf(ps);

    const benchHtml =
      ps.bench && ps.bench.length
        ? ps.bench.map((b) => this.renderPokemonCard(b)).join('')
        : `<span class="hint">(no bench)</span>`;

    const handHtml =
      ps.hand && ps.hand.length
        ? ps.hand.map((c) => this.renderPlainCard(c, !isP0)).join('')
        : `<span class="hint">(no cards in hand)</span>`;

    const discardLen = ps.discard ? ps.discard.length : 0;
    const discardBtn =
      discardLen > 0
        ? `<span class="zone-open" data-pi="${isP0 ? 0 : 1}" title="Click to view discard pile">Discard <b>${discardLen}</b></span>`
        : `<span>Discard <b>0</b></span>`;

    el.className = 'player ' + who + (isTurn ? ' turn' : '');
    el.innerHTML = `
      <div class="prow">
        <span class="tag ${who}">${isP0 ? 'P0 (YOU)' : 'P1 (MUZERO AI)'}</span>
        ${isTurn ? `<span class="turnflag">▶ TURN</span>` : ''}
        <div class="counts">
          <span>Deck <b>${ps.deck_count !== undefined ? ps.deck_count : 60}</b></span>
          <span>Hand <b>${ps.hand ? ps.hand.length : (ps.hand_count || 0)}</b></span>
          <span>Prizes <b>${ps.prizes_left !== undefined ? ps.prizes_left : 6}</b></span>
          ${discardBtn}
        </div>
      </div>
      <div class="prow" style="align-items:flex-end">
        <div>
          <div class="zone-label">ACTIVE SPOT</div>
          <div class="cards">${this.renderPokemonCard(ps.active, { active: true, status: st })}</div>
        </div>
        <div style="flex:1">
          <div class="zone-label">BENCH (${ps.bench ? ps.bench.length : 0}/5)</div>
          <div class="cards">${benchHtml}</div>
        </div>
      </div>
      <div class="zone-label">HAND (${ps.hand ? ps.hand.length : (ps.hand_count || 0)} cards)</div>
      <div class="cards">${handHtml}</div>
    `;
  }

  renderActionsList(options, context) {
    this.actionCountBadgeEl.textContent = options.length;
    const isMulti = context.max_count > 1;

    if (isMulti) {
      this.multiSelectControlsEl.style.display = 'block';
      this.multiSelectedIndices.clear();
    } else {
      this.multiSelectControlsEl.style.display = 'none';
    }

    if (!options || options.length === 0) {
      this.actionsListEl.innerHTML =
        '<div class="hint">No action required. AI turn or match ended.</div>';
      return;
    }

    let html = '';
    options.forEach((opt) => {
      html += `
        <div class="action-item" data-opt-idx="${opt.index}">
          <div class="action-head">
            <span class="action-title">${opt.title}</span>
            <span class="action-tag" style="background:${opt.badge_color || '#4aa3ff'};">${opt.badge}</span>
          </div>
          ${opt.subtitle ? `<span class="action-sub">${opt.subtitle}</span>` : ''}
        </div>
      `;
    });
    this.actionsListEl.innerHTML = html;

    this.actionsListEl.querySelectorAll('.action-item').forEach((item) => {
      const idx = parseInt(item.getAttribute('data-opt-idx'));
      item.addEventListener('click', () => {
        if (isMulti) {
          if (this.multiSelectedIndices.has(idx)) {
            this.multiSelectedIndices.delete(idx);
            item.classList.remove('selected');
          } else {
            if (this.multiSelectedIndices.size < context.max_count) {
              this.multiSelectedIndices.add(idx);
              item.classList.add('selected');
            }
          }
        } else {
          this.submitAction([idx]);
        }
      });
    });
  }

  renderLogs(logs) {
    if (!logs || logs.length === 0) {
      this.logEl.innerHTML = '<span class="hint">Match started...</span>';
      return;
    }
    let html = '';
    logs.forEach((log) => {
      let cls = 'log-line';
      if (log.level === 'victory') cls += ' res';
      else if (log.level === 'defeat') cls += ' hp';
      else if (log.level === 'action') cls += ' player';
      else if (log.level === 'ai') cls += ' ai';
      html += `<div class="${cls}">[T${log.turn || 0}] ${log.message}</div>`;
    });
    this.logEl.innerHTML = html;
    this.logEl.scrollTop = this.logEl.scrollHeight;
  }

  showGameOverModal(result, reason) {
    if (result === 0) {
      this.gameOverIcon.textContent = '🏆';
      this.gameOverTitle.textContent = 'VICTORY!';
      this.gameOverDesc.textContent = reason || 'You triumphed over the MuZero model.';
    } else if (result === 1) {
      this.gameOverIcon.textContent = '💀';
      this.gameOverTitle.textContent = 'DEFEAT BY AI';
      this.gameOverDesc.textContent = reason || 'MuZero AI won the match.';
    } else {
      this.gameOverIcon.textContent = '🤝';
      this.gameOverTitle.textContent = 'DRAW MATCH';
      this.gameOverDesc.textContent = reason || 'Match concluded in a draw.';
    }
    this.gameOverModal.style.display = 'flex';
  }
}

document.addEventListener('DOMContentLoaded', () => {
  window.app = new BattleVisualizerApp();
});

