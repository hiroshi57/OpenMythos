/* ============================================================
 * app.js — オーケストレーション
 * シミュレーション → DSP/推論 → 描画/UI を毎フレーム回す。
 * ============================================================ */
(function () {
  'use strict';

  const { Scene } = SIM;
  const { Tracker, evaluateAlerts, assess, STATE_LABEL } = INFER;
  const { StageView, SignalView, Chart, SpectrogramView } = RENDER;

  const scene = new Scene();
  const tracker = new Tracker();

  // ---- DOM ----
  const $ = (id) => document.getElementById(id);
  const stage = new StageView($('stage'));
  const signal = new SignalView($('signalCanvas'));
  const occChart = new Chart($('occChart'), { color: '#7cff9a', min: 0 });
  const motionChart = new Chart($('motionChart'), { color: '#ffd166', min: 0 });
  const respChart = new Chart($('respChart'), { color: '#5ad1ff' });
  const spectro = new SpectrogramView($('spectroCanvas'), { maxF: 2.5 });
  let spectroMarkers = null;

  // ---- 状態 ----
  const history = {
    occupancy: [],
    motion: [],
    resp: [],
  };
  const csiHistory = []; // number[][]
  const CSI_HIST_LEN = 240;
  let lastInfer = 0;
  let lastDets = [];
  let lastAlerts = [];
  const alertLog = [];
  let running = true;
  let simSpeed = 1;

  scene.loadScenario('care');
  tracker.reset();

  // ---- シナリオボタン ----
  document.querySelectorAll('.scenario-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.scenario-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      scene.loadScenario(btn.dataset.scenario);
      tracker.reset();
      history.occupancy.length = 0; history.motion.length = 0; history.resp.length = 0;
      csiHistory.length = 0;
      alertLog.length = 0;
      renderAlertLog();
      $('scenarioDesc').textContent = SCENARIO_DESC[btn.dataset.scenario];
    });
  });

  const SCENARIO_DESC = {
    care: '介護施設 — 居室のベッド上入居者を臥位・呼吸で見守り、転倒・離床を検知。',
    hotel: 'ホテル — カメラ無しで客室の在室を推定し、清掃・省エネ制御に連携。',
    factory: '工場 — 危険エリアへの人の侵入をリアルタイム検知し設備停止に接続。',
    home: '家庭 — 独居高齢者の活動・呼吸・無動作をプライバシーを守って見守り。',
  };

  // ---- 操作ボタン ----
  $('btnWalk').addEventListener('click', () => scene.triggerWalk());
  $('btnFall').addEventListener('click', () => {
    const e = scene.triggerFall();
    if (e) pushAlert({ level: 'critical', icon: '⚠', title: '転倒イベント発生', body: `${e.label} が転倒。推論エンジンが検知中…` });
  });
  $('btnAdd').addEventListener('click', () => scene.addPerson());
  $('btnPause').addEventListener('click', () => {
    running = !running;
    $('btnPause').textContent = running ? '⏸ 一時停止' : '▶ 再開';
  });
  $('speed').addEventListener('input', (e) => {
    simSpeed = parseFloat(e.target.value);
    $('speedVal').textContent = simSpeed.toFixed(1) + '×';
  });

  // ---- アラート ----
  function pushAlert(a) {
    a.time = new Date();
    alertLog.unshift(a);
    if (alertLog.length > 30) alertLog.pop();
    renderAlertLog();
  }
  const seenAlert = new Map();
  function renderAlertLog() {
    const el = $('alertLog');
    if (!alertLog.length) {
      el.innerHTML = '<div class="empty">アラートはありません</div>';
      return;
    }
    el.innerHTML = alertLog.map((a) => `
      <div class="alert-item ${a.level}">
        <div class="ai-icon">${a.icon}</div>
        <div class="ai-body">
          <div class="ai-title">${a.title}</div>
          <div class="ai-text">${a.body}</div>
          <div class="ai-time">${a.time.toLocaleTimeString('ja-JP')}</div>
        </div>
      </div>`).join('');
  }

  // ---- メインループ ----
  let last = performance.now();
  function loop(now) {
    const rawDt = Math.min((now - last) / 1000, 0.1);
    last = now;
    const dt = rawDt * simSpeed;

    if (running) {
      // 複数サブステップで CSI サンプルを蓄積 (20Hz 相当)
      const subDt = 1 / SIM.SAMPLE_RATE;
      let acc = dt;
      let steps = 0;
      while (acc > 0 && steps < 8) {
        scene.step(subDt);
        tracker.ingest(scene);
        csiHistory.push(Array.from(scene.csi));
        if (csiHistory.length > CSI_HIST_LEN) csiHistory.shift();
        acc -= subDt;
        steps++;
      }
      // 波紋
      if (Math.random() < 0.06 * simSpeed) stage.addRipple();
    }

    // 推論は 250ms ごと
    if (now - lastInfer > 250) {
      lastInfer = now;
      lastDets = tracker.infer(scene, now);
      lastAlerts = evaluateAlerts(lastDets, scene, now);
      // 新規アラートをログへ
      for (const a of lastAlerts) {
        const prev = seenAlert.get(a.id) || 0;
        if (now - prev > 6000) { pushAlert(a); }
        seenAlert.set(a.id, now);
      }
      updatePanels();
    }

    stage.draw(lastDets, scene, rawDt);
    signal.draw(csiHistory);
    spectro.draw(spectroMarkers);
    requestAnimationFrame(loop);
  }

  function updatePanels() {
    const dets = lastDets;
    const humans = dets.filter((d) => d.klass === 'human');
    const objects = dets.filter((d) => d.klass === 'object');
    const moving = humans.filter((d) => d.moving).length;
    const breathing = humans.filter((d) => d.breathing);

    // KPI
    $('kpiPresence').textContent = humans.length;
    $('kpiObjects').textContent = objects.length;
    $('kpiMoving').textContent = moving;
    const avgResp = breathing.length ? DSP.mean(breathing.map((d) => d.respRate)) : 0;
    $('kpiResp').textContent = avgResp ? avgResp.toFixed(0) : '—';

    // 履歴
    const totalMotion = DSP.mean(humans.map((d) => d.motionEnergy)) || 0;
    history.occupancy.push(humans.length);
    history.motion.push(totalMotion);
    history.resp.push(avgResp || null);
    for (const k of ['occupancy', 'motion', 'resp']) {
      if (history[k].length > 80) history[k].shift();
    }
    occChart.draw(history.occupancy, { max: Math.max(4, ...history.occupancy) });
    motionChart.draw(history.motion);
    respChart.draw(history.resp.map((v) => v || 0), { min: 0, max: 30 });

    // スペクトログラム: フォーカス対象 = 心拍検出中 > 呼吸検出中 > 静止者 の優先
    const focus = humans.find((d) => d.heartValid)
      || breathing[0]
      || humans.find((d) => !d.moving)
      || null;
    if (focus && focus.spec) {
      spectro.push(focus.spec.mag, focus.spec.freq);
      spectroMarkers = { respHz: focus.breathing ? focus.respHz : 0, heartHz: focus.heartHz, heartValid: focus.heartValid };
      const hr = focus.heartValid ? `<b>${focus.heartRate.toFixed(0)} bpm</b> を検出` : '心拍は信号不足で未確定';
      $('spectroFocus').innerHTML = `対象: <b>${focus.label}</b> · 呼吸 ${focus.breathing ? focus.respRate.toFixed(0) + 'rpm' : '—'} · ${hr}`;
    } else {
      spectroMarkers = null;
      $('spectroFocus').textContent = '対象: 静止者なし（移動中は微動が体動に埋もれ推定不可）';
    }

    // 人物カード
    renderPeople(humans);
    // バイタル
    renderVitals(breathing, humans);
    // 総評
    renderAssessment(assess(dets, scene, history));
  }

  function confBar(v, color) {
    return `<div class="bar"><span style="width:${(v * 100).toFixed(0)}%;background:${color}"></span></div>`;
  }

  function renderPeople(humans) {
    const el = $('peopleList');
    if (!humans.length) { el.innerHTML = '<div class="empty">在室者を検知していません</div>'; return; }
    el.innerHTML = humans.map((d) => {
      const st = STATE_LABEL[d.state] || '—';
      const stColor = d.fallRisk > 0.6 ? '#ff6b5e' : d.moving ? '#7cff9a' : '#5ad1ff';
      return `
      <div class="person ${d.fallRisk > 0.6 ? 'danger' : ''}">
        <div class="p-head">
          <span class="p-name">${d.label}</span>
          <span class="p-state" style="color:${stColor}">${st}</span>
        </div>
        <div class="p-row"><span>状態確信度</span>${confBar(d.stateConf, stColor)}<b>${(d.stateConf*100).toFixed(0)}%</b></div>
        <div class="p-row"><span>人型分類</span>${confBar(d.klassConf, '#7cff9a')}<b>${(d.klassConf*100).toFixed(0)}%</b></div>
        <div class="p-row"><span>体動</span>${confBar(Math.min(d.motionEnergy/6,1), '#ffd166')}<b>${d.motionEnergy.toFixed(1)}</b></div>
        ${d.fallRisk > 0.4 ? `<div class="p-row"><span>転倒リスク</span>${confBar(d.fallRisk,'#ff6b5e')}<b>${(d.fallRisk*100).toFixed(0)}%</b></div>` : ''}
      </div>`;
    }).join('');
  }

  function renderVitals(breathing, humans) {
    const el = $('vitalsList');
    if (!breathing.length) {
      el.innerHTML = '<div class="empty">呼吸・心拍を検知できる静止者がいません<br><small>移動中は微動が体動に埋もれ推定困難</small></div>';
      return;
    }
    el.innerHTML = breathing.map((d) => {
      const hr = d.heartValid ? d.heartRate.toFixed(0) : '—';
      const hrConf = d.heartValid ? (d.heartConf * 100).toFixed(0) + '%' : '信号不足・未確定';
      return `
      <div class="vital">
        <div class="v-name">${d.label}</div>
        <div class="v-metrics">
          <div class="v-metric breath">
            <div class="v-icon">🫁</div>
            <div><div class="v-val">${d.respRate.toFixed(0)}<small>rpm</small></div>
            <div class="v-label">呼吸レート · 確信度 ${(d.respConf*100).toFixed(0)}%</div></div>
          </div>
          <div class="v-metric heart">
            <div class="v-icon">💓</div>
            <div><div class="v-val">${hr}<small>bpm</small></div>
            <div class="v-label">心拍(推定) · ${hrConf}</div></div>
          </div>
        </div>
      </div>`;
    }).join('');
  }

  function renderAssessment(a) {
    const el = $('assessment');
    const toneColor = { critical: '#ff6b5e', ok: '#7cff9a', idle: '#8aa', };
    el.innerHTML = `
      <div class="as-headline" style="border-color:${toneColor[a.tone] || '#5ad1ff'}">
        <span class="as-dot" style="background:${toneColor[a.tone] || '#5ad1ff'}"></span>${a.headline}
      </div>
      <ul class="as-points">${a.points.map((p) => `<li>${p}</li>`).join('')}</ul>
      <div class="as-sub">📈 予測</div>
      <div class="as-forecast">${a.forecast}</div>
      <div class="as-sub">💡 示唆</div>
      <ul class="as-rec">${a.rec.map((r) => `<li>${r}</li>`).join('')}</ul>`;
  }

  $('scenarioDesc').textContent = SCENARIO_DESC.care;
  document.querySelector('.scenario-btn[data-scenario="care"]').classList.add('active');
  renderAlertLog();
  requestAnimationFrame(loop);
})();
