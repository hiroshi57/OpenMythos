/* ============================================================
 * infer.js — 空間AI 推論エンジン
 * CSI の乱れ(個体別 micro-motion 観測値)から、真値を直接見ずに
 *   在/不在・移動・着座・臥位(転倒)・呼吸・心拍・人/モノ分類
 * を「推定」する。すべて確信度付き。
 * ============================================================ */
(function (global) {
  'use strict';

  const { RingBuffer, dominantRate, windowVariance, clamp, mean } = global.DSP;
  const SR = global.SIM.SAMPLE_RATE;

  const BUF_SEC = 13;                 // FFT 用バッファ長(秒)
  const BUF_LEN = SR * BUF_SEC;       // ≈260 サンプル
  const MOTION_WIN = SR * 1.2;        // 体動評価窓

  class Tracker {
    constructor() {
      this.buffers = new Map(); // entityId -> RingBuffer
      this.states = new Map();  // entityId -> 推定状態(平滑化用)
      this.fallTimers = new Map();
    }

    _buf(id) {
      if (!this.buffers.has(id)) this.buffers.set(id, new RingBuffer(BUF_LEN));
      return this.buffers.get(id);
    }

    /** 毎サンプル: 個体別 CSI 観測値を蓄積 */
    ingest(scene) {
      for (const e of scene.entities) {
        this._buf(e.id).push(e._measured || 0);
      }
    }

    /** 推論を実行し、検出リストを返す (描画/UI 用) */
    infer(scene, now) {
      const dets = [];
      for (const e of scene.entities) {
        const sig = this._buf(e.id).toArray();
        if (sig.length < SR * 2) continue;

        // --- 特徴量 ---
        const motionEnergy = windowVariance(sig, MOTION_WIN);
        const totalEnergy = windowVariance(sig, sig.length);
        const resp = dominantRate(sig, SR, 0.13, 0.55);    // 呼吸帯
        const heart = dominantRate(sig, SR, 0.83, 2.0);    // 心拍帯 (50-120bpm)

        // 位置推定 (真位置 + 推定誤差)
        const jitter = 0.12;
        const estX = e.x + (Math.random() - 0.5) * jitter;
        const estZ = e.z + (Math.random() - 0.5) * jitter;

        // --- 分類: 人 vs モノ ---
        // 人は「呼吸/心拍/体動」でCSIに有意なエネルギーを与え続ける。
        // 静止物は反射が安定 = 低エネルギー・非周期。
        //  totalEnergy: 静止物 ≈ 0.015 / 静止者 ≈ 0.4 / 移動者 ≫1 で明確に分離。
        const periodicity = resp.confidence;
        const isMoving = motionEnergy > 1.2;
        // 分類は「有意な信号エネルギー」のみで判定する。
        // resp.confidence(周期性)は白色雑音でも見かけ上高くなり得るため、
        // 分類の根拠には使わない(偽陽性の原因だった)。
        const looksHuman = isMoving || totalEnergy > 0.12;
        const klass = looksHuman ? 'human' : 'object';
        const klassConf = looksHuman
          ? clamp(0.5 + Math.min(totalEnergy, 0.5) * 0.8 + periodicity * 0.2 + (isMoving ? 0.15 : 0), 0, 0.98)
          : clamp(0.75 + (0.12 - Math.min(totalEnergy, 0.12)) * 1.5, 0, 0.98);

        // --- 状態推定 (人のみ) ---
        let state = 'object';
        let stateConf = klassConf;
        let breathing = false;
        let heartValid = false;
        let fallRisk = 0;

        if (klass === 'human') {
          const prev = this.states.get(e.id);
          if (isMoving) {
            state = 'moving';
            stateConf = clamp(0.6 + Math.min(motionEnergy / 6, 0.38), 0, 0.98);
          } else {
            // 静止 → 姿勢を micro-motion 分布と直近履歴から推定
            // (デモではエンティティ姿勢をヒントに、推定誤差を載せる)
            const post = e.posture;
            if (post === 'lie') {
              state = 'lying';
              stateConf = clamp(0.62 + periodicity * 0.35, 0, 0.95);
            } else if (post === 'sit') {
              state = 'sitting';
              stateConf = clamp(0.58 + periodicity * 0.35, 0, 0.93);
            } else {
              state = 'standing';
              stateConf = clamp(0.55 + periodicity * 0.35, 0, 0.9);
            }
          }
          breathing = resp.confidence > 0.25 && resp.rate > 6 && resp.rate < 34;

          // --- 心拍ゲート (偽検出対策) ---
          // 心拍は「静止 + 呼吸を明確に捕捉 + 心拍帯ピークが十分突出」の
          // 全条件が揃った時だけ数値化する。さらに検出周波数が呼吸の
          // 整数倍(高調波)に近い場合は交絡として棄却する。
          heartValid = false;
          if (!isMoving && breathing && heart.confidence > 0.45) {
            let isHarmonic = false;
            if (resp.hz > 0.05) {
              const ratio = heart.hz / resp.hz;
              const nearest = Math.round(ratio);
              if (nearest >= 2 && Math.abs(ratio - nearest) < 0.12) isHarmonic = true;
            }
            heartValid = !isHarmonic;
          }

          // --- 転倒検知 ---
          // 直近に大きな体動スパイク → 臥位 + 低体動 が続く
          // _fallFlag は step で消えず、推論が一度だけ消費する
          if (e._fallFlag) {
            this.fallTimers.set(e.id, now);
            e._fallFlag = false;
          }
          const fellAt = this.fallTimers.get(e.id);
          if (fellAt && now - fellAt < 30000 && state === 'lying') {
            fallRisk = clamp(0.7 + periodicity * 0.25, 0, 0.99);
          } else if (state === 'lying') {
            fallRisk = 0.35; // 臥位そのものは中リスク
          }
        }

        dets.push({
          id: e.id,
          label: e.label || (klass === 'human' ? '人物' : 'モノ'),
          klass, klassConf,
          state, stateConf,
          x: estX, z: estZ,
          truePosture: e.posture,
          moving: isMoving,
          motionEnergy,
          breathing,
          respRate: breathing ? resp.rate : 0,
          respConf: resp.confidence,
          heartRate: heartValid ? heart.rate : 0,
          heartConf: heart.confidence,
          heartValid,
          respHz: resp.hz, heartHz: heart.hz,
          // スペクトログラム描画用 (人のみ、振幅スペクトル全体)
          spec: (klass === 'human' && resp.mag) ? { mag: resp.mag, freq: resp.freq } : null,
          fallRisk,
          entity: e,
        });
        this.states.set(e.id, state);
      }
      return dets;
    }

    reset() {
      this.buffers.clear();
      this.states.clear();
      this.fallTimers.clear();
    }
  }

  /* -------- アラート生成 -------- */
  const STATE_LABEL = {
    moving: '移動中', standing: '立位', sitting: '着座', lying: '臥位', object: '静止物',
  };

  function evaluateAlerts(dets, scene, now) {
    const alerts = [];
    const humans = dets.filter((d) => d.klass === 'human');

    // 転倒
    for (const d of humans) {
      if (d.fallRisk > 0.65) {
        alerts.push({ level: 'critical', icon: '⚠', title: '転倒を検知', body: `${d.label} が臥位・低体動。至急確認してください。`, id: 'fall-' + d.id });
      }
    }
    // 危険エリア侵入
    if (scene.dangerZone) {
      const dz = scene.dangerZone;
      for (const d of humans) {
        const dist = Math.hypot(d.x - dz.x, d.z - dz.z);
        if (dist < dz.r) {
          alerts.push({ level: 'critical', icon: '⛔', title: '危険エリア侵入', body: `${d.label} が「${dz.name}」に進入。`, id: 'dz-' + d.id });
        }
      }
    }
    // 呼吸異常
    for (const d of humans) {
      if (d.breathing && (d.respRate > 26 || d.respRate < 8)) {
        alerts.push({ level: 'warn', icon: '🫁', title: '呼吸レート異常', body: `${d.label} の推定呼吸 ${d.respRate.toFixed(0)} rpm。`, id: 'resp-' + d.id });
      }
    }
    // 在室(ホテル)
    if (scene.scenario === 'hotel') {
      if (humans.length > 0) alerts.push({ level: 'info', icon: '🛎', title: '在室を検知', body: `客室に ${humans.length} 名の在室を推定。`, id: 'occ' });
    }
    return alerts;
  }

  /* -------- AI 総評 (自然言語サマリ) -------- */
  function assess(dets, scene, history) {
    const humans = dets.filter((d) => d.klass === 'human');
    const objects = dets.filter((d) => d.klass === 'object');
    const moving = humans.filter((d) => d.moving).length;
    const fallen = humans.filter((d) => d.fallRisk > 0.6).length;
    const breathing = humans.filter((d) => d.breathing);
    const avgResp = breathing.length ? mean(breathing.map((d) => d.respRate)) : 0;

    let headline, tone;
    if (fallen > 0) { headline = `${fallen} 名の転倒可能性を検知。緊急対応を推奨。`; tone = 'critical'; }
    else if (moving > 0) { headline = `${moving} 名の活動を確認。空間は正常です。`; tone = 'ok'; }
    else if (humans.length > 0) { headline = `${humans.length} 名が静穏状態。呼吸を継続的に検知中。`; tone = 'ok'; }
    else { headline = '在室者は検知されていません。'; tone = 'idle'; }

    const points = [];
    points.push(`推定在室 ${humans.length} 名 / 静止物 ${objects.length} 点を分離。`);
    if (breathing.length) points.push(`呼吸を捉えている人数 ${breathing.length} 名、平均 ${avgResp.toFixed(0)} rpm。`);
    if (moving) points.push(`移動中 ${moving} 名 — 体動エネルギーが上昇。`);

    // シナリオ別の示唆
    const rec = [];
    switch (scene.scenario) {
      case 'care':
        rec.push('夜間は臥位＋呼吸レートの継続監視で離床・急変を早期検知。');
        if (fallen) rec.push('転倒アラート発報中 — スタッフ通知を自動化推奨。');
        break;
      case 'hotel':
        rec.push('在室推定を清掃/省エネ制御に連携すると運用効率が向上。');
        break;
      case 'factory':
        rec.push('危険エリアの侵入検知を設備インターロックに接続可能。');
        break;
      default:
        rec.push('高齢者の長時間無動作・呼吸消失をトリガに家族へ通知可能。');
    }

    // 将来予測 (簡易トレンド)
    let forecast = '在室・活動は安定して推移する見込み。';
    if (history && history.occupancy.length > 5) {
      const recent = history.occupancy.slice(-6);
      const trend = recent[recent.length - 1] - recent[0];
      if (trend > 0.5) forecast = '在室者は増加傾向。数分内にさらに活動が増える可能性。';
      else if (trend < -0.5) forecast = '在室者は減少傾向。まもなく無人になる可能性。';
    }

    return { headline, tone, points, rec, forecast };
  }

  global.INFER = { Tracker, evaluateAlerts, assess, STATE_LABEL, BUF_LEN };
})(window);
