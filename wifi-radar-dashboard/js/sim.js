/* ============================================================
 * sim.js — Wi-Fi CSI 物理シミュレーション
 * 実機の代わりに、部屋・人・モノが Wi-Fi マルチパスをどう乱すかを
 * 物理的にそれらしくモデル化して CSI 時系列を生成する。
 *
 * モデルの考え方 (簡略):
 *  - AP(送信) と 数本の仮想アンテナ(受信) の間の各サブキャリアで
 *    チャネル応答(CSI)が観測される。
 *  - 各エンティティは自分の反射寄与を持ち、微動(呼吸・心拍・体動)で
 *    位相/振幅を時間変調する。静止物は微動ゼロ。
 *  - 観測 CSI = ベースライン + Σ(反射寄与) + 熱雑音。
 * ============================================================ */
(function (global) {
  'use strict';

  const SAMPLE_RATE = 20;   // Hz (Wi-Fi センシングのパケットレート相当)
  const N_SUBCARRIER = 30;  // 可視化用サブキャリア数
  const ROOM = { w: 6.0, d: 5.0, h: 2.6 }; // メートル
  const AP = { x: 0.4, z: 0.4 };            // アクセスポイント位置(床座標)

  const POSTURE = { STAND: 'stand', SIT: 'sit', LIE: 'lie' };

  let uid = 1;

  function rand(a, b) { return a + Math.random() * (b - a); }
  function gauss() { // Box-Muller
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /* -------- エンティティ -------- */
  class Entity {
    constructor(cfg) {
      Object.assign(this, {
        id: uid++,
        type: 'human',       // 'human' | 'object'
        x: rand(1, ROOM.w - 1),
        z: rand(1, ROOM.d - 1),
        posture: POSTURE.STAND,
        // 生体パラメータ (真値。推論側はこれを直接見ない)
        breathRate: rand(12, 18),  // rpm
        heartRate: rand(58, 78),   // bpm
        // 運動状態
        moving: false,
        speed: 0,
        vx: 0, vz: 0,
        targetX: 0, targetZ: 0,
        // 反射強度 (体格/材質)
        rcs: 1.0,
        label: '',
        // 内部位相
        _bp: Math.random() * Math.PI * 2,
        _hp: Math.random() * Math.PI * 2,
        // 転倒フラグ
        justFell: false,
        fallenAt: 0,
      }, cfg);
      this.targetX = this.x;
      this.targetZ = this.z;
    }

    /** dt 秒進める。micro-motion の瞬時値 m(t) を返す */
    step(dt, t) {
      this.justFell = false;
      if (this.type === 'object') {
        // 静止物: 微動ほぼゼロ、わずかな環境揺らぎのみ (周期性なし)
        return 0.015 * gauss();
      }

      // --- 移動ロジック ---
      if (this.moving) {
        const dx = this.targetX - this.x;
        const dz = this.targetZ - this.z;
        const dist = Math.hypot(dx, dz);
        if (dist < 0.15) {
          // 到着 → 一定確率で停止 or 次の目標
          if (Math.random() < 0.5) {
            this.moving = false;
            this.speed = 0;
            // 停止時にたまに着座
            if (Math.random() < 0.4) this.posture = POSTURE.SIT;
          } else {
            this.pickTarget();
          }
        } else {
          this.speed = 1.1; // m/s 歩行
          this.vx = (dx / dist) * this.speed;
          this.vz = (dz / dist) * this.speed;
          this.x += this.vx * dt;
          this.z += this.vz * dt;
          this.posture = POSTURE.STAND;
        }
      }

      // --- micro-motion 合成 ---
      const fB = this.breathRate / 60;
      const fH = this.heartRate / 60;
      this._bp += 2 * Math.PI * fB * dt;
      this._hp += 2 * Math.PI * fH * dt;
      // 呼吸: 胸郭の動きは純正弦ではない。2〜3倍高調波を含めることで
      // 現実同様に高調波が心拍帯へ漏れ込む(既知の交絡)を再現する。
      let breath = 0.9 * (Math.sin(this._bp)
        + 0.22 * Math.sin(2 * this._bp + 0.6)
        + 0.11 * Math.sin(3 * this._bp + 1.1));
      // 心拍: 微小変調。呼吸高調波と競合するため実測同様に検出が難しい
      // (呼吸の ~1/4 の振幅 → 静止時に約半数のフレームでのみ確信を持って検出)。
      let heart = 0.24 * Math.sin(this._hp);

      // 姿勢で呼吸振幅が変わる (臥位は浅い場合あり)
      if (this.posture === POSTURE.LIE) breath *= 0.7;

      let motion = 0;
      if (this.moving) {
        // 体動が微動を大きく上回る
        motion = 3.5 + 1.5 * Math.sin(t * 6 + this.id) + 0.8 * gauss();
      }
      return breath + heart + motion;
    }

    pickTarget() {
      this.targetX = rand(0.8, ROOM.w - 0.8);
      this.targetZ = rand(0.8, ROOM.d - 0.8);
      this.moving = true;
    }

    fall() {
      this.moving = false;
      this.posture = POSTURE.LIE;
      this.speed = 0;
      this.justFell = true;
      this._fallFlag = true; // 推論が消費するまで持続 (step で消えない)
      this.breathRate = rand(20, 30); // 転倒直後は呼吸が乱れる
    }

    distToAP() {
      return Math.hypot(this.x - AP.x, this.z - AP.z);
    }
    reflectionWeight() {
      // 距離減衰 (経路損失近似) × RCS
      const d = this.distToAP() + 0.6;
      return (this.rcs / (d * d)) * 6.0;
    }
  }

  /* -------- シーン全体 -------- */
  class Scene {
    constructor() {
      this.t = 0;
      this.entities = [];
      this.subcarrierPhase = new Float32Array(N_SUBCARRIER);
      for (let i = 0; i < N_SUBCARRIER; i++) this.subcarrierPhase[i] = rand(0, Math.PI * 2);
      this.csi = new Float32Array(N_SUBCARRIER); // 直近サンプル
      this.aggregate = 0;
      this.scenario = 'home';
      this.dangerZone = null;
    }

    loadScenario(name) {
      this.scenario = name;
      this.entities = [];
      uid = 1;
      const P = SIM_POSTURE;
      const addHuman = (x, z, opt) => this.entities.push(new Entity(Object.assign({ type: 'human', x, z }, opt || {})));
      const addObj = (x, z, rcs, label) => this.entities.push(new Entity({ type: 'object', x, z, rcs: rcs || 1.2, label: label || 'モノ', breathRate: 0, heartRate: 0 }));

      switch (name) {
        case 'care': // 介護施設: ベッド上の高齢者 + 見守り
          addObj(4.6, 1.2, 1.8, 'ベッド');
          addObj(1.2, 4.2, 1.0, '椅子');
          addHuman(4.6, 1.4, { posture: P.LIE, breathRate: rand(14, 20), heartRate: rand(62, 80), label: '入居者A' });
          addHuman(2.4, 3.0, { moving: false, label: 'スタッフ' });
          this.dangerZone = null;
          break;
        case 'hotel': // ホテル: 在室確認
          addObj(4.8, 3.8, 1.6, 'ベッド');
          addObj(1.0, 1.0, 1.0, '机');
          addHuman(3.2, 2.6, { moving: false, posture: P.SIT, label: '宿泊者' });
          this.dangerZone = null;
          break;
        case 'factory': // 工場: 危険エリア侵入検知
          addObj(3.0, 1.0, 2.2, 'プレス機');
          addObj(5.0, 4.0, 1.4, '棚');
          addHuman(1.4, 3.6, { label: '作業員A' });
          addHuman(4.4, 2.4, { label: '作業員B' });
          this.dangerZone = { x: 3.0, z: 1.0, r: 1.3, name: '危険エリア' };
          break;
        case 'home': // 家庭: 高齢者見守り
        default:
          addObj(1.2, 1.0, 1.4, 'ソファ');
          addObj(4.8, 4.4, 1.0, 'テーブル');
          addHuman(2.6, 2.4, { label: '居住者' });
          this.dangerZone = null;
          break;
      }
      // 人は最初少し動く
      this.entities.forEach((e) => { if (e.type === 'human' && Math.random() < 0.5) e.pickTarget(); });
      this.t = 0;
    }

    /** 外部トリガ: ランダムな人を歩かせる */
    triggerWalk() {
      const humans = this.entities.filter((e) => e.type === 'human');
      if (humans.length) humans[Math.floor(Math.random() * humans.length)].pickTarget();
    }
    /** 外部トリガ: ランダムな人を転倒させる */
    triggerFall() {
      const humans = this.entities.filter((e) => e.type === 'human' && e.posture !== SIM_POSTURE.LIE);
      const cand = humans.length ? humans : this.entities.filter((e) => e.type === 'human');
      if (cand.length) {
        const e = cand[Math.floor(Math.random() * cand.length)];
        e.fall();
        return e;
      }
      return null;
    }
    addPerson() {
      const e = new Entity({ type: 'human', label: '人物' + (this.entities.filter(x=>x.type==='human').length + 1) });
      this.entities.push(e);
      if (Math.random() < 0.6) e.pickTarget();
    }

    step(dt) {
      this.t += dt;
      // 各エンティティの micro-motion 合成 → 反射寄与
      let agg = 0;
      const contributions = [];
      for (const e of this.entities) {
        const m = e.step(dt, this.t);
        const w = e.reflectionWeight();
        const contrib = w * (m * 0.15); // 変調成分
        agg += contrib;
        contributions.push({ e, contrib, m });
        // 個体別 micro-motion を "観測値" として保存 (推論用, 雑音付き)
        e._measured = m * (0.9 + 0.2 * Math.random()) + 0.13 * gauss();
      }
      // 熱雑音
      agg += 0.15 * gauss();
      this.aggregate = agg;

      // サブキャリアごとに位相を変えて可視化用 CSI を生成
      for (let i = 0; i < N_SUBCARRIER; i++) {
        const ph = this.subcarrierPhase[i];
        this.csi[i] = 1.0 + agg * (0.6 + 0.4 * Math.sin(ph)) + 0.08 * gauss();
      }
      return { aggregate: agg, contributions };
    }
  }

  const SIM_POSTURE = POSTURE;

  global.SIM = {
    Scene, Entity, POSTURE,
    SAMPLE_RATE, N_SUBCARRIER, ROOM, AP,
  };
})(window);
