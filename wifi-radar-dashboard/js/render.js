/* ============================================================
 * render.js — Canvas 描画
 *  - StageView: アイソメトリックのドーム + グリッド床 + 人/モノ + レーダー波紋
 *  - SignalView: CSI 波形 / 呼吸波形 / スペクトル
 *  - Chart: 汎用時系列スパークライン
 * リファレンス画像(青メッシュドーム・緑グリッド床・発光する人・橙のモノ)の世界観。
 * ============================================================ */
(function (global) {
  'use strict';

  const { ROOM, AP } = global.SIM;

  /* ---------- アイソメトリック投影 ---------- */
  function makeIso(cw, ch) {
    const scale = Math.min(cw / (ROOM.w + ROOM.d) / 1.15, ch / (ROOM.w + ROOM.d) / 1.15) * 1.4;
    const ox = cw / 2;
    const oy = ch * 0.60;
    const A = Math.cos(Math.PI / 6), B = Math.sin(Math.PI / 6);
    return function project(x, z, y) {
      y = y || 0;
      // 部屋中心を原点に
      const cx = x - ROOM.w / 2;
      const cz = z - ROOM.d / 2;
      const sx = ox + (cx - cz) * A * scale;
      const sy = oy + (cx + cz) * B * scale - y * scale;
      return { x: sx, y: sy, scale };
    };
  }

  class StageView {
    constructor(canvas) {
      this.c = canvas;
      this.ctx = canvas.getContext('2d');
      this.ripples = [];   // レーダー波紋
      this.sweep = 0;
      this.tiles = [];     // 床タイルの明滅
      this._initTiles();
      this.resize();
      window.addEventListener('resize', () => this.resize());
    }
    resize() {
      const r = this.c.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      this.c.width = r.width * dpr;
      this.c.height = r.height * dpr;
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.w = r.width; this.h = r.height;
      this.project = makeIso(this.w, this.h);
    }
    _initTiles() {
      this.gx = 12; this.gz = 10;
    }
    addRipple() { this.ripples.push({ r: 0, a: 1 }); }

    draw(dets, scene, dt) {
      const ctx = this.ctx, P = this.project;
      ctx.clearRect(0, 0, this.w, this.h);

      // 背景グラデ
      const g = ctx.createRadialGradient(this.w / 2, this.h * 0.5, 20, this.w / 2, this.h * 0.5, this.w * 0.8);
      g.addColorStop(0, '#0a1730');
      g.addColorStop(1, '#03060f');
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, this.w, this.h);

      this._drawFloor(ctx, P, scene);
      this._drawDanger(ctx, P, scene);
      this._drawRipples(ctx, P, dt);
      this._drawDome(ctx, P);
      this._drawAP(ctx, P);

      // 奥行きソートして描画
      const sorted = dets.slice().sort((a, b) => (a.x + a.z) - (b.x + b.z));
      for (const d of sorted) this._drawEntity(ctx, P, d);

      this._drawSweep(ctx, P, dt);
    }

    _drawFloor(ctx, P, scene) {
      const w = ROOM.w, d = ROOM.d;
      // グリッド床
      ctx.lineWidth = 1;
      for (let i = 0; i <= this.gx; i++) {
        const x = (i / this.gx) * w;
        const a = P(x, 0), b = P(x, d);
        ctx.strokeStyle = 'rgba(40,120,90,0.25)';
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
      for (let j = 0; j <= this.gz; j++) {
        const z = (j / this.gz) * d;
        const a = P(0, z), b = P(w, z);
        ctx.strokeStyle = 'rgba(40,120,90,0.25)';
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
      // 明滅する緑タイル (スキャン波)
      const t = performance.now() / 1000;
      for (let i = 0; i < this.gx; i++) {
        for (let j = 0; j < this.gz; j++) {
          const phase = (i + j) * 0.5 - t * 2.2;
          const lit = Math.sin(phase);
          if (lit > 0.6) {
            const x0 = (i / this.gx) * ROOM.w, z0 = (j / this.gz) * ROOM.d;
            const x1 = ((i + 1) / this.gx) * ROOM.w, z1 = ((j + 1) / this.gz) * ROOM.d;
            const p0 = P(x0 + 0.06, z0 + 0.06), p1 = P(x1 - 0.06, z0 + 0.06),
                  p2 = P(x1 - 0.06, z1 - 0.06), p3 = P(x0 + 0.06, z1 - 0.06);
            ctx.fillStyle = `rgba(80,240,140,${(lit - 0.6) * 0.9})`;
            ctx.beginPath();
            ctx.moveTo(p0.x, p0.y); ctx.lineTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y); ctx.lineTo(p3.x, p3.y); ctx.closePath();
            ctx.fill();
          }
        }
      }
    }

    _drawDanger(ctx, P, scene) {
      if (!scene.dangerZone) return;
      const dz = scene.dangerZone;
      ctx.save();
      const c = P(dz.x, dz.z);
      const pr = P(dz.x + dz.r, dz.z), pd = P(dz.x, dz.z + dz.r);
      const rx = Math.abs(pr.x - c.x), ry = Math.abs(pd.y - c.y) + Math.abs(pr.y - c.y);
      const pulse = 0.35 + 0.25 * Math.sin(performance.now() / 300);
      ctx.strokeStyle = `rgba(255,80,70,${pulse + 0.3})`;
      ctx.fillStyle = `rgba(255,60,50,${pulse * 0.35})`;
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.ellipse(c.x, c.y, rx, ry * 0.5, 0, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      ctx.fillStyle = 'rgba(255,140,130,0.9)';
      ctx.font = '11px sans-serif'; ctx.textAlign = 'center';
      ctx.fillText('⛔ ' + dz.name, c.x, c.y - ry * 0.5 - 6);
      ctx.restore();
    }

    _drawRipples(ctx, P, dt) {
      const ap = P(AP.x, AP.z);
      for (let i = this.ripples.length - 1; i >= 0; i--) {
        const rp = this.ripples[i];
        rp.r += dt * 2.4;   // m/s 拡大
        rp.a -= dt * 0.5;
        if (rp.a <= 0) { this.ripples.splice(i, 1); continue; }
        const edge = P(AP.x + rp.r, AP.z);
        const edgeD = P(AP.x, AP.z + rp.r);
        const rx = Math.abs(edge.x - ap.x);
        const ry = Math.abs(edgeD.y - ap.y) + Math.abs(edge.y - ap.y);
        ctx.strokeStyle = `rgba(90,200,255,${rp.a * 0.5})`;
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.ellipse(ap.x, ap.y, rx, ry * 0.5, 0, 0, Math.PI * 2); ctx.stroke();
      }
    }

    _drawDome(ctx, P) {
      const cx = ROOM.w / 2, cz = ROOM.d / 2;
      const center = P(cx, cz);
      const R = Math.max(ROOM.w, ROOM.d) * 0.62;
      const domeH = ROOM.h * 1.4;
      // 緯度リング
      for (let k = 1; k <= 5; k++) {
        const frac = k / 6;
        const y = domeH * Math.sin((frac * Math.PI) / 2);
        const rr = R * Math.cos((frac * Math.PI) / 2 * 0 + 0) * (1 - frac * 0.0);
        const ringR = R * Math.sqrt(1 - frac * frac);
        const e = P(cx + ringR, cz, y);
        const ed = P(cx, cz + ringR, y);
        const top = P(cx, cz, y);
        const rx = Math.abs(e.x - top.x);
        const ry = Math.abs(ed.y - top.y);
        ctx.strokeStyle = `rgba(70,150,255,${0.25 - k * 0.02})`;
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.ellipse(top.x, top.y, rx, ry, 0, 0, Math.PI * 2); ctx.stroke();
      }
      // 経度アーク
      const segs = 16;
      for (let s = 0; s < segs; s++) {
        const ang = (s / segs) * Math.PI * 2;
        ctx.strokeStyle = 'rgba(70,150,255,0.12)';
        ctx.beginPath();
        for (let t = 0; t <= 1.0001; t += 0.1) {
          const y = domeH * Math.sin((t * Math.PI) / 2);
          const rr = R * Math.cos((t * Math.PI) / 2);
          const px = cx + Math.cos(ang) * rr;
          const pz = cz + Math.sin(ang) * rr;
          const pp = P(px, pz, y);
          if (t === 0) ctx.moveTo(pp.x, pp.y); else ctx.lineTo(pp.x, pp.y);
        }
        ctx.stroke();
      }
    }

    _drawAP(ctx, P) {
      const p = P(AP.x, AP.z, 0.15);
      ctx.fillStyle = '#7fe3ff';
      ctx.shadowColor = '#7fe3ff'; ctx.shadowBlur = 12;
      ctx.beginPath(); ctx.arc(p.x, p.y, 5, 0, Math.PI * 2); ctx.fill();
      ctx.shadowBlur = 0;
      ctx.fillStyle = 'rgba(180,235,255,0.85)';
      ctx.font = '10px sans-serif'; ctx.textAlign = 'center';
      ctx.fillText('Wi-Fi AP', p.x, p.y - 10);
    }

    _drawEntity(ctx, P, d) {
      const isHuman = d.klass === 'human';
      if (isHuman) {
        // 姿勢で高さ/形状
        let bh = 1.65; // 立位
        if (d.state === 'sitting') bh = 1.05;
        if (d.state === 'lying') bh = 0.35;
        const base = P(d.x, d.z, 0);
        const top = P(d.x, d.z, bh);
        const col = d.fallRisk > 0.6 ? '255,90,80' : d.moving ? '120,255,150' : '150,255,200';

        // 影
        ctx.fillStyle = 'rgba(0,0,0,0.35)';
        ctx.beginPath(); ctx.ellipse(base.x, base.y, 16, 8, 0, 0, Math.PI * 2); ctx.fill();

        if (d.state === 'lying') {
          // 横たわり: 楕円グロー
          ctx.save();
          ctx.shadowColor = `rgba(${col},0.9)`; ctx.shadowBlur = 22;
          ctx.fillStyle = `rgba(${col},0.85)`;
          ctx.beginPath(); ctx.ellipse(base.x, base.y - 6, 30, 12, 0, 0, Math.PI * 2); ctx.fill();
          ctx.restore();
        } else {
          // 立位/着座: 縦の発光カプセル
          const grad = ctx.createLinearGradient(base.x, base.y, top.x, top.y);
          grad.addColorStop(0, `rgba(${col},0.15)`);
          grad.addColorStop(0.5, `rgba(${col},0.95)`);
          grad.addColorStop(1, `rgba(${col},0.55)`);
          ctx.save();
          ctx.shadowColor = `rgba(${col},0.9)`; ctx.shadowBlur = 24;
          ctx.strokeStyle = grad;
          ctx.lineWidth = 13;
          ctx.lineCap = 'round';
          ctx.beginPath(); ctx.moveTo(base.x, base.y); ctx.lineTo(top.x, top.y); ctx.stroke();
          // 頭部
          ctx.fillStyle = `rgba(${col},0.95)`;
          ctx.beginPath(); ctx.arc(top.x, top.y, 7, 0, Math.PI * 2); ctx.fill();
          ctx.restore();
        }

        // 呼吸パルスリング
        if (d.breathing) {
          const pulse = 0.5 + 0.5 * Math.sin(performance.now() / 1000 * (d.respRate / 60) * 2 * Math.PI);
          ctx.strokeStyle = `rgba(120,220,255,${0.25 + pulse * 0.4})`;
          ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.ellipse(base.x, base.y, 20 + pulse * 8, 10 + pulse * 4, 0, 0, Math.PI * 2); ctx.stroke();
        }

        // ラベル
        const ly = (d.state === 'lying' ? base.y - 22 : top.y - 14);
        ctx.fillStyle = '#dff6ff';
        ctx.font = 'bold 11px sans-serif'; ctx.textAlign = 'center';
        const st = global.INFER.STATE_LABEL[d.state] || '';
        ctx.fillText(`${d.label}`, base.x, ly);
        ctx.fillStyle = d.fallRisk > 0.6 ? '#ff8a80' : 'rgba(160,230,200,0.85)';
        ctx.font = '10px sans-serif';
        ctx.fillText(`人・${st} ${(d.stateConf * 100).toFixed(0)}%`, base.x, ly + 12);
      } else {
        // モノ: 橙のボックス
        const s = 0.45;
        const y = 0.55;
        const p = [
          P(d.x - s, d.z - s, 0), P(d.x + s, d.z - s, 0), P(d.x + s, d.z + s, 0), P(d.x - s, d.z + s, 0),
          P(d.x - s, d.z - s, y), P(d.x + s, d.z - s, y), P(d.x + s, d.z + s, y), P(d.x - s, d.z + s, y),
        ];
        const face = (a, b, c, dd, col) => {
          ctx.fillStyle = col; ctx.beginPath();
          ctx.moveTo(p[a].x, p[a].y); ctx.lineTo(p[b].x, p[b].y);
          ctx.lineTo(p[c].x, p[c].y); ctx.lineTo(p[dd].x, p[dd].y); ctx.closePath(); ctx.fill();
        };
        face(0, 1, 5, 4, 'rgba(210,110,60,0.92)');
        face(1, 2, 6, 5, 'rgba(180,90,45,0.92)');
        face(4, 5, 6, 7, 'rgba(235,140,80,0.95)');
        ctx.strokeStyle = 'rgba(255,180,120,0.5)'; ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(p[4].x, p[4].y); ctx.lineTo(p[5].x, p[5].y); ctx.lineTo(p[6].x, p[6].y);
        ctx.lineTo(p[7].x, p[7].y); ctx.closePath(); ctx.stroke();
        ctx.fillStyle = 'rgba(255,205,160,0.9)';
        ctx.font = '10px sans-serif'; ctx.textAlign = 'center';
        ctx.fillText(`${d.label}`, p[6].x - 20, p[7].y + 14);
        ctx.fillStyle = 'rgba(255,180,120,0.7)';
        ctx.fillText(`モノ ${(d.klassConf * 100).toFixed(0)}%`, p[6].x - 20, p[7].y + 26);
      }
    }

    _drawSweep(ctx, P, dt) {
      this.sweep += dt * 0.8;
      const cx = ROOM.w / 2, cz = ROOM.d / 2;
      const center = P(cx, cz);
      const R = Math.max(ROOM.w, ROOM.d) * 0.62;
      const ang = this.sweep % (Math.PI * 2);
      const tip = P(cx + Math.cos(ang) * R, cz + Math.sin(ang) * R, 0);
      const grad = ctx.createLinearGradient(center.x, center.y, tip.x, tip.y);
      grad.addColorStop(0, 'rgba(90,220,255,0.35)');
      grad.addColorStop(1, 'rgba(90,220,255,0)');
      ctx.strokeStyle = grad; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(center.x, center.y); ctx.lineTo(tip.x, tip.y); ctx.stroke();
    }
  }

  /* ---------- 信号波形ビュー ---------- */
  class SignalView {
    constructor(canvas) {
      this.c = canvas;
      this.ctx = canvas.getContext('2d');
      this.resize();
      window.addEventListener('resize', () => this.resize());
    }
    resize() {
      const r = this.c.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      this.c.width = r.width * dpr; this.c.height = r.height * dpr;
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.w = r.width; this.h = r.height;
    }
    /** csiHistory: number[][] (時刻 x サブキャリア) の直近 */
    draw(csiHistory) {
      const ctx = this.ctx;
      ctx.clearRect(0, 0, this.w, this.h);
      ctx.fillStyle = '#050b16'; ctx.fillRect(0, 0, this.w, this.h);
      if (!csiHistory.length) return;
      const N = csiHistory[0].length;
      const cols = ['#5ad1ff', '#7cff9a', '#ffd166', '#ff7ac0', '#b48cff'];
      const show = 6;
      const step = Math.max(1, Math.floor(N / show));
      for (let s = 0, ci = 0; s < N; s += step, ci++) {
        ctx.strokeStyle = cols[ci % cols.length];
        ctx.globalAlpha = 0.8;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        for (let t = 0; t < csiHistory.length; t++) {
          const v = csiHistory[t][s];
          const x = (t / (csiHistory.length - 1)) * this.w;
          const y = this.h / 2 - (v - 1) * this.h * 0.32;
          if (t === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
  }

  /* ---------- 汎用スパークライン / チャート ---------- */
  class Chart {
    constructor(canvas, opt) {
      this.c = canvas; this.ctx = canvas.getContext('2d');
      this.opt = opt || {};
      this.resize();
      window.addEventListener('resize', () => this.resize());
    }
    resize() {
      const r = this.c.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      this.c.width = r.width * dpr; this.c.height = r.height * dpr;
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.w = r.width; this.h = r.height;
    }
    draw(series, opt) {
      opt = Object.assign({}, this.opt, opt);
      const ctx = this.ctx;
      ctx.clearRect(0, 0, this.w, this.h);
      if (!series.length) return;
      let min = opt.min != null ? opt.min : Math.min(...series);
      let max = opt.max != null ? opt.max : Math.max(...series);
      if (max - min < 1e-6) { max = min + 1; }
      const pad = 4;
      const color = opt.color || '#5ad1ff';
      // fill
      ctx.beginPath();
      for (let i = 0; i < series.length; i++) {
        const x = pad + (i / (series.length - 1)) * (this.w - pad * 2);
        const y = this.h - pad - ((series[i] - min) / (max - min)) * (this.h - pad * 2);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.lineTo(this.w - pad, this.h - pad);
      ctx.lineTo(pad, this.h - pad);
      ctx.closePath();
      const grad = ctx.createLinearGradient(0, 0, 0, this.h);
      grad.addColorStop(0, color + '55');
      grad.addColorStop(1, color + '05');
      ctx.fillStyle = grad; ctx.fill();
      // line
      ctx.beginPath();
      for (let i = 0; i < series.length; i++) {
        const x = pad + (i / (series.length - 1)) * (this.w - pad * 2);
        const y = this.h - pad - ((series[i] - min) / (max - min)) * (this.h - pad * 2);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.stroke();
    }
  }

  /* ---------- スペクトログラム (呼吸/心拍の周波数×時間) ---------- */
  class SpectrogramView {
    constructor(canvas, opt) {
      opt = opt || {};
      this.c = canvas; this.ctx = canvas.getContext('2d');
      this.maxF = opt.maxF || 2.5;         // 表示上限 Hz (=150bpm)
      this.floor = opt.floor || 0.003;     // 対数マッピング下限
      this.top = opt.top || 0.45;          // 対数マッピング上限
      this.maxCols = opt.maxCols || 220;
      this.H = 120;
      this.off = document.createElement('canvas');
      this.off.width = this.maxCols; this.off.height = this.H;
      this.octx = this.off.getContext('2d');
      this.octx.fillStyle = '#040a14'; this.octx.fillRect(0, 0, this.maxCols, this.H);
      this.filled = 0;
      this.resize(); window.addEventListener('resize', () => this.resize());
    }
    resize() {
      const r = this.c.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      this.c.width = r.width * dpr; this.c.height = r.height * dpr;
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.w = r.width; this.h = r.height;
    }
    _color(v) {
      // 0..1 → 濃紺→青→緑→黄 (viridis 風)
      v = Math.max(0, Math.min(1, v));
      const stops = [
        [4, 10, 30], [30, 50, 120], [30, 130, 160], [80, 210, 120], [240, 230, 90],
      ];
      const p = v * (stops.length - 1);
      const i = Math.floor(p), f = p - i;
      const a = stops[i], b = stops[Math.min(i + 1, stops.length - 1)];
      return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
    }
    push(mag, freq) {
      if (!mag || !freq || freq.length < 2) return;
      // 1px 左スクロール
      this.octx.globalCompositeOperation = 'copy';
      this.octx.drawImage(this.off, -1, 0);
      this.octx.globalCompositeOperation = 'source-over';
      const x = this.maxCols - 1;
      const df = freq[1] - freq[0] || 1;
      const logF = Math.log10(this.floor), logT = Math.log10(this.top);
      for (let b = 0; b < this.H; b++) {
        const f = (b / (this.H - 1)) * this.maxF;
        let i = Math.round(f / df);
        if (i < 0) i = 0; if (i >= mag.length) i = mag.length - 1;
        const m = Math.max(mag[i], 1e-6);
        const inten = (Math.log10(m) - logF) / (logT - logF);
        const [r, g, bl] = this._color(inten);
        this.octx.fillStyle = `rgb(${r | 0},${g | 0},${bl | 0})`;
        this.octx.fillRect(x, this.H - 1 - b, 1, 1); // 低周波=下
      }
      if (this.filled < this.maxCols) this.filled++;
    }
    _y(f) { return this.h - (f / this.maxF) * this.h; }
    draw(markers) {
      const ctx = this.ctx;
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(this.off, 0, 0, this.maxCols, this.H, 0, 0, this.w, this.h);
      // 帯域の目安ライン
      const bands = [
        { f: 0.13, c: 'rgba(90,209,255,0.35)' }, { f: 0.55, c: 'rgba(90,209,255,0.35)' },
        { f: 0.83, c: 'rgba(255,107,94,0.4)' }, { f: 2.0, c: 'rgba(255,107,94,0.4)' },
      ];
      ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
      for (const bd of bands) {
        const y = this._y(bd.f);
        ctx.strokeStyle = bd.c;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(this.w, y); ctx.stroke();
      }
      ctx.setLineDash([]);
      // ラベル
      ctx.fillStyle = 'rgba(90,209,255,0.8)'; ctx.font = '10px sans-serif'; ctx.textAlign = 'left';
      ctx.fillText('呼吸帯 8–33rpm', 6, this._y(0.55) - 4);
      ctx.fillStyle = 'rgba(255,140,130,0.8)';
      ctx.fillText('心拍帯 50–120bpm', 6, this._y(2.0) + 12);
      // 検出ピークの水平マーカー
      if (markers) {
        if (markers.respHz) {
          const y = this._y(markers.respHz);
          ctx.strokeStyle = '#5ad1ff'; ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.moveTo(this.w - 60, y); ctx.lineTo(this.w, y); ctx.stroke();
          ctx.fillStyle = '#5ad1ff'; ctx.textAlign = 'right';
          ctx.fillText(`${(markers.respHz * 60).toFixed(0)}rpm`, this.w - 62, y - 3);
        }
        if (markers.heartValid && markers.heartHz) {
          const y = this._y(markers.heartHz);
          ctx.strokeStyle = '#ff6b5e'; ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.moveTo(this.w - 60, y); ctx.lineTo(this.w, y); ctx.stroke();
          ctx.fillStyle = '#ff8a80'; ctx.textAlign = 'right';
          ctx.fillText(`${(markers.heartHz * 60).toFixed(0)}bpm`, this.w - 62, y + 12);
        }
      }
      // 右端=現在
      ctx.strokeStyle = 'rgba(255,255,255,0.15)';
      ctx.beginPath(); ctx.moveTo(this.w - 1, 0); ctx.lineTo(this.w - 1, this.h); ctx.stroke();
    }
  }

  global.RENDER = { StageView, SignalView, Chart, SpectrogramView };
})(window);
