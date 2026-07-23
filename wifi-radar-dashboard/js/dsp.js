/* ============================================================
 * dsp.js — 信号処理ユーティリティ
 * Wi-Fi CSI 波形から呼吸・心拍・体動を推定するための DSP。
 * すべてブラウザ内 (vanilla JS) で完結。外部依存なし。
 * ============================================================ */
(function (global) {
  'use strict';

  /* -------- リングバッファ -------- */
  class RingBuffer {
    constructor(size) {
      this.size = size;
      this.buf = new Float32Array(size);
      this.idx = 0;
      this.filled = 0;
    }
    push(v) {
      this.buf[this.idx] = v;
      this.idx = (this.idx + 1) % this.size;
      if (this.filled < this.size) this.filled++;
    }
    /** 時系列順（古い→新しい）に取り出す */
    toArray() {
      const out = new Float32Array(this.filled);
      const start = this.filled < this.size ? 0 : this.idx;
      for (let i = 0; i < this.filled; i++) {
        out[i] = this.buf[(start + i) % this.size];
      }
      return out;
    }
    last() {
      const i = (this.idx - 1 + this.size) % this.size;
      return this.buf[i];
    }
  }

  /* -------- 統計 -------- */
  function mean(a) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i];
    return a.length ? s / a.length : 0;
  }
  function variance(a) {
    if (a.length < 2) return 0;
    const m = mean(a);
    let s = 0;
    for (let i = 0; i < a.length; i++) {
      const d = a[i] - m;
      s += d * d;
    }
    return s / (a.length - 1);
  }
  function std(a) { return Math.sqrt(variance(a)); }

  /** 直近 win サンプルの分散 = 短時間の体動エネルギー */
  function windowVariance(a, win) {
    if (a.length === 0) return 0;
    const n = Math.min(win, a.length);
    const slice = a.subarray(a.length - n);
    return variance(slice);
  }

  /* -------- 反復 radix-2 FFT -------- */
  function nextPow2(n) {
    let p = 1;
    while (p < n) p <<= 1;
    return p;
  }
  function fft(re, im) {
    const n = re.length;
    // ビット反転並べ替え
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = (-2 * Math.PI) / len;
      const wRe = Math.cos(ang), wIm = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let curRe = 1, curIm = 0;
        for (let k = 0; k < len / 2; k++) {
          const uRe = re[i + k], uIm = im[i + k];
          const vRe = re[i + k + len / 2] * curRe - im[i + k + len / 2] * curIm;
          const vIm = re[i + k + len / 2] * curIm + im[i + k + len / 2] * curRe;
          re[i + k] = uRe + vRe;
          im[i + k] = uIm + vIm;
          re[i + k + len / 2] = uRe - vRe;
          im[i + k + len / 2] = uIm - vIm;
          const nRe = curRe * wRe - curIm * wIm;
          curIm = curRe * wIm + curIm * wRe;
          curRe = nRe;
        }
      }
    }
  }

  /** Hann 窓 → FFT → 片側振幅スペクトル */
  function amplitudeSpectrum(signal, sampleRate) {
    const N = nextPow2(signal.length);
    const re = new Float32Array(N);
    const im = new Float32Array(N);
    const m = mean(signal);
    for (let i = 0; i < signal.length; i++) {
      // Hann 窓で漏れ抑制、DC 除去
      const w = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (signal.length - 1)));
      re[i] = (signal[i] - m) * w;
    }
    fft(re, im);
    const half = N >> 1;
    const mag = new Float32Array(half);
    const freq = new Float32Array(half);
    for (let i = 0; i < half; i++) {
      mag[i] = Math.hypot(re[i], im[i]) / signal.length;
      freq[i] = (i * sampleRate) / N;
    }
    return { mag, freq };
  }

  /**
   * 指定周波数帯の卓越ピークを検出し「1分あたりレート」に換算。
   * 呼吸: 0.13–0.55Hz (≈8–33 rpm) / 心拍: 0.83–2.0Hz (≈50–120 bpm)
   *
   * ノイズ床は帯域振幅の *中央値(median)* を用いる。
   * 従来の平均(bandMean)は強いピーク自身に押し上げられて突出度を過小評価し、
   * 逆にノイズだけでも見かけ上の突出が出て偽検出を招いていた。
   * median はピークの影響を受けにくい頑健なノイズ推定量。
   * 戻り値 { rate, hz, prominence, snr, confidence, peak, median, mag, freq, fLo, fHi }
   */
  function dominantRate(signal, sampleRate, fLo, fHi) {
    const empty = { rate: 0, hz: 0, prominence: 0, snr: 0, confidence: 0, peak: 0, median: 0 };
    if (signal.length < 16) return empty;
    const { mag, freq } = amplitudeSpectrum(signal, sampleRate);
    let peakIdx = -1, peakVal = -1;
    const band = [];
    for (let i = 1; i < mag.length; i++) {
      if (freq[i] >= fLo && freq[i] <= fHi) {
        band.push(mag[i]);
        if (mag[i] > peakVal) { peakVal = mag[i]; peakIdx = i; }
      }
    }
    if (peakIdx < 0 || band.length < 3) return Object.assign({}, empty, { mag, freq, fLo, fHi });
    const sorted = band.slice().sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)] || 1e-9;
    const prominence = peakVal / (median + 1e-9); // 頑健な突出度
    // 放物線補間でサブビン精度
    const N = nextPow2(signal.length);
    let hz = freq[peakIdx];
    if (peakIdx > 0 && peakIdx < mag.length - 1) {
      const a = mag[peakIdx - 1], b = mag[peakIdx], c = mag[peakIdx + 1];
      const denom = a - 2 * b + c;
      if (Math.abs(denom) > 1e-9) {
        const shift = (0.5 * (a - c)) / denom;
        hz = (peakIdx + shift) * sampleRate / N;
      }
    }
    const rate = hz * 60;
    // 確信度: 突出度を 0..1 に圧縮。白色雑音の median 比は ~2-3 なので
    // 2.5 を下限に、明確な信号(>9)で高確信になるよう設定。
    const confidence = Math.max(0, Math.min(0.98, (prominence - 2.5) / 6.5));
    return { rate, hz, prominence, snr: prominence, confidence, peak: peakVal, median, mag, freq, fLo, fHi };
  }

  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  function lerp(a, b, t) { return a + (b - a) * t; }

  global.DSP = {
    RingBuffer, mean, variance, std, windowVariance,
    fft, amplitudeSpectrum, dominantRate, clamp, lerp, nextPow2,
  };
})(window);
