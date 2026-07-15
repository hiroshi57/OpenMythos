/* 外部の公告ドキュメント(PDF/HTML)をサーバ側で取得して返すプロキシ
   ブラウザからの外部URL直fetchはCORSで不可のため中継する。
   GET /api/fetch-doc?url=<公告のURL>
   返却: { type:"pdf", b64, name } / { type:"text", text } / { error } */

function stripHtml(html) {
  return html
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/g, " ")
    .replace(/&[a-z]+;/gi, " ")
    .replace(/\s+/g, " ")
    .trim();
}

module.exports = async (req, res) => {
  try {
    const url = String(req.query.url || "");
    if (!/^https?:\/\//i.test(url)) {
      res.status(400).json({ error: "有効な url を指定してください" });
      return;
    }
    // SSRF対策: 政府系ドメインに限定しないが、ローカル/内部宛は拒否
    const host = new URL(url).hostname;
    if (/^(localhost|127\.|10\.|192\.168\.|169\.254\.|0\.)/i.test(host)) {
      res.status(400).json({ error: "許可されないホストです" });
      return;
    }
    const r = await fetch(url, {
      headers: { "User-Agent": "Mozilla/5.0 (koukan-cho-dashboard; RFP fetch)" },
      redirect: "follow",
      signal: AbortSignal.timeout(25000),
    });
    if (!r.ok) {
      res.status(502).json({ error: `取得元が ${r.status} を返しました` });
      return;
    }
    const ct = (r.headers.get("content-type") || "").toLowerCase();
    const isPdf = ct.includes("pdf") || /\.pdf($|\?)/i.test(url);
    res.setHeader("Cache-Control", "s-maxage=86400, stale-while-revalidate=604800");

    if (isPdf) {
      const buf = Buffer.from(await r.arrayBuffer());
      if (buf.length > 12 * 1024 * 1024) {
        res.status(413).json({ error: "PDFが大きすぎます(12MB超)" });
        return;
      }
      res.status(200).json({
        type: "pdf",
        name: (url.split("/").pop() || "document.pdf").split("?")[0],
        b64: buf.toString("base64"),
      });
      return;
    }
    // HTML/テキスト
    const raw = await r.text();
    const text = ct.includes("html") ? stripHtml(raw) : raw;
    res.status(200).json({ type: "text", text: text.slice(0, 40000) });
  } catch (e) {
    res.status(500).json({ error: e.message || "fetch-doc proxy error" });
  }
};
