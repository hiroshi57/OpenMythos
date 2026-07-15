/* 官公需情報ポータル(kkj.go.jp) 公示中案件のプロキシAPI
   ブラウザ直アクセスはCORSで不可のため、Vercelサーバレス関数で中継してJSON化する。
   GET /api/open-bids?q=キーワード&count=40 */

const KKJ_API = "https://www.kkj.go.jp/api/";

function pick(block, tag) {
  const m = block.match(new RegExp(`<${tag}>([\\s\\S]*?)</${tag}>`));
  if (!m) return null;
  return m[1].replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g, "$1").trim() || null;
}

function parseKkj(xml) {
  const hits = +(xml.match(/<SearchHits>(\d+)<\/SearchHits>/) || [])[1] || 0;
  const items = [];
  const re = /<SearchResult>([\s\S]*?)<\/SearchResult>/g;
  let m;
  while ((m = re.exec(xml))) {
    const b = m[1];
    const name = pick(b, "ProjectName");
    if (!name) continue;
    items.push({
      name,
      url: pick(b, "ExternalDocumentURI"),
      org: pick(b, "OrganizationName"),
      pref: pick(b, "PrefectureName"),
      city: pick(b, "CityName"),
      category: pick(b, "Category"),
      procedure: pick(b, "ProcedureType"),
      issue_date: (pick(b, "CftIssueDate") || "").slice(0, 10) || null,
      deadline: (pick(b, "TenderSubmissionDeadline") || "").slice(0, 10) || null,
      description: (pick(b, "ProjectDescription") || "").replace(/\s+/g, " ").slice(0, 1200),
      file_type: pick(b, "FileType"),
    });
  }
  return { hits, items };
}

module.exports = async (req, res) => {
  try {
    const q = String(req.query.q || "").slice(0, 100);
    const count = Math.min(100, Math.max(1, parseInt(req.query.count, 10) || 40));
    if (!q.trim()) {
      res.status(400).json({ error: "q (検索キーワード) を指定してください" });
      return;
    }
    const url = `${KKJ_API}?Query=${encodeURIComponent(q)}&Count=${count}`;
    const r = await fetch(url, {
      headers: { "User-Agent": "Mozilla/5.0 (koukan-cho-dashboard; bid research)" },
      signal: AbortSignal.timeout(20000),
    });
    if (!r.ok) {
      res.status(502).json({ error: `kkj.go.jp が ${r.status} を返しました` });
      return;
    }
    const xml = await r.text();
    const data = parseKkj(xml);
    res.setHeader("Cache-Control", "s-maxage=1800, stale-while-revalidate=3600");
    res.status(200).json({ source: "官公需情報ポータル(kkj.go.jp)", query: q, ...data });
  } catch (e) {
    res.status(500).json({ error: e.message || "proxy error" });
  }
};

module.exports.parseKkj = parseKkj;
