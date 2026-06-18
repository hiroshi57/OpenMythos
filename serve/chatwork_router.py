"""
serve/chatwork_router.py — Chatwork 共有投稿エンドポイント

POST /v1/chatwork/preview : URL から 4 行 (＜事業領域/サービス＞/件名/URL/80文字要約) を生成 (投稿しない)
POST /v1/chatwork/share   : 上記を生成し Chatwork へ投稿 (トークン未設定なら preview)
GET  /chatwork            : ブラウザUI (URL 入力 → プレビュー / 投稿ボタン)
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from open_mythos.skills.chatwork_poster import (
    BusinessDomain,
    ChatworkShareEngineFactory,
    DEFAULT_ROOM_ID,
    DEFAULT_SUMMARY_CHARS,
)

router = APIRouter()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# リクエスト / レスポンス モデル
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ShareRequest(BaseModel):
    url:      str           = Field(..., description="共有したいページの URL")
    room_id:  Optional[str] = Field(None, description="投稿先 Chatwork ルーム ID (既定: 環境変数 / 336261448)")
    domain:   Optional[str] = Field(None, description="事業領域/サービスを手動指定 (省略時は自動分類)")
    subject:  Optional[str] = Field(None, description="件名を手動指定 (省略時はページ title)")
    dry_run:  bool          = Field(False, description="True なら投稿せずプレビューのみ")


class SharePostModel(BaseModel):
    domain:  str
    subject: str
    url:     str
    summary: str
    message: str


class ShareResponse(BaseModel):
    posted:            bool
    room_id:           str
    message:           str
    post:              SharePostModel
    reason:            Optional[str] = None
    message_id:        Optional[str] = None
    chatwork_response: Optional[dict] = None


def _resolve_domain(value: Optional[str]) -> Optional[BusinessDomain]:
    if not value:
        return None
    v = value.strip()
    for d in BusinessDomain:
        if v == d.value or v == d.name:
            return d
    valid = "、".join(d.value for d in BusinessDomain)
    raise HTTPException(
        status_code=422,
        detail=f"不正な事業領域/サービス: {value} (有効値: {valid})",
    )


def _run(req: ShareRequest, *, force_dry: bool) -> dict:
    engine = ChatworkShareEngineFactory.from_env()
    domain = _resolve_domain(req.domain)
    try:
        return engine.share(
            req.url,
            room_id=req.room_id,
            dry_run=force_dry or req.dry_run,
            domain_override=domain,
            subject_override=req.subject,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# エンドポイント
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/v1/chatwork/preview", response_model=ShareResponse, tags=["chatwork"])
def chatwork_preview(req: ShareRequest):
    """投稿せずに 4 行プレビューを生成する。"""
    return _run(req, force_dry=True)


@router.post("/v1/chatwork/share", response_model=ShareResponse, tags=["chatwork"])
def chatwork_share(req: ShareRequest):
    """4 行を生成して Chatwork に投稿する (トークン未設定時は preview)。"""
    return _run(req, force_dry=False)


@router.get("/chatwork", response_class=HTMLResponse, include_in_schema=False)
def chatwork_ui():
    """URL を入力して投稿ボタンを押すだけの最小 UI。"""
    domains = "".join(
        f'<option value="{d.value}">{d.value}</option>' for d in BusinessDomain
    )
    return HTMLResponse(content=f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chatwork 共有投稿</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
  h1 {{ font-size: 1.3rem; }}
  label {{ display: block; margin: .8rem 0 .2rem; font-weight: 600; font-size: .9rem; }}
  input, select {{ width: 100%; padding: .55rem; font-size: 1rem; box-sizing: border-box; border: 1px solid #ccc; border-radius: 6px; }}
  .row {{ display: flex; gap: .6rem; margin-top: 1rem; }}
  button {{ flex: 1; padding: .7rem; font-size: 1rem; border: 0; border-radius: 6px; cursor: pointer; }}
  #preview {{ background: #eef; color: #224; }}
  #post {{ background: #2563eb; color: #fff; }}
  button:disabled {{ opacity: .5; cursor: progress; }}
  pre {{ background: #0f172a; color: #e2e8f0; padding: 1rem; border-radius: 8px; white-space: pre-wrap; word-break: break-all; margin-top: 1rem; min-height: 4rem; }}
  .meta {{ font-size: .8rem; color: #666; margin-top: .5rem; }}
</style>
</head>
<body>
  <h1>Chatwork 共有投稿</h1>
  <p class="meta">URL を入れて「投稿」を押すと、件名・80文字要約・事業領域を自動付与して Chatwork に投稿します。</p>

  <label for="url">URL</label>
  <input id="url" type="url" placeholder="https://..." autofocus>

  <label for="domain">事業領域/サービス (空欄なら自動分類)</label>
  <select id="domain"><option value="">自動分類</option>{domains}</select>

  <label for="room">Chatwork ルーム ID</label>
  <input id="room" type="text" value="{DEFAULT_ROOM_ID}">

  <div class="row">
    <button id="preview">プレビュー</button>
    <button id="post">投稿</button>
  </div>

  <pre id="out">ここに結果が表示されます</pre>
  <p class="meta" id="status"></p>

<script>
async function run(path) {{
  const url = document.getElementById('url').value.trim();
  const out = document.getElementById('out');
  const status = document.getElementById('status');
  if (!url) {{ out.textContent = 'URL を入力してください'; return; }}
  const body = {{
    url,
    room_id: document.getElementById('room').value.trim() || null,
    domain: document.getElementById('domain').value || null,
  }};
  const btns = document.querySelectorAll('button');
  btns.forEach(b => b.disabled = true);
  status.textContent = '処理中...';
  try {{
    const res = await fetch(path, {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify(body),
    }});
    const data = await res.json();
    if (!res.ok) {{ out.textContent = 'エラー: ' + (data.detail || res.status); }}
    else {{
      out.textContent = data.message;
      status.textContent = data.posted
        ? '✅ 投稿しました (message_id: ' + (data.message_id || '?') + ')'
        : '📝 プレビュー (' + (data.reason === 'no_token' ? 'トークン未設定のため未投稿' : data.reason) + ')';
    }}
  }} catch (e) {{
    out.textContent = '通信エラー: ' + e;
  }} finally {{
    btns.forEach(b => b.disabled = false);
  }}
}}
document.getElementById('preview').onclick = () => run('/v1/chatwork/preview');
document.getElementById('post').onclick = () => run('/v1/chatwork/share');
</script>
</body>
</html>""")
