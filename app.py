"""
一橋地理対策 AI 時事論述フラッシュカードアプリ
=================================================
* UI    : Streamlit
* LLM   : Hugging Face InferenceClient (Llama-3.2-3B-Instruct)
* 環境変数: プロジェクト直下の .env のみを明示的に読み込む
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 環境変数 — このファイルが置かれているディレクトリの .env のみを読み込む
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ENV_PATH = _HERE / ".env"

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=_ENV_PATH, override=False)
except ImportError:
    print(
        "[警告] python-dotenv が未インストールです。",
        file=sys.stderr,
    )

import streamlit as st
from huggingface_hub import InferenceClient

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

SEED_CATEGORIES: list[str] = [
    "資源・エネルギー（石油・天然ガス・再生可能エネルギー・資源ナショナリズム）",
    "食料・農業（食料安全保障・フードマイレージ・農地争奪・遺伝子組み換え）",
    "地政学・国際秩序（ウクライナ・台湾・中東・南シナ海・NATO拡大）",
    "人口・移民・難民（少子高齢化・移民受け入れ・難民問題・人口ボーナス）",
    "環境・気候変動（気候変動枠組み条約・砂漠化・熱帯林消失・海面上昇）",
    "貿易・サプライチェーン（グローバリゼーション・デカップリング・半導体・港湾）",
    "都市化・メガシティ（スラム・プライメイトシティ・スマートシティ・ジェントリフィケーション）",
    "民族・宗教・言語（多民族国家・宗教対立・国境問題・植民地遺制）",
    "水資源・水問題（国際河川・仮想水・水ストレス・ダム建設と下流国）",
    "交通・物流インフラ（一帯一路・海上航路・陸橋・チョークポイント）",
    "格差・開発（南北問題・BRICs・援助と依存・LDC・SDGs）",
    "感染症・医療（パンデミック・医療格差・グローバルヘルス・薬剤耐性菌）",
]

_SYSTEM_PROMPT = """\
あなたは一橋大学の地理論述問題を作成する専門家AIです。
問題を生成する前に必ず「自己ファクトチェック」を実施し、
事実誤認・不正確な統計・存在しない条約名などが含まれていないかを検証してください。
誤りが検出された場合は内部フィードバックとして記録し、正確な情報で問題を作り直してください。

出力は**必ず下記のJSONスキーマのみ**で返してください。
説明文・Markdown・コードブロック記号（```）は一切含めないこと。

JSONスキーマ:
{
  "fact_check": {
    "is_correct": <true | false>,
    "verification_notes": "<検証ログ（100字以内）>",
    "internal_feedback": "<修正点または'なし'>"
  },
  "status": "<'valid' | 'revised'>",
  "data": {
    "theme": "<テーマ名（30字以内）>",
    "region": "<対象地域>",
    "keywords": ["<KW1>", "<KW2>", "<KW3>", "<KW4>"],
    "question": "<150字程度の論述問題文（地理的因果関係を問う形式）>",
    "answer": "<150〜200字程度の模範解答のポイント・要約>"
  }
}
"""

_USER_PROMPT_TEMPLATE = """\
今日のシードカテゴリ: 【{category}】

上記カテゴリに関連し、2015年以降の実際の時事・地政学的事象を題材に、
一橋大学二次試験らしい「地理的因果関係・空間的相互依存・地域間格差」を問う
論述問題を1問生成してください。

ファクトチェックを必ず実施し、指定のJSONフォーマットで返答してください。
"""

# ---------------------------------------------------------------------------
# LLM クライアント（セッションキャッシュ）
# ---------------------------------------------------------------------------
@st.cache_resource
def _get_client() -> InferenceClient:
    # Streamlit Cloud の Secrets → 環境変数 の順で取得
    token = st.secrets.get("HF_TOKEN", "") if hasattr(st, "secrets") else ""
    if not token:
        token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise ValueError(
            "HF_TOKEN が設定されていません。\n"
            "Streamlit Cloud の場合: Settings → Secrets に HF_TOKEN を追加してください。\n"
            "ローカルの場合: .env ファイルに HF_TOKEN=hf_... を記載してください。"
        )
    return InferenceClient(token=token)


# ---------------------------------------------------------------------------
# JSON 抽出ユーティリティ
# ---------------------------------------------------------------------------
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(raw: str) -> dict:
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    match = _JSON_RE.search(cleaned)
    if not match:
        raise ValueError(f"JSONが見つかりませんでした。生のレスポンス:\n{raw[:500]}")
    return json.loads(match.group())


# ---------------------------------------------------------------------------
# カード生成ロジック
# ---------------------------------------------------------------------------
def _validate_card(data: dict) -> None:
    required = ("theme", "region", "keywords", "question", "answer")
    missing = [k for k in required if not data.get(k)]
    if missing:
        raise ValueError(f"必須フィールドが欠落しています: {missing}")
    if not isinstance(data["keywords"], list) or len(data["keywords"]) < 1:
        raise ValueError("keywords は配列である必要があります。")


def generate_card() -> dict:
    category = random.choice(SEED_CATEGORIES)
    client = _get_client()

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _USER_PROMPT_TEMPLATE.format(category=category)},
    ]

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        max_tokens=1200,
        temperature=0.85,
        top_p=0.95,
    )
    raw_text: str = response.choices[0].message.content or ""
    parsed = _extract_json(raw_text)

    fact_check: dict = parsed.get("fact_check", {})
    status: str = parsed.get("status", "unknown")
    card: dict = parsed.get("data", {})

    _validate_card(card)

    return {"card": card, "fact_check": fact_check, "status": status}


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="一橋地理 AI フラッシュカード",
        page_icon="📚",
        layout="centered",
    )

    # カスタムCSS
    st.markdown("""
    <style>
    .card-box {
        background: linear-gradient(135deg, #f0f7ff 0%, #fafcff 100%);
        border: 1.5px solid #b8d4ee;
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 16px;
    }
    .question-box {
        background: #fff;
        border: 1px solid #cce0f5;
        border-radius: 8px;
        padding: 16px 18px;
        font-size: 1.0em;
        line-height: 1.85;
        color: #1a1a2e;
        margin-top: 12px;
    }
    .kw-tag {
        display: inline-block;
        background: #e8f4f8;
        color: #1a6e8a;
        border-radius: 4px;
        padding: 2px 10px;
        margin: 3px 3px 3px 0;
        font-size: 0.88em;
        font-weight: 600;
    }
    .footer-note {
        margin-top: 28px;
        padding-top: 12px;
        border-top: 1px solid #ddd;
        color: #999;
        font-size: 0.82em;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    # ヘッダー
    st.title("📚 一橋地理対策 AI 時事論述フラッシュカード")
    st.caption(
        "過去10年の世界の時事・政治経済・地誌・社会課題を題材に、"
        "AI が一橋大学二次試験スタイルの論述問題を **ファクトチェック付き** でリアルタイム生成します。"
    )
    st.divider()

    # 生成ボタン
    if st.button("🎲 新しい問題を生成（ファクトチェック付）", type="primary", use_container_width=True):
        with st.spinner("⏳ AI がファクトチェックを実施しながら問題を生成しています…"):
            try:
                result = generate_card()
                st.session_state["result"] = result
            except Exception as exc:
                st.session_state["error"] = str(exc)
                st.session_state.pop("result", None)

    # エラー表示
    if "error" in st.session_state and "result" not in st.session_state:
        st.error(f"⚠️ エラーが発生しました:\n\n{st.session_state['error']}")

    # カード表示
    if "result" in st.session_state:
        r = st.session_state["result"]
        card = r["card"]
        fact_check = r["fact_check"]
        status = r["status"]

        # キーワードタグ
        kw_tags = " ".join(
            f'<span class="kw-tag">{kw}</span>' for kw in card["keywords"]
        )

        # カード本体
        st.markdown(f"""
        <div class="card-box">
            <div style="font-size:1.15em;font-weight:700;color:#1a3a5c;">{card['theme']}</div>
            <div style="color:#666;font-size:0.92em;margin-bottom:10px;">🌏 対象地域: {card['region']}</div>
            <div>{kw_tags}</div>
            <div class="question-box">
                <b>【論述問題】</b><br><br>{card['question']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 模範解答（折りたたみ）
        with st.expander("📝 模範解答のポイントを見る"):
            st.markdown(
                f'<div style="line-height:1.9;font-size:0.97em;">{card["answer"]}</div>',
                unsafe_allow_html=True,
            )

            # ファクトチェックログ
            is_correct = bool(fact_check.get("is_correct", True))
            check_icon = "✅" if is_correct else "⚠️"
            check_color = "#1a7a2e" if is_correct else "#b03a2e"
            status_badge = "valid" if status == "valid" else "revised（自己修正済み）"

            st.markdown(f"""
            <div style="background:#f8f9fa;border-left:4px solid {check_color};
                        padding:10px 14px;border-radius:4px;font-size:0.9em;
                        line-height:1.7;margin-top:12px;">
                <b>{check_icon} ファクトチェック結果</b>
                （ステータス: <code>{status_badge}</code>）<br>
                <b>検証ログ:</b> {fact_check.get("verification_notes", "—")}<br>
                <b>内部フィードバック:</b> {fact_check.get("internal_feedback", "なし")}
            </div>
            """, unsafe_allow_html=True)

    # フッター
    st.markdown("""
    <div class="footer-note">
        このアプリは学習支援目的で作成されています。生成された問題・解答は AI による自動生成であり、
        実際の一橋大学入試問題とは異なります。<br>
        使用モデル: meta-llama/Llama-3.2-3B-Instruct（Hugging Face Inference API）
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
