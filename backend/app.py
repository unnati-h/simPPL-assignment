import os
import re
import io
import time
import shutil
from typing import List, Dict, Any, Tuple, Optional
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  

import pandas as pd
from dotenv import load_dotenv

from flask import Flask, request, jsonify, send_file, send_from_directory, abort, make_response
from flask_cors import CORS

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak,
    KeepInFrame
)

from xml.sax.saxutils import escape

from autogen import AssistantAgent, UserProxyAgent


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

config_list = [
    {"api_type": "google", "model": "gemini-2.5-flash", "api_key": GOOGLE_API_KEY}
]
llm_config = {"config_list": config_list, "temperature": 0.2}

CSV1 = '/Users/unnatihassanandani/Desktop/SimPPL/simppl data/comments Data Dump - Reddit.csv'
CSV2 = '/Users/unnatihassanandani/Desktop/SimPPL/simppl data/comments Data Dump - Youtube.csv'
CSV3 = '/Users/unnatihassanandani/Desktop/SimPPL/simppl data/posts Data Dump - Reddit.csv'
CSV4 = '/Users/unnatihassanandani/Desktop/SimPPL/simppl data/posts Data Dump - Youtube.csv'

ARTIFACTS_DIR = "artifacts"
SCRATCH_DIR = "scratch"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(SCRATCH_DIR, exist_ok=True)

#this is the data agent, this generates all the code needed to fulfil the user query and gives the code for the charts when required
#this is also the agent that debugs the code if there is an error during execution or output is not as expected
assistant = AssistantAgent(
    name="DataAgent",
    llm_config=llm_config,
    system_message=f"""
You are a Python data analysis agent.

ALWAYS start your reply with a single Python code block (no text before/after). Do not include Markdown or prose outside the fenced code.

Use THESE EXACT CSV paths (do NOT invent new ones):
- {CSV1}
- {CSV2}
- {CSV3}
- {CSV4}

RULES

1. You MUST always include at least one executable python code block (using triple backticks). Never just describe what the code would do.
2. In your FIRST code block, ALWAYS load the four CSVs exactly like this, then work on `data`:

import pandas as pd
df1 = pd.read_csv(r"{CSV1}")
df2 = pd.read_csv(r"{CSV2}")
df3 = pd.read_csv(r"{CSV3}")
df4 = pd.read_csv(r"{CSV4}")
data = pd.concat([df1, df2, df3, df4], ignore_index=True)

3. Be robust to column names:
   • For timestamps: look for columns with names containing ["date","time","created","timestamp","published","utc"].
   • For text: look for columns with names containing ["text","body","title","content","message","description","caption","comment"].
   • Use pd.to_datetime(..., errors='coerce', utc=True) for date parsing.

4. USE ALL text columns NOT JUST best coloumn

5. Adapt your analysis to the USER'S QUESTION:
   • If the question asks about hashtags: extract them with regex r'(?<!\\w)#\\w+' from text-like columns, normalize to lowercase, and count. If that yields no hashtags, try alternative extraction across multiple text-like columns.
   • If the question is about something else (e.g., word frequency, sentiment, posting activity, correlations, trends, counts), design the code accordingly.
     whenever looking for a word or phrase convert everything to lowercase first also makesure to account for puctuations before or after target word or phrase
     try to find words or phrases by using regex first and if it doesnt work use other methods
     NEVER use best coloumn for text only use ALL columns EXCEPT text_analysis also exclude counting NaN as words
     DO NOT COUNT nan / NAN / NaN as words
     When talking about user popularity measure metrics like likes,comments,views
   • Do not always return hashtags unless explicitly asked.

6. If visualization helps, generate a matplotlib chart, save it to 'output.png' (no folder).

7. OUTPUT FORMAT (always print these lines in your final output):
   • If a chart was saved: PLOT: output.png
   • FINAL: <FULL answer>
   • TERMINATE

8. if you think the output of executor/User is appropriate send a 'OUTPUT OK' message to stop

9. make sure that FINAL is not just the answer is displayed above but the answer itself is included also INCLUDE ALL LISTS IN FINAL
10. INCLUDE FINAL IN EVERY CODE

11. USE USERNAMES instead of user IDs
12. ONLY AND ONLY USE WEB FALLBACK IF DATASET IS INSUFFICIENT, TRY A FEW TIMES BEFORE USING WEB FALLBACK
13. **Web fallback via ScrapingDog Google API (ONLY if dataset is insufficient):**
    If after robust attempts the dataset cannot produce a meaningful result for the user's question (e.g., zero relevant rows, no hashtags, or dataset clearly empty/degenerate), then:

      • Use ScrapingDog Google Search API to fetch top results about the user's topic and synthesize an answer.
      • Endpoint: "https://api.scrapingdog.com/google"
      • Required params: 
        - api_key = os.environ.get("SCRAPINGDOG_API_KEY", "")
        - query = <topic or refined query>
        - results = 10
        - country = "us"
        - advance_search = "true"
        - domain = "google.com"

      • Example pattern (you can reuse/adapt this):

      api_key = os.environ.get("SCRAPINGDOG_API_KEY", "")
url = "https://api.scrapingdog.com/google"
params = {{
    "api_key": api_key,
    "query": "<your query here>",
    "results": "10",
    "country": "us",
    "advance_search": "true",
    "domain": "google.com"
}}

resp = requests.get(url, params=params, timeout=15)
if resp.status_code == 200:
    data = resp.json()
    for item in data.get("organic_results", []) or data.get("organic_data", []):
        print(item.get("title"), "-", item.get("link"), "-", item.get("snippet"))
else:
    print(f"Request failed with status code: {{resp.status_code}}")

      • Summarize concisely from those results into a clearly labeled "Web fallback" section.
      • In FINAL, include a bulleted list of the top sources (title + URL).
      • If the API key is missing or request fails, state clearly that web fallback could not be performed and still return FINAL.

13. Always keep everything inside a single Python code block and respect the OUTPUT FORMAT (PLOT, FINAL, TERMINATE).
""",
)
#had to use it over numerous iterations and fine tune the prompt to get it to this one
#I've included a web fallback: if the question is not answerable by the data the agent uses the web search results
#I wanted to use the brave seach api but it was paid

#user agent and code executor, the fist one to initiate the flow as it gets the user query from the flask endpoint
# also executes the code(autogen has built in support for code exeution ) 
# output of the code goes to datagent, if there is an error or output is not proper the data agent gives it a new code to execute
# else the data agent gives an "OUTPUT OK" Message that is the stopping criteria (or max turns)for this user agent
user_proxy = UserProxyAgent(
    name="User",
    llm_config=llm_config,
    code_execution_config={"work_dir": SCRATCH_DIR, "use_docker": False},
    human_input_mode="NEVER",
    is_termination_msg=lambda m: (
        m.get("name") == "DataAgent"
        and isinstance(m.get("content"), str)
        and m["content"].strip().endswith("OUTPUT OK")
    ),
    default_auto_reply="CONTINUE",
)

# just checks if data exists
BOOTSTRAP_CODE = f"""
import pandas as pd
CSV1 = r"{CSV1}"
CSV2 = r"{CSV2}"
CSV3 = r"{CSV3}"
CSV4 = r"{CSV4}"
if 'data' not in globals():
    df1 = pd.read_csv(CSV1)
    df2 = pd.read_csv(CSV2)
    df3 = pd.read_csv(CSV3)
    df4 = pd.read_csv(CSV4)
    data = pd.concat([df1, df2, df3, df4], ignore_index=True)
"""


CTRL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
#pdf formatting was giving me xml issues this is just to solve those
def xml_safe(s: str) -> str:
    """Remove control chars and escape XML specials."""
    return escape(CTRL_CHARS.sub("", s or ""))

#print cleaner output and plot if it exists
def extract_final_and_plot(text: str) -> Tuple[str, Optional[str]]:
    if not text:
        return "", None
    m = re.search(r"FINAL:\s*(.+?)(?:\n[A-Z]+:|$)", text, flags=re.S)
    final = m.group(1).strip() if m else text.strip()
    p = re.search(r"PLOT:\s*(\S+)", text)
    plot = p.group(1).strip() if p else None
    if plot and not os.path.isabs(plot) and os.path.dirname(plot) == "":
        plot = os.path.join(SCRATCH_DIR, plot)
    return final, plot

#Looks for FINAL from USER (happened a few times while i was trying things out that it gave the output as th print statement from the data agent)
def extract_from_history(chat_history: List[Dict[str, Any]]) -> Tuple[str, Optional[str]]:
    for msg in reversed(chat_history or []):
        if msg.get("name") == "User" and isinstance(msg.get("content"), str) and "FINAL:" in msg["content"]:
            return extract_final_and_plot(msg["content"])
    for msg in reversed(chat_history or []):
        if isinstance(msg.get("content"), str) and "FINAL:" in msg["content"]:
            return extract_final_and_plot(msg["content"])
    return "", None

# getting final code block that got the output 
def extract_last_code_block_from_dataagent(chat_history: List[Dict[str, Any]]) -> Optional[str]:
    for msg in reversed(chat_history or []):
        if msg.get("name") == "DataAgent" and isinstance(msg.get("content"), str):
            m = re.search(r"```(?:python)?\s*(.+?)```", msg["content"], flags=re.S | re.I)
            if m:
                return m.group(1).strip()
    return None
#saving the media for each question to artifacts folder and its retreival 
def safe_copy_artifact(src_path: Optional[str], turn_idx: int) -> Optional[str]:
    if not src_path or not os.path.exists(src_path):
        return None
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    base = os.path.basename(src_path)
    dst = os.path.join(ARTIFACTS_DIR, f"run_{turn_idx}_{base}")
    try:
        shutil.copy(src_path, dst)
        return dst
    except Exception:
        return None

def _shorten(text: str, max_chars: int = 1200) -> str:
    if not text:
        return "(no answer)"
    t = re.sub(r"\s+", " ", text).strip()
    return (t[: max_chars - 1] + "…") if len(t) > max_chars else t

def _try_read_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def _dataset_descriptions_only() -> Dict[str, Any]:
    sources = [
        ("Reddit Comments", CSV1),
        ("YouTube Comments", CSV2),
        ("Reddit Posts", CSV3),
        ("YouTube Posts", CSV4),
    ]
    key_columns = set()
    for _, p in sources:
        df = _try_read_csv(p)
        if df is None or df.empty:
            continue
        cols = list(df.columns)
        text_like = [c for c in cols if re.search(r"(text|body|title|content|message|description|caption|comment)", c, re.I)]
        time_like = [c for c in cols if re.search(r"(date|time|created|timestamp|published|utc)", c, re.I)]
        engagement = [c for c in cols if re.search(r"(like|reply|comment|retweet|upvote|view|score)", c, re.I)]
        chosen = (text_like[:2] or []) + (time_like[:1] or []) + engagement[:3]
        key_columns.update(chosen[:6])
    return {
        "descriptions": {
            "Reddit Comments": "User-submitted comments from Reddit threads; rich in conversational text and timestamps; may include upvotes/replies.",
            "YouTube Comments": "Viewer comments on YouTube videos; often include likes/replies and published time.",
            "Reddit Posts": "Top-level submissions on Reddit (titles + self/text posts); include creation time and community signals.",
            "YouTube Posts": "Top-level content from YouTube channels (e.g., titles/descriptions); include engagement counters and publish time.",
        },
        "key_columns": sorted(key_columns)[:12]
    }

def P(text: str, style: ParagraphStyle) -> Paragraph:
    """Always pass XML-safe, markup-free text into Paragraph."""
    return Paragraph(xml_safe(text), style)

def code_to_paragraphs(code_text: str, style: ParagraphStyle) -> list:
    """
    Render code safely with wrapping:
    - escape specials
    - preserve indentation with &nbsp;
    - preserve newlines with <br/> (only markup we add)
    We split into chunks to avoid gigantic paragraphs.
    """
    if not code_text:
        return [P("(no code captured)", style)]

    safe = xml_safe(code_text)
    safe = safe.replace("  ", "&nbsp;&nbsp;").replace("\t", "&nbsp;&nbsp;&nbsp;&nbsp;")
    safe = safe.replace("\n", "<br/>")  

    chunks, size, out = [], 8000, []
    for i in range(0, len(safe), size):
        chunks.append(safe[i:i+size])

    for chunk in chunks:
        try:
            out.append(Paragraph(chunk, style))
        except Exception:
            out.append(P(chunk.replace("<br/>", "\n"), style))
    return out

#executing the actual agent "pipeline" here (autogen is actually more chat based)
def answer_query(query: str, turn_idx: int) -> Tuple[str, Optional[str], Optional[str]]:
    user_proxy.run_code(BOOTSTRAP_CODE)
    chat_result = user_proxy.initiate_chat(
        assistant,
        message=(
            "Answer the following about the dataframe `data`.\n"
            "Question: " + query + "\n"
            "Remember to follow the instructions: write code if needed, "
            "save plots to output.png (no folder), end with 'FINAL:' + answer and optional 'PLOT:' line, then 'TERMINATE'."
        ),
        max_turns=5,
    )
    final_text, plot_path = extract_from_history(chat_result.chat_history)
    dataagent_code = extract_last_code_block_from_dataagent(chat_result.chat_history)
    artifact_plot = safe_copy_artifact(plot_path, turn_idx)
    return final_text, (artifact_plot or plot_path), dataagent_code

#chat history for the paper report
SESSIONS: List[Dict[str, Any]] = []
HISTORY: List[Tuple[str, str, Optional[str]]] = []

def next_turn_index() -> int:
    return len(SESSIONS) + 1

def answer_and_record(query: str) -> Tuple[str, Optional[str]]:
    turn_idx = next_turn_index()
    answer, chart, code = answer_query(query, turn_idx)
    SESSIONS.append({
        "question": query,
        "answer": answer or "",
        "code": code or "",
        "figure": chart,
        "timestamp": time.time(),
    })
    HISTORY.append((query, answer, chart))
    return answer, chart

#generates the paper report
def generate_pdf_paper(sessions: List[Dict[str, Any]], title="Research Report", author="AutoGen Data Lab") -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54)

    styles = getSampleStyleSheet()
    body_wrap = ParagraphStyle(name="BodyWrap", parent=styles["BodyText"], fontSize=10, leading=13, splitLongWords=True)
    h1 = styles["Heading1"]; h2 = styles["Heading2"]; h3 = styles["Heading3"]
    mono_wrap = ParagraphStyle(name="MonoWrap", parent=styles["Code"], fontName="Courier", fontSize=9, leading=11, textColor=colors.black, splitLongWords=True)

    story = []

    created = time.strftime("%Y-%m-%d %H:%M:%S")
    story.append(P(title, h1))
    story.append(Spacer(1, 6))
    story.append(P(f"Author: {author}", body_wrap))
    story.append(P(f"Created: {created}", body_wrap))
    story.append(Spacer(1, 12))

    story.append(P("Abstract", h2))
    story.append(P("This report compiles chat-driven analyses over combined social datasets. Each chat includes a concise finding and any generated visualization.", body_wrap))
    story.append(Spacer(1, 12))

    story.append(P("Datasets", h2))
    ds_meta = _dataset_descriptions_only()
    for label, desc in ds_meta["descriptions"].items():
        story.append(P(f"{label}: {desc}", body_wrap))
        story.append(Spacer(1, 4))
    if ds_meta.get("key_columns"):
        story.append(Spacer(1, 6))
        story.append(P("Common/Key Columns Observed: " + ", ".join(ds_meta.get("key_columns", [])), body_wrap))
    story.append(Spacer(1, 12))

    story.append(P("Methods (Agentic Analysis Pipeline)", h2))
    story.append(P('We used an AutoGen two-agent setup: a "DataAgent" that writes Python to load and analyze the CSVs, and a "User" executor that runs the code in a sandbox. The DataAgent searches across all text-like columns, parses timestamps robustly, and generates Matplotlib figures saved to output.png when useful.', body_wrap))
    story.append(Spacer(1, 12))

    story.append(P("Chat-by-Chat Summaries", h2))
    max_width = A4[0] - (doc.leftMargin + doc.rightMargin)
    if not sessions:
        story.append(P("No analysis runs captured yet.", body_wrap))
        story.append(Spacer(1, 12))
    else:
        for i, s in enumerate(sessions, 1):
            q = s.get("question", "") or "(no question)"
            a = s.get("answer", "") or "(no answer)"
            story.append(P(f"S{i}. {q}", h3))
            story.append(P(f"Summary: {_shorten(a)}", body_wrap))
            story.append(Spacer(1, 6))
            fig_path = s.get("figure")
            if fig_path and os.path.exists(fig_path):
                try:
                    img = RLImage(fig_path)
                    iw, ih = img.wrap(0, 0)
                    scale = min(max_width / iw, 1.0) if iw else 1.0
                    img.drawWidth = iw * scale
                    img.drawHeight = ih * scale
                    story.append(img)
                    story.append(P(f"Figure S{i}. Visualization generated during this chat.", body_wrap))
                    story.append(Spacer(1, 12))
                except Exception:
                    story.append(P(f"(Figure could not be embedded: {fig_path})", body_wrap))
                    story.append(Spacer(1, 12))
            else:
                story.append(P("(No figure for this chat.)", body_wrap))
                story.append(Spacer(1, 12))
    story.append(PageBreak())

    story.append(P("Discussion", h2))
    story.append(P("The summaries reflect automated analyses over combined social datasets. Future iterations can enrich entity resolution, topic modeling, and align spikes to external events/news.", body_wrap))
    story.append(Spacer(1, 12))

    story.append(P("Limitations", h2))
    for b in [
        "Dependent on column naming and data cleanliness; missing/ambiguous fields can affect counts.",
        "Visualizations are single snapshots; multi-facet or interactive plots may improve insight.",
        "No external knowledge integration beyond the provided CSVs.",
    ]:
        story.append(P(f"- {b}", body_wrap))
    story.append(PageBreak())

    story.append(P("Reproducibility Appendix (Code)", h2))
    if not sessions:
        story.append(P("No code captured yet.", body_wrap))
    else:
        usable_width = max_width
        for i, s in enumerate(sessions, 1):
            story.append(P(f"A{i}. Code for S{i}", h3))
            code_text = s.get("code") or "# (No code block captured for this run.)"
            code_flows = code_to_paragraphs(code_text, mono_wrap)
            kif = KeepInFrame(usable_width, A4[1] - (doc.topMargin + doc.bottomMargin) - 60, code_flows, mode='shrink')
            story.append(kif)
            story.append(Spacer(1, 12))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

#flask endpoints
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": [
    "http://127.0.0.1:5173","http://localhost:5173",
    "http://127.0.0.1:5174","http://localhost:5174",
    "http://127.0.0.1:4173","http://localhost:4173",
    "http://127.0.0.1:5500","http://localhost:5500"
], "methods": ["GET","POST","OPTIONS"], "allow_headers": ["Content-Type"]}})

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.post("/echo")
def echo():
    data = request.get_json(silent=True)
    return jsonify({"received": data, "ok": True})

@app.post("/analyze")
def analyze():
    try:
        data = request.get_json(silent=True) or {}
        print("[/analyze] incoming JSON:", data, flush=True)
        query = (data.get("query") or "").strip()
        if not query:
            payload = {"error": "Missing 'query' in JSON body"}
            print("[/analyze] RESPONSE:", payload, flush=True)
            resp = make_response(json.dumps(payload), 400)
            resp.headers["Content-Type"] = "application/json"
            return resp
        turn_idx_before = next_turn_index()
        final, chart = answer_and_record(query)
        code = SESSIONS[-1].get("code", "") if SESSIONS else ""
        plot_path = SESSIONS[-1].get("figure")
        plot_url = f"/artifacts/{os.path.basename(plot_path)}" if plot_path else None
        payload = {"final": final, "plot_path": plot_path, "plot_url": plot_url, "code": code, "turn": turn_idx_before}
        print("[/analyze] RESPONSE:", payload, flush=True)
        resp = make_response(json.dumps(payload), 200)
        resp.headers["Content-Type"] = "application/json"
        return resp
    except Exception as e:
        payload = {"error": str(e)}
        print("[/analyze] ERROR RESPONSE:", payload, flush=True)
        resp = make_response(json.dumps(payload), 500)
        resp.headers["Content-Type"] = "application/json"
        return resp

@app.get("/artifacts/<path:filename>")
def serve_artifact(filename):
    file_path = os.path.join(ARTIFACTS_DIR, filename)
    if not os.path.isfile(file_path):
        abort(404)
    return send_from_directory(ARTIFACTS_DIR, filename, as_attachment=False)

@app.post("/paper")
def make_paper():
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "Research Report").strip()
    author = (data.get("author") or "AutoGen Data Lab").strip()
    try:
        pdf_bytes = generate_pdf_paper(SESSIONS, title=title, author=author)
        ts = time.strftime("%Y%m%d_%H%M%S")
        pdf_name = f"research_report_{ts}.pdf"
        with open(os.path.join(ARTIFACTS_DIR, pdf_name), "wb") as f:
            f.write(pdf_bytes)
        return send_file(io.BytesIO(pdf_bytes), mimetype="application/pdf", as_attachment=True, download_name=pdf_name)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False, use_reloader=False)
