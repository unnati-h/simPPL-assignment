import { useEffect, useRef, useState } from "react";
import { API_BASE } from "../lib/api";

function Message({ role, text, code, plotUrl, turn }) {
  return (
    <div className={`w-full flex ${role === "user" ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[85%] md:max-w-[70%] rounded-2xl p-4 shadow-sm border ${
          role === "user"
            ? "bg-white text-black border-neutral-200"
            : "bg-neutral-950 text-white border-neutral-800"
        }`}
      >
        <div className="text-xs opacity-60 mb-1">{role === "user" ? "You" : `Assistant${turn !== undefined ? ` • Turn ${turn}` : ""}`}</div>
        {text && <div className="leading-relaxed whitespace-pre-wrap">{text}</div>}

        {plotUrl && (
          <a
            className="block mt-4 rounded-xl overflow-hidden border border-neutral-800 hover:border-neutral-700 transition"
            href={`${API_BASE}${plotUrl}`}
            target="_blank"
            rel="noreferrer"
            title="Open chart in new tab"
          >
            <img src={`${API_BASE}${plotUrl}`} alt="Chart" className="w-full h-auto block" />
          </a>
        )}

        {code && code.trim() && (
          <details className="mt-4 group">
            <summary className="cursor-pointer select-none text-sm opacity-80 hover:opacity-100">
              View generated code
            </summary>
            <pre className="mt-2 p-3 bg-black text-white/90 rounded-lg overflow-auto text-sm border border-neutral-800">
{code}
            </pre>
          </details>
        )}
      </div>
    </div>
  );
}

export default function Chat() {
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      text: "Hi! Ask a question about your data and I’ll analyze it, generate charts, and show the code.",
    },
  ]);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy]);

  async function sendQuery(e) {
    e?.preventDefault();
    const q = input.trim();
    if (!q || busy) return;

    setMessages((m) => [...m, { role: "user", text: q }]);
    setInput("");
    setBusy(true);

    try {
      const res = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q }),
      });

      const data = await res.json().catch(() => ({}));

      if (!res.ok) {
        const msg = data?.error || `Server error (${res.status})`;
        setMessages((m) => [...m, { role: "assistant", text: `❌ ${msg}` }]);
      } else {
        const { final, plot_url, code, turn } = {
          final: data.final ?? "",
          plot_url: data.plot_url ?? data.plotPath ?? null,
          code: data.code ?? "",
          turn: data.turn,
        };

        setMessages((m) => [
          ...m,
          { role: "assistant", text: final || "No answer returned.", plotUrl: plot_url, code, turn },
        ]);
      }
    } catch (err) {
      setMessages((m) => [...m, { role: "assistant", text: `❌ Network error: ${String(err)}` }]);
    } finally {
      setBusy(false);
    }
  }

  async function makePaper() {
    try {
      const res = await fetch(`${API_BASE}/paper`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: "Research Report", author: "AutoGen Data Lab" }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => null);
        throw new Error(err?.error || `HTTP ${res.status}`);
      }

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `research_report_${Date.now()}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (e) {
      setMessages((m) => [...m, { role: "assistant", text: `❌ Could not create paper: ${String(e.message || e)}` }]);
    }
  }

  return (
    <div className="min-h-dvh bg-black text-white flex flex-col">
      <header className="sticky top-0 z-10 border-b border-neutral-900 bg-black/70 backdrop-blur">
        <div className="mx-auto max-w-4xl px-4 py-3 flex items-center justify-between">
          <div className="font-semibold text-neutral-300">Datafy</div>
          <div className="flex gap-2">
            <button
              onClick={makePaper}
              className="rounded-full border border-neutral-800 px-4 py-2 text-sm hover:bg-white hover:text-black transition"
              title="Compile past turns into a PDF via /paper"
            >
              Make Paper (PDF)
            </button>
          </div>
        </div>
      </header>


      <main className="flex-1">
        <div className="mx-auto max-w-4xl px-4 py-6 space-y-4">
          {messages.map((m, i) => (
            <Message key={i} role={m.role} text={m.text} code={m.code} plotUrl={m.plotUrl} turn={m.turn} />
          ))}

          {busy && (
            <div className="w-full flex justify-start">
              <div className="text-neutral-400 text-sm">Analyzing…</div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </main>

      <footer className="border-t border-neutral-900 bg-black/70 backdrop-blur">
        <form onSubmit={sendQuery} className="mx-auto max-w-4xl px-4 py-4">
          <div className="flex items-center gap-2">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question about your data…"
              className="flex-1 rounded-2xl bg-neutral-950 border border-neutral-800 px-4 py-3 text-white placeholder:text-neutral-500 focus:outline-none focus:ring-2 focus:ring-white/20"
            />
            <button
              disabled={busy || !input.trim()}
              className="rounded-2xl px-5 py-3 font-semibold border border-neutral-700 bg-white text-black disabled:opacity-50 hover:bg-neutral-200 active:scale-95 transition"
              type="submit"
            >
              Send
            </button>
          </div>
          <div className="mt-2 text-xs text-neutral-500">
            Hints: “what is the most popular hashtag”, “show top accounts”, “plot comments per day”…
          </div>
        </form>
      </footer>
    </div>
  );
}
