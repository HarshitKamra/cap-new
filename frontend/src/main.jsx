import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  AlertCircle,
  Eye,
  Flame,
  ImageUp,
  Loader2,
  MessageSquare,
  Send,
  Sparkles,
  Target,
} from "lucide-react";
import "./styles.css";

const SCORE_BANDS = [
  { min: 80, label: "Excellent", tone: "strong" },
  { min: 65, label: "Good", tone: "good" },
  { min: 50, label: "Average", tone: "fair" },
  { min: -Infinity, label: "Needs Improvement", tone: "weak" },
];

function bandFor(score) {
  return SCORE_BANDS.find((band) => score >= band.min);
}

function ComponentBar({ name, value, weight }) {
  const tone = bandFor(value).tone;
  return (
    <div className="component-row">
      <div className="component-label">
        <span>{name}</span>
        <span className="component-weight">{weight}%</span>
      </div>
      <div className="meter" role="img" aria-label={`${name}: ${Math.round(value)} out of 100`}>
        <div className={`meter-fill tone-${tone}`} style={{ width: `${Math.max(2, value)}%` }} />
      </div>
      <strong className="component-value">{Math.round(value)}</strong>
    </div>
  );
}

function AttentionBar({ label, percent }) {
  return (
    <div className="attention-row">
      <span className="attention-label">{label}</span>
      <div className="meter">
        <div
          className={label === "Background" ? "meter-fill tone-muted" : "meter-fill tone-accent"}
          style={{ width: `${Math.max(1, percent)}%` }}
        />
      </div>
      <strong className="attention-value">{percent.toFixed(1)}%</strong>
    </div>
  );
}

function ChatPanel({ analysisId, chatReady, chatModel }) {
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const [chatError, setChatError] = useState("");
  const scrollRef = useRef(null);

  useEffect(() => {
    setMessages([]);
    setChatError("");
  }, [analysisId]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, sending]);

  async function send(event) {
    event.preventDefault();
    const text = draft.trim();
    if (!text || sending) return;

    // Snapshot the prior turns: the server needs history without the message
    // it is about to answer, and setState is async.
    const history = messages.map(({ role, content }) => ({ role, content }));

    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setDraft("");
    setSending(true);
    setChatError("");

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ analysis_id: analysisId, message: text, history }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "The assistant could not reply.");
      setMessages((prev) => [...prev, { role: "assistant", content: payload.reply }]);
    } catch (err) {
      setChatError(err.message);
    } finally {
      setSending(false);
    }
  }

  const starters = [
    "What is my weakest component and how do I fix it?",
    "How do I make the CTA stand out more?",
    "Is the visual hierarchy working?",
  ];

  return (
    <div className="chat-panel">
      <div className="panel-heading">
        <MessageSquare size={20} />
        <h2>Ask about this poster</h2>
        {chatReady && <span className="chat-model">{chatModel}</span>}
      </div>

      {!chatReady ? (
        <div className="empty-state compact">
          Set <code>GROQ_API_KEY</code> in a <code>.env</code> file at the project root to enable
          the assistant.
        </div>
      ) : (
        <>
          <div className="chat-log" ref={scrollRef}>
            {messages.length === 0 && (
              <div className="chat-starters">
                <p>The assistant sees your poster and its scores. Try:</p>
                {starters.map((text) => (
                  <button key={text} type="button" onClick={() => setDraft(text)}>
                    {text}
                  </button>
                ))}
              </div>
            )}

            {messages.map((message, index) => (
              <div key={index} className={`bubble bubble-${message.role}`}>
                {message.content}
              </div>
            ))}

            {sending && (
              <div className="bubble bubble-assistant bubble-pending">
                <Loader2 className="spin" size={16} /> Thinking
              </div>
            )}
          </div>

          {chatError && (
            <div className="error-box">
              <AlertCircle size={18} />
              <span>{chatError}</span>
            </div>
          )}

          <form className="chat-composer" onSubmit={send}>
            <input
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              placeholder="Ask how to improve this poster"
              aria-label="Message the assistant"
            />
            <button className="secondary-button" type="submit" disabled={sending || !draft.trim()}>
              <Send size={17} />
              Send
            </button>
          </form>
        </>
      )}
    </div>
  );
}

function App() {
  const [poster, setPoster] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [view, setView] = useState("boxes");
  const [health, setHealth] = useState({ chat_configured: false, chat_model: "" });

  useEffect(() => {
    fetch("/api/health")
      .then((response) => response.json())
      .then(setHealth)
      .catch(() => setHealth({ chat_configured: false, chat_model: "" }));
  }, []);

  // Object URLs leak until revoked, and a user may try many posters in a session.
  useEffect(() => () => previewUrl && URL.revokeObjectURL(previewUrl), [previewUrl]);

  const score = result?.pes?.score ?? 0;
  const band = useMemo(() => bandFor(score), [score]);

  const attention = useMemo(() => {
    const entries = Object.entries(result?.pes?.attention_percentages ?? {});
    return entries.sort((a, b) => b[1] - a[1]);
  }, [result]);

  function handlePosterChange(event) {
    const file = event.target.files?.[0];
    setResult(null);
    setError("");
    setView("boxes");
    setPoster(file ?? null);
    setPreviewUrl((old) => {
      if (old) URL.revokeObjectURL(old);
      return file ? URL.createObjectURL(file) : "";
    });
  }

  async function analyzePoster() {
    if (!poster) {
      setError("Upload a poster image first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", poster);
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch("/api/analyze-poster", { method: "POST", body: formData });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Analysis failed.");
      setResult(payload);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Food &amp; beverage poster analysis</p>
            <h1>Poster PES Analyzer</h1>
          </div>
          <div className="status-pill">
            <Eye size={16} />
            YOLOv8 + Saliency PES
          </div>
        </header>

        <div className="layout">
          <section className="panel uploader-panel">
            <div className="panel-heading">
              <ImageUp size={20} />
              <h2>Upload Poster</h2>
            </div>

            <label className="dropzone">
              <input type="file" accept="image/*" onChange={handlePosterChange} />
              {previewUrl ? (
                <img src={previewUrl} alt="Uploaded poster preview" />
              ) : (
                <span>Choose a JPG, PNG, WEBP, or AVIF poster</span>
              )}
            </label>

            <button className="primary-button" onClick={analyzePoster} disabled={loading}>
              {loading ? <Loader2 className="spin" size={18} /> : <Target size={18} />}
              {loading ? "Analyzing" : "Analyze Poster"}
            </button>

            {error && (
              <div className="error-box">
                <AlertCircle size={18} />
                <span>{error}</span>
              </div>
            )}
          </section>

          <section className="panel result-panel">
            <div className="panel-heading">
              <Target size={20} />
              <h2>Detection Output</h2>
              {result && (
                <div className="view-toggle">
                  <button
                    type="button"
                    className={view === "boxes" ? "active" : ""}
                    onClick={() => setView("boxes")}
                  >
                    <Target size={15} /> Boxes
                  </button>
                  <button
                    type="button"
                    className={view === "saliency" ? "active" : ""}
                    onClick={() => setView("saliency")}
                  >
                    <Flame size={15} /> Attention
                  </button>
                </div>
              )}
            </div>

            {result ? (
              <img
                className="annotated-image"
                src={view === "boxes" ? result.annotated_image : result.saliency_image}
                alt={view === "boxes" ? "YOLO bounding boxes" : "Estimated attention heatmap"}
              />
            ) : (
              <div className="empty-state">Bounding boxes appear here after analysis.</div>
            )}
          </section>
        </div>

        {result && (
          <section className="insights-grid">
            <article className="score-panel">
              <p className="eyebrow">Poster Effectiveness Score</p>
              <div className="score-row">
                <strong>{Math.round(score)}</strong>
                <span className={`band band-${band.tone}`}>{result.pes.category}</span>
              </div>
              <div className="meter meter-lg">
                <div className={`meter-fill tone-${band.tone}`} style={{ width: `${score}%` }} />
              </div>
            </article>

            <article className="components-panel">
              <p className="eyebrow">Component breakdown</p>
              {Object.entries(result.pes.components).map(([name, value]) => (
                <ComponentBar
                  key={name}
                  name={name}
                  value={value}
                  weight={result.pes.weights[name]}
                />
              ))}
            </article>

            <article className="attention-panel">
              <p className="eyebrow">Estimated attention share</p>
              {attention.map(([label, percent]) => (
                <AttentionBar key={label} label={label} percent={percent} />
              ))}
            </article>
          </section>
        )}

        {result && (
          <section className="details">
            <div className="detections">
              <h2>Detected AOIs</h2>
              {result.detections.length === 0 ? (
                <div className="empty-state compact">
                  The detector found no poster elements above the confidence threshold.
                </div>
              ) : (
                <div className="table">
                  {result.detections.map((item, index) => (
                    <div className="table-row" key={`${item.class_name}-${index}`}>
                      <span>{item.class_name}</span>
                      <span>{Math.round(item.confidence * 100)}%</span>
                      <span>{item.relative_position}</span>
                    </div>
                  ))}
                </div>
              )}

              <h2 className="stacked-heading">
                <Sparkles size={17} /> Findings
              </h2>
              <ul className="insight-list">
                {result.pes.insights.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>

            <div className="suggestions">
              <ChatPanel
                analysisId={result.analysis_id}
                chatReady={health.chat_configured}
                chatModel={health.chat_model}
              />
            </div>
          </section>
        )}
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
