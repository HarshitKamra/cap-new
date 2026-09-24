import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  AlertCircle,
  Eye,
  Flame,
  ImageUp,
  Lightbulb,
  Loader2,
  MessageSquare,
  Send,
  Sparkles,
  Target,
  X,
} from "lucide-react";
import "./styles.css";

const SCORE_BANDS = [
  { min: 80, label: "Excellent", tone: "strong" },
  { min: 65, label: "Good", tone: "good" },
  { min: 50, label: "Average", tone: "fair" },
  { min: -Infinity, label: "Needs Improvement", tone: "weak" },
];

const ACCEPTED_TYPES = /^image\/(jpeg|png|webp|avif)$/;

function bandFor(score) {
  return SCORE_BANDS.find((band) => score >= band.min);
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

/* --- Minimal markdown --------------------------------------------------- */
// Groq replies arrive as markdown (bold lead-ins, numbered steps). Rendering
// them as plain text put literal ** in the bubbles, so this walks the handful of
// constructs the assistant actually emits instead of pulling in a parser.

const INLINE_PATTERN = /(\*\*[^*]+\*\*|`[^`]+`)/g;

function renderInline(text, keyPrefix) {
  return text
    .split(INLINE_PATTERN)
    .filter(Boolean)
    .map((chunk, index) => {
      const key = `${keyPrefix}-${index}`;
      if (chunk.startsWith("**") && chunk.endsWith("**")) {
        return <strong key={key}>{chunk.slice(2, -2)}</strong>;
      }
      if (chunk.startsWith("`") && chunk.endsWith("`")) {
        return <code key={key}>{chunk.slice(1, -1)}</code>;
      }
      return <React.Fragment key={key}>{chunk}</React.Fragment>;
    });
}

function renderMarkdown(text) {
  const blocks = [];
  let list = null;
  let para = [];

  function flushPara() {
    if (!para.length) return;
    const index = blocks.length;
    blocks.push(<p key={`p-${index}`}>{renderInline(para.join(" "), `p${index}`)}</p>);
    para = [];
  }

  function flushList() {
    if (!list) return;
    const index = blocks.length;
    const Tag = list.ordered ? "ol" : "ul";
    blocks.push(
      <Tag key={`l-${index}`} className="md-list">
        {list.items.map((item, itemIndex) => (
          <li key={itemIndex}>{renderInline(item, `l${index}-${itemIndex}`)}</li>
        ))}
      </Tag>,
    );
    list = null;
  }

  for (const rawLine of text.split("\n")) {
    const line = rawLine.trim();

    if (!line) {
      flushPara();
      flushList();
      continue;
    }

    const ordered = line.match(/^\d+[.)]\s+(.*)$/);
    const bullet = line.match(/^[-*\u2022]\s+(.*)$/);

    if (ordered) {
      flushPara();
      if (!list || !list.ordered) {
        flushList();
        list = { ordered: true, items: [] };
      }
      list.items.push(ordered[1]);
    } else if (bullet) {
      flushPara();
      if (!list || list.ordered) {
        flushList();
        list = { ordered: false, items: [] };
      }
      list.items.push(bullet[1]);
    } else if (list && /^\s/.test(rawLine)) {
      // Indented run-on belongs to the list item above it, not a new paragraph.
      list.items[list.items.length - 1] += ` ${line}`;
    } else {
      flushList();
      para.push(line);
    }
  }

  flushPara();
  flushList();
  return blocks;
}

/* --- Presentational pieces ---------------------------------------------- */

function ScoreRing({ score, tone }) {
  const radius = 54;
  const circumference = 2 * Math.PI * radius;
  const clamped = Math.max(0, Math.min(100, score));
  const dash = (clamped / 100) * circumference;

  return (
    <div className="score-ring">
      <svg viewBox="0 0 128 128" aria-hidden="true">
        <circle className="ring-track" cx="64" cy="64" r={radius} />
        <circle
          className={`ring-value stroke-${tone}`}
          cx="64"
          cy="64"
          r={radius}
          strokeDasharray={`${dash} ${circumference}`}
          transform="rotate(-90 64 64)"
        />
      </svg>
      <div className="score-ring-center">
        <strong>{Math.round(score)}</strong>
        <span>/ 100</span>
      </div>
    </div>
  );
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
  const muted = label === "Background";
  return (
    <div className="attention-row">
      <span className="attention-label">{label}</span>
      <div className="meter" role="img" aria-label={`${label}: ${percent.toFixed(1)} percent`}>
        <div
          className={muted ? "meter-fill tone-muted" : "meter-fill tone-accent"}
          style={{ width: `${Math.max(1, percent)}%` }}
        />
      </div>
      <strong className="attention-value">{percent.toFixed(1)}%</strong>
    </div>
  );
}

/* --- Chat --------------------------------------------------------------- */

const STARTERS = [
  "What is my weakest component and how do I fix it?",
  "How do I make the CTA stand out more?",
  "Is the visual hierarchy working?",
];

function ChatPanel({ analysisId, chatReady, chatModel }) {
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const [chatError, setChatError] = useState("");
  const scrollRef = useRef(null);
  const inputRef = useRef(null);

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

  function useStarter(text) {
    setDraft(text);
    inputRef.current?.focus();
  }

  return (
    <div className="chat-panel">
      <div className="panel-heading">
        <MessageSquare size={18} />
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
                {STARTERS.map((text) => (
                  <button key={text} type="button" onClick={() => useStarter(text)}>
                    {text}
                  </button>
                ))}
              </div>
            )}

            {messages.map((message, index) => (
              <div key={index} className={`bubble bubble-${message.role}`}>
                {message.role === "assistant" ? renderMarkdown(message.content) : message.content}
              </div>
            ))}

            {sending && (
              <div className="bubble bubble-assistant bubble-pending">
                <Loader2 className="spin" size={15} /> Thinking
              </div>
            )}
          </div>

          {chatError && (
            <div className="error-box">
              <AlertCircle size={17} />
              <span>{chatError}</span>
            </div>
          )}

          <form className="chat-composer" onSubmit={send}>
            <input
              ref={inputRef}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              placeholder="Ask how to improve this poster"
              aria-label="Message the assistant"
            />
            <button className="secondary-button" type="submit" disabled={sending || !draft.trim()}>
              <Send size={16} />
              Send
            </button>
          </form>
        </>
      )}
    </div>
  );
}

/* --- App ---------------------------------------------------------------- */

function App() {
  const [poster, setPoster] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
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

  const weakest = useMemo(() => {
    const entries = Object.entries(result?.pes?.components ?? {});
    if (!entries.length) return null;
    return entries.sort((a, b) => a[1] - b[1])[0];
  }, [result]);

  const acceptFile = useCallback((file) => {
    setResult(null);
    setError("");
    setView("boxes");

    if (file && !ACCEPTED_TYPES.test(file.type)) {
      setPoster(null);
      setPreviewUrl((old) => {
        if (old) URL.revokeObjectURL(old);
        return "";
      });
      setError("That file is not a JPG, PNG, WEBP, or AVIF image.");
      return;
    }

    setPoster(file ?? null);
    setPreviewUrl((old) => {
      if (old) URL.revokeObjectURL(old);
      return file ? URL.createObjectURL(file) : "";
    });
  }, []);

  function handlePosterChange(event) {
    acceptFile(event.target.files?.[0]);
  }

  // The label only ever looked like a dropzone; these handlers make it behave
  // like one. A dropped File goes straight to state, so the hidden input stays
  // empty and is re-read on the next click.
  function handleDrop(event) {
    event.preventDefault();
    setDragging(false);
    acceptFile(event.dataTransfer.files?.[0]);
  }

  function clearPoster(event) {
    event.preventDefault();
    event.stopPropagation();
    acceptFile(null);
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
            <Eye size={15} />
            YOLOv8 + Saliency PES
          </div>
        </header>

        <div className="layout">
          <section className="panel uploader-panel">
            <div className="panel-heading">
              <ImageUp size={18} />
              <h2>Upload Poster</h2>
            </div>

            <label
              className={`dropzone${dragging ? " dragging" : ""}${previewUrl ? " filled" : ""}`}
              onDragOver={(event) => {
                event.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
            >
              <input type="file" accept="image/*" onChange={handlePosterChange} />
              {previewUrl ? (
                <>
                  <img src={previewUrl} alt="Uploaded poster preview" />
                  <button
                    type="button"
                    className="clear-button"
                    onClick={clearPoster}
                    aria-label="Remove poster"
                  >
                    <X size={15} />
                  </button>
                </>
              ) : (
                <span className="dropzone-hint">
                  <ImageUp size={26} />
                  <strong>Drop a poster here</strong>
                  JPG, PNG, WEBP, or AVIF &mdash; or click to browse
                </span>
              )}
            </label>

            {poster && (
              <p className="file-chip">
                <span title={poster.name}>{poster.name}</span>
                <span className="file-size">{formatBytes(poster.size)}</span>
              </p>
            )}

            <button className="primary-button" onClick={analyzePoster} disabled={loading}>
              {loading ? <Loader2 className="spin" size={17} /> : <Target size={17} />}
              {loading ? "Analyzing" : "Analyze Poster"}
            </button>

            {error && (
              <div className="error-box">
                <AlertCircle size={17} />
                <span>{error}</span>
              </div>
            )}
          </section>

          <section className="panel result-panel">
            <div className="panel-heading">
              <Target size={18} />
              <h2>Detection Output</h2>
              {result && (
                <div className="view-toggle">
                  <button
                    type="button"
                    className={view === "boxes" ? "active" : ""}
                    onClick={() => setView("boxes")}
                  >
                    <Target size={14} /> Boxes
                  </button>
                  <button
                    type="button"
                    className={view === "saliency" ? "active" : ""}
                    onClick={() => setView("saliency")}
                  >
                    <Flame size={14} /> Attention
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
              <div className="empty-state">
                {loading ? (
                  <span className="loading-state">
                    <Loader2 className="spin" size={22} />
                    Detecting elements and estimating attention
                  </span>
                ) : (
                  "Bounding boxes appear here after analysis."
                )}
              </div>
            )}
          </section>
        </div>

        {result && (
          <section className="insights-grid">
            <article className="score-panel">
              <p className="eyebrow">Poster Effectiveness Score</p>
              <ScoreRing score={score} tone={band.tone} />
              <span className={`band band-${band.tone}`}>{result.pes.category}</span>
              {weakest && (
                <p className="score-note">
                  Weakest: <strong>{weakest[0]}</strong> at {Math.round(weakest[1])}
                </p>
              )}
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

        {result && result.recommendations?.length > 0 && (
          <section className="panel recommendations-panel">
            <div className="panel-heading">
              <Lightbulb size={18} />
              <h2>How to improve this poster</h2>
            </div>
            <ol className="rec-list">
              {result.recommendations.map((item, index) => (
                <li key={item}>
                  <span className="rec-index">{index + 1}</span>
                  <span>{item}</span>
                </li>
              ))}
            </ol>
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
                  <div className="table-row table-head">
                    <span>Element</span>
                    <span>Conf.</span>
                    <span>Position</span>
                  </div>
                  {result.detections.map((item, index) => (
                    <div className="table-row" key={`${item.class_name}-${index}`}>
                      <span>{item.class_name}</span>
                      <span className="mono">{Math.round(item.confidence * 100)}%</span>
                      <span className="dim">{item.relative_position}</span>
                    </div>
                  ))}
                </div>
              )}

              <h2 className="stacked-heading">
                <Sparkles size={16} /> Findings
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
