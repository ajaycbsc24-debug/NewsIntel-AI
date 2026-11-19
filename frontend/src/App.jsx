
import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  Clock3,
  ExternalLink,
  FileSearch,
  LayoutDashboard,
  LoaderCircle,
  Menu,
  Newspaper,
  RefreshCw,
  Search,
  ShieldAlert,
  Sparkles,
  Trash2,
  X,
  Globe,
  CalendarDays,
  Info,
  TrendingUp,
} from "lucide-react";

import {
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

import "./App.css";

const API_URL = "http://127.0.0.1:8000";

const NEWS_CATEGORIES = [
  { label: "Technology", value: "technology" },
  { label: "Business", value: "business" },
  { label: "World", value: "world" },
  { label: "India", value: "nation" },
  { label: "Sports", value: "sports" },
  { label: "Health", value: "health" },
  { label: "Science", value: "science" },
  { label: "Entertainment", value: "entertainment" },
  { label: "General", value: "general" },
];

function formatConfidence(value) {
  if (value === null || value === undefined || value === "") {
    return "Unavailable";
  }

  const number = Number(value);

  if (!Number.isFinite(number)) return "Unavailable";

  return `${(number <= 1 ? number * 100 : number).toFixed(1)}%`;
}

function getConfidenceNumber(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }

  const number = Number(value);

  if (!Number.isFinite(number)) return null;

  return Math.min(100, Math.max(0, number <= 1 ? number * 100 : number));
}

function formatDate(value) {
  if (!value) return "Date unavailable";

  const date = new Date(value);

  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleString([], {
        dateStyle: "medium",
        timeStyle: "short",
      });
}

function normalizePrediction(value) {
  return String(value || "").trim().toUpperCase();
}

function App() {
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [result, setResult] = useState(null);

  const [selectedArticle, setSelectedArticle] = useState(null);
  const [history, setHistory] = useState([]);

  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(true);

  const [error, setError] = useState("");

  const [searchTerm, setSearchTerm] = useState("");
  const [filter, setFilter] = useState("ALL");

  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // LIVE NEWS STATE
  const [news, setNews] = useState([]);
  const [newsCategory, setNewsCategory] = useState("technology");
  const [newsQuery, setNewsQuery] = useState("");
  const [newsLoading, setNewsLoading] = useState(false);
  const [newsError, setNewsError] = useState("");
  const [newsCount, setNewsCount] = useState(0);
  const [newsLoaded, setNewsLoaded] = useState(false);

  // LOCAL AI STATE
  const [aiLoading, setAiLoading] = useState("");
  const [aiResult, setAiResult] = useState(null);
  const [aiError, setAiError] = useState("");
  const [aiQuestion, setAiQuestion] = useState("");

  // LOAD PREDICTION HISTORY
  async function loadHistory() {
    setHistoryLoading(true);

    try {
      const response = await fetch(`${API_URL}/history`);

      if (!response.ok) {
        throw new Error(`History API error: ${response.status}`);
      }

      const data = await response.json();

      const records = Array.isArray(data)
        ? data
        : Array.isArray(data.history)
          ? data.history
          : [];

      setHistory(records);
      setError("");
    } catch (err) {
      console.error("History loading error:", err);
      setError(err.message || "Unable to connect to the backend.");
    } finally {
      setHistoryLoading(false);
    }
  }

  // LOAD LIVE NEWS FROM GNEWS BACKEND
  async function loadNews(category = newsCategory, query = newsQuery) {
    setNewsLoading(true);
    setNewsError("");

    try {
      const params = new URLSearchParams({
        category,
        max_results: "10",
      });

      if (query.trim()) {
        params.set("query", query.trim());
      }

      const response = await fetch(`${API_URL}/news?${params.toString()}`);
      const data = await response.json();

      if (!response.ok) {
        const detail =
          typeof data.detail === "string"
            ? data.detail
            : data.detail?.message ||
              data.detail?.provider_error?.join(" ") ||
              "Unable to load news.";

        throw new Error(detail);
      }

      const articles = Array.isArray(data.articles)
        ? data.articles
        : [];

      setNews(articles);
      setNewsCount(Number(data.total_articles || articles.length));
      setNewsLoaded(true);
    } catch (err) {
      console.error("Live news loading error:", err);
      setNewsError(
        err.message ||
          "Could not load live news. Make sure the backend is running."
      );
    } finally {
      setNewsLoading(false);
    }
  }

  // INITIAL DATA LOAD
  useEffect(() => {
    loadHistory();
    loadNews("technology", "");
  }, []);

  // DASHBOARD STATISTICS
  const stats = useMemo(() => {
    const real = history.filter(
      (item) => normalizePrediction(item.prediction) === "REAL"
    ).length;

    const fake = history.filter(
      (item) => normalizePrediction(item.prediction) === "FAKE"
    ).length;

    return {
      total: history.length,
      real,
      fake,
      realPercent: history.length
        ? Math.round((real / history.length) * 100)
        : 0,
    };
  }, [history]);

  // ANALYTICS CHART
  const chartData = [
    { name: "Real", value: stats.real, color: "#31d6a3" },
    { name: "Fake", value: stats.fake, color: "#ff6685" },
  ];

  // SEARCH AND FILTER HISTORY
  const filteredHistory = useMemo(() => {
    const term = searchTerm.toLowerCase().trim();

    return history.filter((item) => {
      const prediction = normalizePrediction(item.prediction);

      const matchesFilter =
        filter === "ALL" || prediction === filter;

      const matchesSearch =
        !term ||
        String(item.title || "").toLowerCase().includes(term) ||
        prediction.toLowerCase().includes(term);

      return matchesFilter && matchesSearch;
    });
  }, [history, searchTerm, filter]);

  // ANALYZE NEWS ARTICLE USING ML MODEL
  async function handlePredict(event) {
    event.preventDefault();

    setError("");
    setResult(null);

    if (!title.trim() || !content.trim()) {
      setError("Please enter both a news headline and article content.");
      return;
    }

    if (content.trim().length < 30) {
      setError("Please enter at least 30 characters of article content.");
      return;
    }

    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          title: title.trim(),
          text: content.trim(),
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Prediction request failed."
        );
      }

      setResult(data);
      await loadHistory();
    } catch (err) {
      console.error("Prediction error:", err);

      setError(
        err.message ||
          "Could not connect to the backend. Check that your FastAPI server is running."
      );
    } finally {
      setLoading(false);
    }
  }

  // RUN LOCAL AI FEATURES USING OLLAMA
  async function handleAiAction(endpoint, feature) {
    setAiError("");
    setAiResult(null);

    if (!title.trim() || !content.trim()) {
      setAiError("Please enter a headline and article content first.");
      return;
    }

    if (content.trim().length < 30) {
      setAiError("Please enter at least 30 characters of article content.");
      return;
    }

    if (endpoint === "ask" && !aiQuestion.trim()) {
      setAiError("Please enter a question about this article.");
      return;
    }

    setAiLoading(endpoint);

    try {
      const payload = {
        title: title.trim(),
        text: content.trim(),
      };

      if (endpoint === "ask") {
        payload.question = aiQuestion.trim();
      }

      const response = await fetch(`${API_URL}/ai/${endpoint}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "AI request failed."
        );
      }

      setAiResult(data);
    } catch (err) {
      console.error(`${feature} error:`, err);

      setAiError(
        err.message || `Could not generate ${feature}.`
      );
    } finally {
      setAiLoading("");
    }
  }

  // LOAD A LIVE ARTICLE INTO THE ANALYZER
  function analyzeNewsArticle(article) {
    const articleTitle = article.title || "";
    const articleContent =
      article.content || article.description || "";

    setTitle(articleTitle);
    setContent(articleContent);

    setSelectedArticle({
      title: articleTitle,
      source: article.source || "Source unavailable",
      url: article.url || "",
      publishedAt: article.published_at || "",
    });

    setResult(null);
    setError("");
    setAiResult(null);
    setAiError("");
    setAiQuestion("");

    scrollToSection("analyzer");

    if (articleContent.trim().length < 30) {
      setError(
        "This news provider returned limited article text. Please add more article content before running the ML analysis."
      );
    }
  }

  // SEARCH LIVE NEWS
  function handleNewsSearch(event) {
    event.preventDefault();
    loadNews(newsCategory, newsQuery);
  }

  // CHANGE NEWS CATEGORY
  function handleCategoryChange(category) {
    setNewsCategory(category);
    setNewsQuery("");
    loadNews(category, "");
  }

  // CLEAR PREDICTION HISTORY
  async function clearHistory() {
    if (!history.length) return;

    const confirmed = window.confirm(
      "Are you sure you want to permanently clear all prediction history?"
    );

    if (!confirmed) return;

    try {
      const response = await fetch(`${API_URL}/history`, {
        method: "DELETE",
      });

      if (!response.ok) {
        throw new Error("Could not clear prediction history.");
      }

      setHistory([]);
      setResult(null);
      setError("");

      await loadHistory();
    } catch (err) {
      setError(err.message || "Unable to clear history.");
    }
  }

  // RESULT INFORMATION
  const latestPrediction = result
    ? normalizePrediction(result.prediction)
    : "";

  const isFake = latestPrediction === "FAKE";
  const isReal = latestPrediction === "REAL";

  const confidence = result
    ? formatConfidence(result.confidence)
    : "";

  const confidenceNumber = result
    ? getConfidenceNumber(result.confidence)
    : null;

  // NAVIGATION
  function scrollToSection(id) {
    document
      .getElementById(id)
      ?.scrollIntoView({ behavior: "smooth" });

    setMobileMenuOpen(false);
  }

  return (
    <div className="app-shell">
      {/* SIDEBAR */}
      <aside
        className={`sidebar ${
          mobileMenuOpen ? "sidebar-open" : ""
        }`}
      >
        <div className="brand">
          <div className="brand-icon">
            <BrainCircuit size={23} />
          </div>

          <div>
            <h2>
              News<span>Intel</span>
            </h2>
            <p>AI INTELLIGENCE PLATFORM</p>
          </div>

          <button
            className="mobile-close"
            onClick={() => setMobileMenuOpen(false)}
            aria-label="Close menu"
          >
            <X size={20} />
          </button>
        </div>

        <div className="sidebar-label">WORKSPACE</div>

        <nav className="sidebar-nav">
          <button
            className="nav-link active"
            onClick={() => scrollToSection("overview")}
          >
            <LayoutDashboard size={18} />
            Dashboard
          </button>

          <button
            className="nav-link"
            onClick={() => scrollToSection("live-news")}
          >
            <Globe size={18} />
            Live News
          </button>

          <button
            className="nav-link"
            onClick={() => scrollToSection("analyzer")}
          >
            <FileSearch size={18} />
            News Analyzer
          </button>

          <button
            className="nav-link"
            onClick={() => scrollToSection("analytics")}
          >
            <BarChart3 size={18} />
            Analytics
          </button>

          <button
            className="nav-link"
            onClick={() => scrollToSection("history")}
          >
            <Clock3 size={18} />
            Prediction History
          </button>
        </nav>

        <div className="sidebar-bottom">
          <div className="sidebar-status">
            <span className="status-dot" />
            <div>
              <strong>ML Engine</strong>
              <small>Local model interface</small>
            </div>
          </div>

          <p className="sidebar-version">NEWSINTEL · VERSION 1.2</p>
        </div>
      </aside>

      {mobileMenuOpen && (
        <button
          className="mobile-overlay"
          onClick={() => setMobileMenuOpen(false)}
          aria-label="Close navigation"
        />
      )}

      {/* MAIN CONTENT */}
      <main className="main-content">
        {/* TOPBAR */}
        <header className="topbar">
          <div className="topbar-left">
            <button
              className="mobile-menu"
              onClick={() => setMobileMenuOpen(true)}
              aria-label="Open menu"
            >
              <Menu size={21} />
            </button>

            <div>
              <div className="breadcrumb">
                Workspace <span>/</span> Dashboard
              </div>
              <h1>News Intelligence Dashboard</h1>
            </div>
          </div>

          <div className="topbar-right">
            <div className="engine-pill">
              <span className="status-dot" />
              MODEL ONLINE
            </div>

            <div className="avatar">NI</div>
          </div>
        </header>

        <div className="page-content">
          {/* WELCOME SECTION */}
          <section className="welcome-row" id="overview">
            <div>
              <p className="eyebrow">
                <Sparkles size={14} />
                YOUR NEWS ANALYSIS WORKSPACE
              </p>

              <h2>
                Understand the news.
                <br />
                <span>Question what you read.</span>
              </h2>

              <p className="welcome-description">
                Explore news from GNews, analyze articles using your
                machine learning model and review your prediction
                activity in one place.
              </p>
            </div>

            <div className="welcome-art">
              <div className="orbit orbit-one" />
              <div className="orbit orbit-two" />

              <div className="art-core">
                <BrainCircuit size={48} />
              </div>

              <span className="art-spark spark-one">✦</span>
              <span className="art-spark spark-two">✧</span>
            </div>
          </section>

          {/* ERROR MESSAGE */}
          {error && (
            <div className="alert-error">
              <ShieldAlert size={18} />
              <span>{error}</span>

              <button
                onClick={() => setError("")}
                aria-label="Dismiss error"
              >
                <X size={17} />
              </button>
            </div>
          )}

          {/* STATISTICS */}
          <section className="stats-grid">
            <div className="stat-card">
              <div className="stat-top">
                <span className="stat-icon purple">
                  <Activity size={19} />
                </span>
                <span className="stat-caption">ALL TIME</span>
              </div>

              <div className="stat-value">{stats.total}</div>
              <div className="stat-label">Articles analyzed</div>

              <div className="stat-foot">
                <span className="stat-neutral">
                  Your total prediction records
                </span>
              </div>
            </div>

            <div className="stat-card">
              <div className="stat-top">
                <span className="stat-icon green">
                  <CheckCircle2 size={19} />
                </span>
                <span className="stat-caption">MODEL OUTPUT</span>
              </div>

              <div className="stat-value">{stats.real}</div>
              <div className="stat-label">Predicted real</div>

              <div className="stat-foot">
                <ArrowUpRight size={14} className="green-text" />
                <span className="green-text">{stats.realPercent}%</span>
                <span className="stat-neutral">
                  of analyzed articles
                </span>
              </div>
            </div>

            <div className="stat-card">
              <div className="stat-top">
                <span className="stat-icon red">
                  <ShieldAlert size={19} />
                </span>
                <span className="stat-caption">MODEL OUTPUT</span>
              </div>

              <div className="stat-value">{stats.fake}</div>
              <div className="stat-label">Predicted fake</div>

              <div className="stat-foot">
                <ArrowDownRight size={14} className="red-text" />
                <span className="red-text">
                  {stats.total
                    ? Math.round((stats.fake / stats.total) * 100)
                    : 0}%
                </span>
                <span className="stat-neutral">
                  of analyzed articles
                </span>
              </div>
            </div>

            <div className="stat-card">
              <div className="stat-top">
                <span className="stat-icon blue">
                  <BrainCircuit size={19} />
                </span>
                <span className="stat-caption">SYSTEM</span>
              </div>

              <div className="stat-value stat-online">Online</div>
              <div className="stat-label">ML analysis engine</div>

              <div className="stat-foot">
                <span className="status-dot" />
                <span className="green-text">Connected interface</span>
              </div>
            </div>
          </section>

          {/* LIVE NEWS SECTION */}
          <section className="panel live-news-panel" id="live-news">
            <div className="live-news-heading">
              <div>
                <div className="section-kicker">
                  <Globe size={15} />
                  GNEWS · LIVE NEWS FEED
                </div>

                <h3>Explore the latest news</h3>

                <p>
                  Browse news articles and send selected articles to
                  your machine learning analyzer.
                </p>
              </div>

              <button
                className="refresh-button"
                onClick={() => loadNews(newsCategory, newsQuery)}
                disabled={newsLoading}
              >
                <RefreshCw
                  size={15}
                  className={newsLoading ? "spin" : ""}
                />
                Refresh news
              </button>
            </div>

            <div className="news-toolbar">
              <div className="news-categories">
                {NEWS_CATEGORIES.map((category) => (
                  <button
                    key={category.value}
                    className={
                      newsCategory === category.value
                        ? "news-category active-category"
                        : "news-category"
                    }
                    onClick={() =>
                      handleCategoryChange(category.value)
                    }
                    disabled={newsLoading}
                  >
                    {category.label}
                  </button>
                ))}
              </div>

              <form
                className="news-search"
                onSubmit={handleNewsSearch}
              >
                <Search size={17} />

                <input
                  type="search"
                  placeholder="Search news by keyword..."
                  value={newsQuery}
                  onChange={(event) =>
                    setNewsQuery(event.target.value)
                  }
                  maxLength={200}
                />

                <button type="submit" disabled={newsLoading}>
                  Search
                </button>
              </form>
            </div>

            <div className="news-feed-status">
              <span>
                <span className="status-dot" />
                {newsLoaded
                  ? `${news.length} articles loaded`
                  : "Waiting for news"}
              </span>

              {newsLoaded && (
                <span>
                  {newsCount.toLocaleString()} matching articles reported
                  by provider
                </span>
              )}
            </div>

            {newsError && (
              <div className="news-error">
                <ShieldAlert size={17} />
                <span>{newsError}</span>

                <button
                  onClick={() => loadNews(newsCategory, newsQuery)}
                  disabled={newsLoading}
                >
                  Try again
                </button>
              </div>
            )}

            {newsLoading ? (
              <div className="news-loading">
                <LoaderCircle size={28} className="spin" />
                <strong>Loading news articles...</strong>
                <span>Fetching articles from GNews.</span>
              </div>
            ) : news.length > 0 ? (
              <div className="news-grid">
                {news.map((article, index) => (
                  <article
                    className="news-card"
                    key={article.id || article.url || index}
                  >
                    <div className="news-image-wrap">
                      {article.image ? (
                        <img
                          src={article.image}
                          alt=""
                          className="news-image"
                          loading="lazy"
                          onError={(event) => {
                            event.currentTarget.style.display = "none";
                          }}
                        />
                      ) : (
                        <div className="news-image-placeholder">
                          <Newspaper size={35} />
                        </div>
                      )}

                      <span className="news-source-tag">
                        {article.source || "News source"}
                      </span>
                    </div>

                    <div className="news-card-body">
                      <div className="news-meta">
                        <span>
                          <CalendarDays size={12} />
                          {formatDate(article.published_at)}
                        </span>
                      </div>

                      <h4>{article.title}</h4>

                      <p className="news-description">
                        {article.description ||
                          article.content ||
                          "No article summary is available."}
                      </p>

                      <div className="news-card-actions">
                        <button
                          className="news-analyze-button"
                          onClick={() => analyzeNewsArticle(article)}
                        >
                          <BrainCircuit size={15} />
                          Analyze news
                        </button>

                        {article.url && (
                          <a
                            className="news-read-link"
                            href={article.url}
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            Read article
                            <ExternalLink size={13} />
                          </a>
                        )}
                      </div>
                    </div>
                  </article>
                ))}
              </div>
            ) : (
              <div className="news-empty">
                <div className="empty-chart-icon">
                  <Newspaper size={25} />
                </div>

                <strong>
                  {newsError
                    ? "News could not be loaded"
                    : "No articles found"}
                </strong>

                <span>
                  Try another category or search using a different keyword.
                </span>

                <button
                  className="refresh-button"
                  onClick={() => loadNews(newsCategory, "")}
                  disabled={newsLoading}
                >
                  <RefreshCw size={14} />
                  Load category news
                </button>
              </div>
            )}

            <div className="news-disclaimer">
              <ShieldAlert size={15} />
              <span>
                News articles are provided by GNews and may have publication
                delays. The ML prediction is a text classification result,
                not independent verification of an article.
              </span>
            </div>
          </section>

          {/* WORKSPACE */}
          <section className="workspace-grid">
            {/* NEWS ANALYZER */}
            <div className="panel analyzer-panel" id="analyzer">
              <div className="panel-heading">
                <div>
                  <div className="section-kicker">
                    <FileSearch size={15} />
                    ANALYSIS TOOL
                  </div>

                  <h3>Analyze a news article</h3>

                  <p>
                    Enter a headline and article text to run your ML model.
                  </p>
                </div>

                <div className="heading-icon">
                  <BrainCircuit size={22} />
                </div>
              </div>

              {selectedArticle && (
                <div className="selected-article-info">
                  <div className="selected-article-icon">
                    <Newspaper size={18} />
                  </div>

                  <div className="selected-article-details">
                    <strong>Article selected from Live News</strong>
                    <span>
                      Source: {selectedArticle.source}
                      {selectedArticle.publishedAt
                        ? ` · ${formatDate(selectedArticle.publishedAt)}`
                        : ""}
                    </span>

                    {selectedArticle.url && (
                      <a
                        href={selectedArticle.url}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        Open original article
                        <ExternalLink size={12} />
                      </a>
                    )}
                  </div>

                  <button
                    type="button"
                    className="remove-selected-article"
                    onClick={() => setSelectedArticle(null)}
                    aria-label="Remove selected article source"
                  >
                    <X size={16} />
                  </button>
                </div>
              )}

              {/* EXISTING ML PREDICTION FORM */}
              <form onSubmit={handlePredict}>
                <label htmlFor="news-title">NEWS HEADLINE</label>

                <input
                  id="news-title"
                  type="text"
                  placeholder="Enter the article headline..."
                  value={title}
                  onChange={(event) => {
                    setTitle(event.target.value);
                    setResult(null);
                    setSelectedArticle(null);
                    setAiResult(null);
                    setAiError("");
                  }}
                  maxLength={1000}
                />

                <div className="label-row">
                  <label htmlFor="news-content">ARTICLE CONTENT</label>
                  <span>
                    {content.length.toLocaleString()} / 20,000
                  </span>
                </div>

                <textarea
                  id="news-content"
                  placeholder="Paste the news article content here. Add enough context for the model to analyze..."
                  value={content}
                  onChange={(event) => {
                    setContent(event.target.value);
                    setResult(null);
                    setSelectedArticle(null);
                    setAiResult(null);
                    setAiError("");
                  }}
                  maxLength={20000}
                  rows={8}
                />

                <div className="form-bottom">
                  <span className="input-hint">
                    <ShieldAlert size={14} />
                    Minimum 30 characters
                  </span>

                  <button
                    className="analyze-button"
                    type="submit"
                    disabled={loading}
                  >
                    {loading ? (
                      <>
                        <LoaderCircle className="spin" size={17} />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Sparkles size={17} />
                        Analyze article
                        <ArrowUpRight size={16} />
                      </>
                    )}
                  </button>
                </div>
              </form>

              {/* LOCAL AI TOOLS */}
              <div className="ai-tools-panel">
                <div className="ai-tools-heading">
                  <BrainCircuit size={19} />
                  <h4>LOCAL AI TOOLS</h4>
                </div>

                <p className="ai-tools-description">
                  Use your local Qwen3 AI model to understand the article.
                  Enter an article above, then select an AI feature.
                </p>

                <div className="ai-tools-grid">
                  <button
                    type="button"
                    className="ai-tool-button"
                    onClick={() =>
                      handleAiAction("summarize", "Article Summary")
                    }
                    disabled={Boolean(aiLoading)}
                  >
                    <FileSearch size={15} />
                    Article Summary
                  </button>

                  <button
                    type="button"
                    className="ai-tool-button"
                    onClick={() =>
                      handleAiAction("key-points", "Key Points")
                    }
                    disabled={Boolean(aiLoading)}
                  >
                    <CheckCircle2 size={15} />
                    Key Points
                  </button>

                  <button
                    type="button"
                    className="ai-tool-button"
                    onClick={() =>
                      handleAiAction("explain", "Simple Explanation")
                    }
                    disabled={Boolean(aiLoading)}
                  >
                    <Sparkles size={15} />
                    Simple Explanation
                  </button>

                  <button
                    type="button"
                    className="ai-tool-button"
                    onClick={() =>
                      handleAiAction("entities", "Important Entities")
                    }
                    disabled={Boolean(aiLoading)}
                  >
                    <Globe size={15} />
                    Important Entities
                  </button>
                </div>

                {/* ASK AI */}
                <div className="ai-question-row">
                  <input
                    type="text"
                    placeholder="Ask a question about this article..."
                    value={aiQuestion}
                    onChange={(event) => setAiQuestion(event.target.value)}
                    maxLength={1000}
                    disabled={Boolean(aiLoading)}
                  />

                  <button
                    type="button"
                    className="ai-tool-button"
                    onClick={() =>
                      handleAiAction("ask", "Ask AI")
                    }
                    disabled={Boolean(aiLoading)}
                  >
                    <Sparkles size={15} />
                    Ask AI
                  </button>
                </div>

                {/* AI LOADING */}
                {aiLoading && (
                  <div className="ai-loading">
                    <LoaderCircle size={17} className="spin" />
                    <span>
                      {aiLoading === "summarize"
                        ? "Generating article summary..."
                        : aiLoading === "key-points"
                          ? "Extracting key points..."
                          : aiLoading === "explain"
                            ? "Preparing simple explanation..."
                            : aiLoading === "entities"
                              ? "Identifying important entities..."
                              : "Preparing your answer..."}
                    </span>
                  </div>
                )}

                {/* AI ERROR */}
                {aiError && (
                  <div className="ai-error">
                    <ShieldAlert size={16} />
                    <span>{aiError}</span>
                  </div>
                )}

                {/* AI RESULT */}
                {aiResult && (
                  <div className="ai-result-card">
                    <div className="ai-result-top">
                      <span>
                        <Sparkles size={14} />
                        {aiResult.feature || "AI RESULT"}
                      </span>

                      <span>
                        {aiResult.model || "Local AI"}
                      </span>
                    </div>

                    <div className="ai-result-content">
                      {typeof aiResult.result === "string"
                        ? aiResult.result
                        : JSON.stringify(aiResult.result, null, 2)}
                    </div>

                    {aiResult.disclaimer && (
                      <p className="ai-result-disclaimer">
                        <ShieldAlert size={13} />
                        {aiResult.disclaimer}
                      </p>
                    )}
                  </div>
                )}
              </div>

              {/* ADVANCED PREDICTION RESULT */}
              {result && (
                <div
                  className={`result-card ${
                    isFake ? "result-fake" : "result-real"
                  }`}
                >
                  <div className="result-topline">
                    <span className="result-label">
                      <Sparkles size={14} />
                      ANALYSIS COMPLETE
                    </span>

                    <span className="result-time">
                      {formatDate(result.date || result.created_at)}
                    </span>
                  </div>

                  <div className="result-main">
                    <div className="result-symbol">
                      {isFake ? (
                        <ShieldAlert size={27} />
                      ) : (
                        <CheckCircle2 size={27} />
                      )}
                    </div>

                    <div className="result-prediction-text">
                      <div className="result-small">MODEL PREDICTION</div>
                      <h4>{latestPrediction || "UNKNOWN"}</h4>
                      <span>
                        {isFake
                          ? "The model classified this text as FAKE."
                          : isReal
                            ? "The model classified this text as REAL."
                            : "The model returned an unrecognized result."}
                      </span>
                    </div>
                  </div>

                  <div className="advanced-confidence">
                    <div className="confidence-heading">
                      <div>
                        <span>MODEL CONFIDENCE SCORE</span>
                        <strong>{confidence}</strong>
                      </div>

                      <TrendingUp size={21} />
                    </div>

                    <div
                      className="advanced-confidence-track"
                      role="progressbar"
                      aria-label="Model confidence score"
                      aria-valuemin={0}
                      aria-valuemax={100}
                      aria-valuenow={confidenceNumber ?? 0}
                    >
                      <div
                        className="advanced-confidence-fill"
                        style={{
                          width:
                            confidenceNumber === null
                              ? "0%"
                              : `${confidenceNumber}%`,
                        }}
                      />
                    </div>

                    <div className="confidence-scale">
                      <span>0%</span>
                      <span>100%</span>
                    </div>

                    <p className="confidence-explanation">
                      This score represents the model's confidence in its
                      classification output. It is not the probability that
                      the article is factually true or false.
                    </p>
                  </div>

                  <div className="result-meaning">
                    <div className="result-meaning-icon">
                      <Info size={17} />
                    </div>

                    <div>
                      <strong>What does this result mean?</strong>
                      <p>
                        {isFake
                          ? "Your trained machine learning model found patterns in the submitted text that match its FAKE training category. This is a model classification, not proof that the story is false."
                          : isReal
                            ? "Your trained machine learning model found patterns in the submitted text that match its REAL training category. This is a model classification, not proof that the story is true."
                            : "The model returned a result that could not be recognized as REAL or FAKE. Review the backend response before interpreting this result."}
                      </p>
                    </div>
                  </div>

                  {selectedArticle && (
                    <div className="result-source-details">
                      <span>ARTICLE SOURCE INFORMATION</span>
                      <strong>{selectedArticle.source}</strong>

                      {selectedArticle.publishedAt && (
                        <small>
                          Published: {formatDate(selectedArticle.publishedAt)}
                        </small>
                      )}

                      {selectedArticle.url && (
                        <a
                          href={selectedArticle.url}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          View original article
                          <ExternalLink size={13} />
                        </a>
                      )}
                    </div>
                  )}

                  <div className="result-disclaimer">
                    <ShieldAlert size={17} />
                    <p>
                      {result.disclaimer ||
                        "This is a machine learning prediction, not a verified fact-check. Independently verify the article using reliable sources."}
                    </p>
                  </div>
                </div>
              )}
            </div>

            {/* ANALYTICS */}
            <div className="panel analytics-panel" id="analytics">
              <div className="panel-heading compact-heading">
                <div>
                  <div className="section-kicker">
                    <BarChart3 size={15} />
                    YOUR ACTIVITY
                  </div>

                  <h3>Prediction overview</h3>

                  <p>
                    Breakdown of your saved model predictions.
                  </p>
                </div>
              </div>

              {history.length ? (
                <>
                  <div className="chart-wrap">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={chartData}
                          dataKey="value"
                          nameKey="name"
                          cx="50%"
                          cy="50%"
                          innerRadius={65}
                          outerRadius={91}
                          paddingAngle={5}
                          stroke="none"
                        >
                          {chartData.map((entry) => (
                            <Cell
                              key={entry.name}
                              fill={entry.color}
                            />
                          ))}
                        </Pie>

                        <Tooltip
                          contentStyle={{
                            background: "#151a2a",
                            border: "1px solid #2b3248",
                            borderRadius: "10px",
                            color: "#fff",
                          }}
                        />
                      </PieChart>
                    </ResponsiveContainer>

                    <div className="chart-center">
                      <strong>{stats.total}</strong>
                      <span>TOTAL</span>
                    </div>
                  </div>

                  <div className="chart-legend">
                    <div>
                      <span className="legend-dot real-dot" />
                      Real
                      <strong>{stats.real}</strong>
                    </div>

                    <div>
                      <span className="legend-dot fake-dot" />
                      Fake
                      <strong>{stats.fake}</strong>
                    </div>
                  </div>
                </>
              ) : (
                <div className="empty-chart">
                  <div className="empty-chart-icon">
                    <BarChart3 size={25} />
                  </div>

                  <strong>No analytics yet</strong>

                  <span>
                    Analyze an article to see your prediction breakdown.
                  </span>
                </div>
              )}

              <div className="analytics-note">
                <Activity size={16} />
                <span>
                  Statistics are calculated from your saved prediction history.
                </span>
              </div>
            </div>
          </section>

          {/* PREDICTION HISTORY */}
          <section className="panel history-panel" id="history">
            <div className="history-heading">
              <div>
                <div className="section-kicker">
                  <Clock3 size={15} />
                  YOUR RECORDS
                </div>

                <h3>Prediction history</h3>

                <p>
                  Review and search articles previously analyzed.
                </p>
              </div>

              <button
                className="refresh-button"
                onClick={loadHistory}
                disabled={historyLoading}
              >
                <RefreshCw
                  size={15}
                  className={historyLoading ? "spin" : ""}
                />
                Refresh
              </button>
            </div>

            <div className="history-toolbar">
              <div className="search-box">
                <Search size={17} />

                <input
                  type="search"
                  placeholder="Search by headline or prediction..."
                  value={searchTerm}
                  onChange={(event) => setSearchTerm(event.target.value)}
                />
              </div>

              <div className="filter-tabs">
                {["ALL", "REAL", "FAKE"].map((item) => (
                  <button
                    key={item}
                    className={filter === item ? "filter-active" : ""}
                    onClick={() => setFilter(item)}
                  >
                    {item === "ALL"
                      ? "All"
                      : item === "REAL"
                        ? "Real"
                        : "Fake"}
                  </button>
                ))}
              </div>
            </div>

            {historyLoading ? (
              <div className="history-empty">
                <LoaderCircle className="spin" size={22} />
                Loading history...
              </div>
            ) : filteredHistory.length ? (
              <div className="history-list">
                {filteredHistory.map((item) => {
                  const prediction = normalizePrediction(item.prediction);
                  const fake = prediction === "FAKE";

                  return (
                    <div className="history-item" key={item.id}>
                      <div
                        className={`history-result-icon ${
                          fake ? "fake-icon" : "real-icon"
                        }`}
                      >
                        {fake ? (
                          <ShieldAlert size={18} />
                        ) : (
                          <CheckCircle2 size={18} />
                        )}
                      </div>

                      <div className="history-info">
                        <strong>
                          {item.title || "Untitled article"}
                        </strong>

                        <span>
                          <Clock3 size={13} />
                          {formatDate(item.created_at || item.date)}
                        </span>
                      </div>

                      <div className="history-meta">
                        <span
                          className={`prediction-badge ${
                            fake ? "badge-fake" : "badge-real"
                          }`}
                        >
                          {prediction || "UNKNOWN"}
                        </span>

                        <strong>
                          {formatConfidence(item.confidence)}
                        </strong>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="history-empty">
                <Newspaper size={25} />

                <strong>
                  {history.length
                    ? "No matching records"
                    : "No predictions saved yet"}
                </strong>

                <span>
                  {history.length
                    ? "Try another search or filter."
                    : "Your analyzed articles will appear here."}
                </span>
              </div>
            )}

            <div className="history-footer">
              <span>
                Showing {filteredHistory.length} of {history.length} records
              </span>

              <button
                className="clear-button"
                onClick={clearHistory}
                disabled={!history.length}
              >
                <Trash2 size={15} />
                Clear history
              </button>
            </div>
          </section>

          {/* DISCLAIMER */}
          <section className="notice-panel">
            <div className="notice-icon">
              <ShieldAlert size={19} />
            </div>

            <div>
              <strong>
                Important: AI predictions are not proof of truth
              </strong>

              <p>
                This platform classifies text using a machine learning model.
                Its prediction and confidence score do not establish whether
                a story is factually true. Check the original source and
                compare the report with multiple reliable news outlets.
              </p>
            </div>
          </section>

          {/* FOOTER */}
          <footer className="page-footer">
            <span>
              © {new Date().getFullYear()} NewsIntel · AI News Intelligence
              Platform
            </span>

            <span>
              <span className="status-dot" />
              Local ML system
            </span>
          </footer>
        </div>
      </main>
    </div>
  );
}

export default App;
