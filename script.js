/* =========================================================================
   Quote Explorer
   - "Discover" tries the live Quotable API first, falls back to the
     locally trained/classified dataset if the API is slow or down.
   - Category filtering uses categories assigned by a Naive Bayes classifier
     trained locally (see /model/train_classifier.py), not tags handed to us
     by an API.
   - "Match my mood" converts free-text input into the SAME TF-IDF vector
     space the classifier was trained in, then ranks quotes by cosine
     similarity against that vector — a lightweight vector-search engine
     running entirely client-side.
   ========================================================================= */

const STATE = {
  dataset: null,          // { vocab, idf, quotes }
  vocabIndex: null,       // Map(term -> index)
  currentQuote: null,     // quote shown on Discover tab
  currentSource: null,    // 'live' | 'offline'
  favorites: [],
};

const FAVORITES_KEY = "quoteExplorer.favorites";
const API_URL = "https://api.quotable.io/quotes/random";
const API_TIMEOUT_MS = 4000;

/* ---------------------------- Bootstrapping ---------------------------- */

document.addEventListener("DOMContentLoaded", async () => {
  setupTabs();
  loadFavorites();
  renderFavorites();

  await loadDataset();

  document.getElementById("newQuoteBtn").addEventListener("click", showNewQuote);
  document.getElementById("categorySelect").addEventListener("change", showNewQuote);
  document.getElementById("favBtn").addEventListener("click", () => toggleFavorite(STATE.currentQuote, "favBtn"));
  document.getElementById("copyBtn").addEventListener("click", () => copyQuote(STATE.currentQuote));
  document.getElementById("twitterBtn").addEventListener("click", () => shareOnTwitter(STATE.currentQuote));

  document.getElementById("moodForm").addEventListener("submit", handleMoodSubmit);
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      document.getElementById("moodInput").value = chip.dataset.mood;
      handleMoodSubmit(new Event("submit"));
    });
  });

  showNewQuote();
});

async function loadDataset() {
  const res = await fetch("data/quotes-dataset.json");
  STATE.dataset = await res.json();
  STATE.vocabIndex = new Map(STATE.dataset.vocab.map((term, i) => [term, i]));
}

/* ------------------------------ Tabs UI --------------------------------- */

function setupTabs() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));

      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");
      document.getElementById(btn.dataset.tab).classList.add("active");
    });
  });
}

/* --------------------------- Discover tab ------------------------------- */

async function showNewQuote() {
  const category = document.getElementById("categorySelect").value;
  toggleSkeleton(true);

  let quote = null;
  let source = "offline";

  // Try the live API first, but never let it block the UI for long.
  try {
    quote = await withTimeout(fetchFromLiveApi(category), API_TIMEOUT_MS);
    source = "live";
  } catch (err) {
    quote = pickFromLocalDataset(category);
    source = "offline";
  }

  toggleSkeleton(false);
  renderQuote(quote, source);
}

function withTimeout(promise, ms) {
  return Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error("timeout")), ms)),
  ]);
}

async function fetchFromLiveApi(category) {
  const url = category === "all" ? API_URL : `${API_URL}?tags=${encodeURIComponent(category)}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error("API error");
  const data = await res.json();
  const raw = Array.isArray(data) ? data[0] : data;
  if (!raw || !raw.content) throw new Error("Empty API response");
  return { content: raw.content, author: raw.author, category, source: "live-api" };
}

function pickFromLocalDataset(category) {
  const pool =
    category === "all"
      ? STATE.dataset.quotes
      : STATE.dataset.quotes.filter((q) => q.category === category);
  const source = pool.length ? pool : STATE.dataset.quotes;
  return source[Math.floor(Math.random() * source.length)];
}

function toggleSkeleton(show) {
  document.getElementById("skeleton").hidden = !show;
  document.getElementById("quoteText").style.display = show ? "none" : "block";
  document.getElementById("quoteAuthor").style.display = show ? "none" : "block";
}

function renderQuote(quote, source) {
  STATE.currentQuote = quote;
  STATE.currentSource = source;

  document.getElementById("quoteText").textContent = quote.content;
  document.getElementById("quoteAuthor").textContent = `— ${quote.author}`;
  document.getElementById("offlineNotice").hidden = source !== "offline";

  const favBtn = document.getElementById("favBtn");
  const isFav = isFavorited(quote);
  favBtn.classList.toggle("active", isFav);
  favBtn.textContent = isFav ? "♥ Saved" : "♡ Save";
  favBtn.setAttribute("aria-pressed", String(isFav));
}

/* --------------------------- Mood matching -------------------------------
   Converts free-text mood input into the same TF-IDF vector space the
   classifier was trained on (using the saved vocabulary + IDF weights),
   then ranks all quotes by cosine similarity to find the closest match.
---------------------------------------------------------------------------*/

const STOPWORDS = new Set([
  "a","an","the","is","am","are","was","were","be","been","i","me","my",
  "to","of","in","on","for","and","or","it","this","that","about","with",
  "feel","feeling","feelings","really","just","so","very","today"
]);

function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z\s]/g, " ")
    .split(/\s+/)
    .filter((t) => t && !STOPWORDS.has(t));
}

function textToVector(text) {
  const tokens = tokenize(text);
  const termCounts = {};
  tokens.forEach((t) => (termCounts[t] = (termCounts[t] || 0) + 1));

  const vec = {};
  const totalTerms = tokens.length || 1;
  Object.entries(termCounts).forEach(([term, count]) => {
    const idx = STATE.vocabIndex.get(term);
    if (idx === undefined) return; // term not in training vocabulary
    const tf = count / totalTerms;
    const idf = STATE.dataset.idf[idx];
    vec[idx] = tf * idf;
  });
  return vec;
}

function cosineSimilarity(vecA, vecB) {
  let dot = 0, normA = 0, normB = 0;
  for (const k in vecA) { dot += vecA[k] * (vecB[k] || 0); normA += vecA[k] ** 2; }
  for (const k in vecB) { normB += vecB[k] ** 2; }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function findBestQuoteForMood(moodText) {
  const queryVec = textToVector(moodText);
  let best = null;
  let bestScore = -1;

  STATE.dataset.quotes.forEach((q) => {
    const score = cosineSimilarity(queryVec, q.vector);
    if (score > bestScore) {
      bestScore = score;
      best = q;
    }
  });

  // If nothing in the vocabulary matched at all (score 0), fall back to a
  // quote from a broadly relevant category rather than returning noise.
  if (bestScore <= 0) {
    const fallbackPool = STATE.dataset.quotes.filter((q) => q.category === "wisdom");
    best = fallbackPool[Math.floor(Math.random() * fallbackPool.length)];
    bestScore = 0;
  }

  return { quote: best, score: bestScore };
}

function explainMatch(moodText, quote, score) {
  const pct = Math.round(score * 100);
  if (score <= 0) {
    return `I couldn't find strong keyword overlap for "${moodText.trim()}", so here's a broadly relevant quote instead.`;
  }
  return `Matched to "${moodText.trim()}" by shared themes with your input (similarity score: ${pct}%), landing in the "${quote.category}" category.`;
}

function handleMoodSubmit(e) {
  e.preventDefault();
  const moodInput = document.getElementById("moodInput");
  const moodText = moodInput.value.trim();
  if (!moodText) return;

  const { quote, score } = findBestQuoteForMood(moodText);

  document.getElementById("moodResult").hidden = false;
  document.getElementById("moodQuoteText").textContent = quote.content;
  document.getElementById("moodQuoteAuthor").textContent = `— ${quote.author}`;
  document.getElementById("moodExplanation").textContent = explainMatch(moodText, quote, score);

  const favBtn = document.getElementById("moodFavBtn");
  const isFav = isFavorited(quote);
  favBtn.classList.toggle("active", isFav);
  favBtn.textContent = isFav ? "♥ Saved" : "♡ Save";

  favBtn.onclick = () => toggleFavorite(quote, "moodFavBtn");
  document.getElementById("moodCopyBtn").onclick = () => copyQuote(quote);
  document.getElementById("moodTwitterBtn").onclick = () => shareOnTwitter(quote);
}

/* ------------------------------ Favorites -------------------------------- */

function loadFavorites() {
  try {
    STATE.favorites = JSON.parse(localStorage.getItem(FAVORITES_KEY)) || [];
  } catch {
    STATE.favorites = [];
  }
}

function saveFavorites() {
  localStorage.setItem(FAVORITES_KEY, JSON.stringify(STATE.favorites));
  document.getElementById("favCount").textContent = STATE.favorites.length;
}

function isFavorited(quote) {
  return STATE.favorites.some((f) => f.content === quote.content);
}

function toggleFavorite(quote, btnId) {
  if (!quote) return;
  const idx = STATE.favorites.findIndex((f) => f.content === quote.content);
  if (idx >= 0) {
    STATE.favorites.splice(idx, 1);
  } else {
    STATE.favorites.push({ content: quote.content, author: quote.author, category: quote.category || "general" });
  }
  saveFavorites();
  renderFavorites();

  const btn = document.getElementById(btnId);
  const nowFav = isFavorited(quote);
  btn.classList.toggle("active", nowFav);
  btn.textContent = nowFav ? "♥ Saved" : "♡ Save";
  btn.setAttribute("aria-pressed", String(nowFav));
}

function renderFavorites() {
  const list = document.getElementById("favoritesList");
  const empty = document.getElementById("emptyFavorites");
  document.getElementById("favCount").textContent = STATE.favorites.length;

  list.innerHTML = "";
  if (STATE.favorites.length === 0) {
    empty.hidden = false;
    return;
  }
  empty.hidden = true;

  STATE.favorites.forEach((fav, i) => {
    const item = document.createElement("div");
    item.className = "favorite-item";
    item.innerHTML = `
      <button class="remove-fav" data-index="${i}" aria-label="Remove from favorites">✕ Remove</button>
      <p class="quote-text">${escapeHtml(fav.content)}</p>
      <p class="quote-author">— ${escapeHtml(fav.author)}</p>
    `;
    list.appendChild(item);
  });

  list.querySelectorAll(".remove-fav").forEach((btn) => {
    btn.addEventListener("click", () => {
      const i = Number(btn.dataset.index);
      STATE.favorites.splice(i, 1);
      saveFavorites();
      renderFavorites();
      // keep the Discover/Mood save buttons in sync if the same quote is shown
      if (STATE.currentQuote) renderQuote(STATE.currentQuote, STATE.currentSource);
    });
  });
}

/* --------------------------- Copy & Share -------------------------------- */

function copyQuote(quote) {
  if (!quote) return;
  const text = `"${quote.content}" — ${quote.author}`;
  navigator.clipboard.writeText(text).then(() => {
    flashCopied("copyBtn");
    flashCopied("moodCopyBtn");
  });
}

function flashCopied(id) {
  const btn = document.getElementById(id);
  if (!btn) return;
  const original = btn.textContent;
  btn.classList.add("copied");
  btn.textContent = "Copied";
  setTimeout(() => {
    btn.classList.remove("copied");
    btn.textContent = original;
  }, 1500);
}

function shareOnTwitter(quote) {
  if (!quote) return;
  const text = encodeURIComponent(`"${quote.content}" — ${quote.author}`);
  window.open(`https://twitter.com/intent/tweet?text=${text}`, "_blank", "noopener");
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
