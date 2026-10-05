// Archiva AI Frontend Application Logic
const API_BASE = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
  ? "http://127.0.0.1:8000"
  : window.location.origin;

let currentMode = "ask"; // "ask" | "search"
let currentQuery = "";
let lastHelpfulChoice = null;

// DOM Elements
const askForm = document.getElementById("askForm");
const queryInput = document.getElementById("queryInput");
const submitBtn = document.getElementById("submitBtn");
const modeAskBtn = document.getElementById("modeAskBtn");
const modeSearchBtn = document.getElementById("modeSearchBtn");
const queryChips = document.getElementById("queryChips");
const loadingState = document.getElementById("loadingState");
const loadingMessage = document.getElementById("loadingMessage");
const resultsSection = document.getElementById("resultsSection");
const echoQueryText = document.getElementById("echoQueryText");
const answerCard = document.getElementById("answerCard");
const answerTitle = document.getElementById("answerTitle");
const answerContent = document.getElementById("answerContent");
const answerSourceCountBadge = document.getElementById("answerSourceCountBadge");
const sourceCardsGrid = document.getElementById("sourceCardsGrid");
const backendStatus = document.getElementById("backendStatus");
const docCountBadge = document.getElementById("docCountBadge");
const viewDocsBtn = document.getElementById("viewDocsBtn");
const docsModal = document.getElementById("docsModal");
const closeModalBtn = document.getElementById("closeModalBtn");
const modalOverlay = document.getElementById("modalOverlay");
const modalDocsList = document.getElementById("modalDocsList");

// Feedback Elements
const btnHelpfulYes = document.getElementById("btnHelpfulYes");
const btnHelpfulNo = document.getElementById("btnHelpfulNo");
const feedbackCommentBox = document.getElementById("feedbackCommentBox");
const feedbackCommentInput = document.getElementById("feedbackCommentInput");
const submitFeedbackBtn = document.getElementById("submitFeedbackBtn");
const feedbackNotice = document.getElementById("feedbackNotice");

// Initialize
document.addEventListener("DOMContentLoaded", () => {
  checkBackendHealth();
  fetchDocumentCount();
  setupEventListeners();
});

// Setup event listeners
function setupEventListeners() {
  // Mode selection
  modeAskBtn.addEventListener("click", () => setMode("ask"));
  modeSearchBtn.addEventListener("click", () => setMode("search"));

  // Form submission
  askForm.addEventListener("submit", handleFormSubmit);

  // Suggested queries chips
  queryChips.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const q = chip.getAttribute("data-query");
      queryInput.value = q;
      handleQueryExecution(q);
    });
  });

  // Feedback buttons
  btnHelpfulYes.addEventListener("click", () => handleFeedbackVote(true));
  btnHelpfulNo.addEventListener("click", () => handleFeedbackVote(false));
  submitFeedbackBtn.addEventListener("click", submitFullFeedback);

  // Knowledge base modal
  viewDocsBtn.addEventListener("click", openDocsModal);
  closeModalBtn.addEventListener("click", closeDocsModal);
  modalOverlay.addEventListener("click", closeDocsModal);

  // Keyboard shortcut: Shift+Enter or Enter in textarea
  queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      askForm.requestSubmit();
    }
  });
}

function setMode(mode) {
  currentMode = mode;
  if (mode === "ask") {
    modeAskBtn.classList.add("active");
    modeSearchBtn.classList.remove("active");
    submitBtn.querySelector(".btn-text").textContent = "Ask Archiva";
  } else {
    modeSearchBtn.classList.add("active");
    modeAskBtn.classList.remove("active");
    submitBtn.querySelector(".btn-text").textContent = "Search Vectors";
  }
}

// Health check
async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      backendStatus.classList.remove("offline");
      backendStatus.classList.add("online");
      backendStatus.querySelector(".status-text").textContent = "Backend Online";
    } else {
      throw new Error("Health check returned status " + res.status);
    }
  } catch (err) {
    backendStatus.classList.remove("online");
    backendStatus.classList.add("offline");
    backendStatus.querySelector(".status-text").textContent = "Backend Offline";
  }
}

// Fetch document count
async function fetchDocumentCount() {
  try {
    const res = await fetch(`${API_BASE}/documents`);
    if (res.ok) {
      const data = await res.json();
      if (docCountBadge && data.total !== undefined) {
        docCountBadge.textContent = data.total;
      }
    }
  } catch (err) {
    console.debug("Could not fetch documents count:", err);
  }
}

// Form submit handler
function handleFormSubmit(e) {
  e.preventDefault();
  const query = queryInput.value.trim();
  if (!query) return;
  handleQueryExecution(query);
}

// Main Query Execution
async function handleQueryExecution(query) {
  currentQuery = query;
  echoQueryText.textContent = query;
  
  // UI states
  resultsSection.classList.add("hidden");
  loadingState.classList.remove("hidden");
  submitBtn.disabled = true;
  resetFeedbackUI();

  if (currentMode === "ask") {
    loadingMessage.textContent = "Retrieving engineering knowledge & synthesizing grounded answer...";
    await executeAskQuery(query);
  } else {
    loadingMessage.textContent = "Searching vector embeddings with cosine similarity...";
    await executeSearchQuery(query);
  }

  loadingState.classList.add("hidden");
  resultsSection.classList.remove("hidden");
  submitBtn.disabled = false;
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

// Execute POST /ask
async function executeAskQuery(query) {
  try {
    const res = await fetch(`${API_BASE}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query })
    });

    if (!res.ok) {
      throw new Error(`API error: HTTP ${res.status}`);
    }

    const data = await res.json();

    // Render Answer
    answerCard.classList.remove("hidden");
    answerTitle.textContent = "Grounded Solution";
    answerContent.innerHTML = formatMarkdown(data.answer);
    const sourceCount = (data.sources && Array.isArray(data.sources)) ? data.sources.length : 0;
    answerSourceCountBadge.textContent = `${sourceCount} ${sourceCount === 1 ? 'Source' : 'Sources'} Cited`;

    // Render Sources
    renderSourceCards(data.sources);

  } catch (err) {
    console.error("Ask query failed:", err);
    answerCard.classList.remove("hidden");
    answerTitle.textContent = "Error";
    answerContent.innerHTML = `<p style="color: #f43f5e;">Failed to generate answer. Ensure backend server is running on ${API_BASE}.</p>`;
    sourceCardsGrid.innerHTML = "";
  }
}

// Execute POST /search
async function executeSearchQuery(query) {
  try {
    const res = await fetch(`${API_BASE}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: 6 })
    });

    if (!res.ok) {
      throw new Error(`API error: HTTP ${res.status}`);
    }

    const data = await res.json();

    // In search mode, focus on vector rankings
    answerCard.classList.remove("hidden");
    answerTitle.textContent = "Semantic Vector Search Results";
    answerContent.innerHTML = `<p>Retrieved <strong>${data.results.length}</strong> top matching engineering documents based on 384-dimensional cosine similarity ranking.</p>`;
    answerSourceCountBadge.textContent = `${data.results.length} Matches Found`;

    renderSourceCards(data.results);

  } catch (err) {
    console.error("Search query failed:", err);
    answerCard.classList.remove("hidden");
    answerTitle.textContent = "Search Error";
    answerContent.innerHTML = `<p style="color: #f43f5e;">Failed to perform vector search. Ensure backend server is running.</p>`;
    sourceCardsGrid.innerHTML = "";
  }
}

// Render source cards
function renderSourceCards(sources) {
  sourceCardsGrid.innerHTML = "";

  if (!sources || sources.length === 0) {
    sourceCardsGrid.innerHTML = `<p style="color: var(--text-muted); grid-column: 1/-1;">No matching sources found above the threshold.</p>`;
    return;
  }

  sources.forEach((item) => {
    const card = document.createElement("div");
    card.className = "source-card";

    const typeSlug = (item.document_type || "adr").toLowerCase().replace(/\s+/g, "-");
    const typeClass = `type-${typeSlug}`;

    card.innerHTML = `
      <div class="source-card-top">
        <div>
          <span class="source-doc-id">${escapeHtml(item.document_id || "")}</span>
          <h3 class="source-title">${escapeHtml(item.title || "Untitled")}</h3>
        </div>
        <span class="score-badge">Similarity: ${(item.score !== undefined ? (item.score * 100).toFixed(1) + "%" : "N/A")}</span>
      </div>
      <div>
        <span class="type-badge ${typeClass}">${escapeHtml(item.document_type || "Doc")}</span>
      </div>
      ${item.content ? `<div class="source-content">${escapeHtml(item.content)}</div>` : ""}
      <div class="source-meta">
        <span><strong>Team:</strong> ${escapeHtml(item.team || "Engineering")}</span>
        <span>&bull;</span>
        <span><strong>Service:</strong> ${escapeHtml(item.service || "Core")}</span>
      </div>
    `;

    sourceCardsGrid.appendChild(card);
  });
}

// Simple Markdown Formatter for RAG answers
function formatMarkdown(text) {
  if (!text) return "";
  
  let formatted = escapeHtml(text);

  // Bold text: **text**
  formatted = formatted.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

  // Bullet items: • or -
  const lines = formatted.split("\n");
  let inList = false;
  let htmlLines = [];

  for (let line of lines) {
    line = line.trim();
    if (line.startsWith("• ") || line.startsWith("- ")) {
      if (!inList) {
        htmlLines.push("<ul>");
        inList = true;
      }
      htmlLines.push(`<li>${line.substring(2)}</li>`);
    } else {
      if (inList) {
        htmlLines.push("</ul>");
        inList = false;
      }
      if (line) {
        htmlLines.push(`<p>${line}</p>`);
      }
    }
  }

  if (inList) {
    htmlLines.push("</ul>");
  }

  return htmlLines.join("");
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// Feedback Handling
function handleFeedbackVote(isHelpful) {
  lastHelpfulChoice = isHelpful;
  btnHelpfulYes.classList.toggle("selected", isHelpful === true);
  btnHelpfulNo.classList.toggle("selected", isHelpful === false);

  feedbackCommentBox.classList.remove("hidden");
  feedbackNotice.classList.add("hidden");
}

async function submitFullFeedback() {
  const comment = feedbackCommentInput.value.trim();
  const helpful = lastHelpfulChoice !== null ? lastHelpfulChoice : true;

  try {
    const res = await fetch(`${API_BASE}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: currentQuery,
        helpful: helpful,
        comment: comment
      })
    });

    if (res.ok) {
      feedbackNotice.textContent = "✓ Thank you! Feedback recorded.";
      feedbackNotice.classList.remove("hidden");
      feedbackCommentBox.classList.add("hidden");
      feedbackCommentInput.value = "";
    }
  } catch (err) {
    console.error("Failed to submit feedback:", err);
  }
}

function resetFeedbackUI() {
  lastHelpfulChoice = null;
  btnHelpfulYes.classList.remove("selected");
  btnHelpfulNo.classList.remove("selected");
  feedbackCommentBox.classList.add("hidden");
  feedbackNotice.classList.add("hidden");
}

// Knowledge Base Modal Logic
async function openDocsModal() {
  docsModal.classList.remove("hidden");
  modalDocsList.innerHTML = "<p style='color: var(--text-secondary);'>Loading repository documents...</p>";

  try {
    const res = await fetch(`${API_BASE}/documents`);
    if (!res.ok) throw new Error("Failed to load documents");
    const data = await res.json();

    modalDocsList.innerHTML = "";
    data.documents.forEach((doc) => {
      const typeSlug = (doc.document_type || "adr").toLowerCase().replace(/\s+/g, "-");
      const typeClass = `type-${typeSlug}`;

      const card = document.createElement("div");
      card.className = "source-card";
      card.innerHTML = `
        <div class="source-card-top">
          <div>
            <span class="source-doc-id">${escapeHtml(doc.document_id)}</span>
            <h3 class="source-title">${escapeHtml(doc.title)}</h3>
          </div>
          <span class="type-badge ${typeClass}">${escapeHtml(doc.document_type)}</span>
        </div>
        <div class="source-content">${escapeHtml(doc.content)}</div>
        <div class="source-meta">
          <span><strong>Team:</strong> ${escapeHtml(doc.team)}</span>
          <span>&bull;</span>
          <span><strong>Service:</strong> ${escapeHtml(doc.service)}</span>
        </div>
      `;
      modalDocsList.appendChild(card);
    });
  } catch (err) {
    modalDocsList.innerHTML = `<p style="color: #f43f5e;">Could not load documents: ${err.message}</p>`;
  }
}

function closeDocsModal() {
  docsModal.classList.add("hidden");
}
