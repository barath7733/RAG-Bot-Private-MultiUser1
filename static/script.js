// ----------------------------------------------------------------------
// Frontend logic for the AI Assistant + RAG Document Intelligence app.
// No API keys or secrets ever live in this file.
// Every privileged call goes through the backend.
// ----------------------------------------------------------------------

const API = {
  chat: "/api/chat",
  upload: "/api/documents/upload",
  documents: "/api/documents",
  health: "/api/health",
  me: "/api/auth/me",
  logout: "/api/auth/logout",
  chatHistory: "/api/chat-history",
  visionAnalyze: "/api/vision/analyze",
};

const state = {
  mode: "auto",
  history: [],
  documents: [],
  sessionId: null,
  sessions: [],
  pendingImage: null,
};

const el = {
  chatWindow: document.getElementById("chat-window"),
  chatForm: document.getElementById("chat-form"),
  chatInput: document.getElementById("chat-input"),
  sendBtn: document.getElementById("send-btn"),
  loadingIndicator: document.getElementById("loading-indicator"),
  modeToggle: document.getElementById("mode-toggle"),
  fileInput: document.getElementById("file-input"),
  uploadStatus: document.getElementById("upload-status"),
  documentList: document.getElementById("document-list"),
  clearChatBtn: document.getElementById("clear-chat-btn"),
  healthIndicator: document.getElementById("health-indicator"),
  logoutBtn: document.getElementById("logout-btn"),
  newChatBtn: document.getElementById("new-chat-btn"),
  historyList: document.getElementById("history-list"),
  attachImageBtn: document.getElementById("attach-image-btn"),
  visionFileInput: document.getElementById("vision-file-input"),
  imageAttachPreview: document.getElementById("image-attach-preview"),
  imageAttachThumb: document.getElementById("image-attach-thumb"),
  imageAttachName: document.getElementById("image-attach-name"),
  imageAttachRemove: document.getElementById("image-attach-remove"),
};

// ----------------------------------------------------------------------
// Auth guard
// ----------------------------------------------------------------------

function handleUnauthorized(response) {
  if (response.status === 401) {
    window.location.href = "/login";
    return true;
  }

  return false;
}

if (el.logoutBtn) {
  el.logoutBtn.addEventListener("click", async () => {
    try {
      await fetch(API.logout, {
        method: "POST",
        credentials: "same-origin",
      });
    } finally {
      window.location.href = "/login";
    }
  });
}

// ----------------------------------------------------------------------
// Mode selector
// ----------------------------------------------------------------------

if (el.modeToggle) {
  el.modeToggle.addEventListener("click", (e) => {
    const btn = e.target.closest(".mode-btn");

    if (!btn) return;

    document.querySelectorAll(".mode-btn").forEach((b) => {
      b.classList.remove("active");
    });

    btn.classList.add("active");
    state.mode = btn.dataset.mode;
  });
}

// ----------------------------------------------------------------------
// Chat input
// ----------------------------------------------------------------------

function autoResizeTextarea() {
  if (!el.chatInput) return;

  el.chatInput.style.height = "auto";

  el.chatInput.style.height =
    Math.min(el.chatInput.scrollHeight, 160) + "px";
}

if (el.chatInput) {
  el.chatInput.addEventListener("input", autoResizeTextarea);

  el.chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();

      if (el.chatForm) {
        el.chatForm.requestSubmit();
      }
    }
  });
}

// ----------------------------------------------------------------------
// Welcome message
// ----------------------------------------------------------------------

function clearWelcomeMessage() {
  const welcome = el.chatWindow?.querySelector(".welcome-message");

  if (welcome) {
    welcome.remove();
  }
}

// ----------------------------------------------------------------------
// Mode labels
// ----------------------------------------------------------------------

const MODE_LABELS = {
  rag: "📄 Document Mode",
  web: "🌐 Web Search",
  general: "💬 General AI",
  vision: "🖼️ Image Analysis",
};

// ----------------------------------------------------------------------
// Markdown rendering
// ----------------------------------------------------------------------

function renderMarkdown(text) {
  if (!text) return "";

  const escaped = escapeHtml(String(text));
  const lines = escaped.split("\n");

  const htmlParts = [];

  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // --------------------------------------------------------------
    // Markdown table
    // --------------------------------------------------------------

    if (
      line.trim().startsWith("|") &&
      i + 1 < lines.length &&
      isMarkdownTableSeparator(lines[i + 1])
    ) {
      const headerCells = parseTableRow(line);

      let tableHtml =
        '<div class="table-wrapper">' +
        '<table class="md-table">' +
        "<thead><tr>";

      headerCells.forEach((cell) => {
        tableHtml += `<th>${inlineMarkdown(cell)}</th>`;
      });

      tableHtml += "</tr></thead><tbody>";

      i += 2;

      while (
        i < lines.length &&
        lines[i].trim().startsWith("|")
      ) {
        const rowCells = parseTableRow(lines[i]);

        tableHtml += "<tr>";

        rowCells.forEach((cell) => {
          tableHtml += `<td>${inlineMarkdown(cell)}</td>`;
        });

        tableHtml += "</tr>";

        i++;
      }

      tableHtml +=
        "</tbody></table></div>";

      htmlParts.push(tableHtml);

      continue;
    }

    // --------------------------------------------------------------
    // Headings
    // --------------------------------------------------------------

    const headingMatch = line.match(/^(#{1,6})\s+(.+)$/);

    if (headingMatch) {
      const level = Math.min(headingMatch[1].length, 6);

      htmlParts.push(
        `<h${level} class="md-heading">${inlineMarkdown(
          headingMatch[2]
        )}</h${level}>`
      );

      i++;
      continue;
    }

    // --------------------------------------------------------------
    // Bullet list
    // --------------------------------------------------------------

    if (/^\s*[-*+]\s+/.test(line)) {
      let listHtml = '<ul class="md-list">';

      while (
        i < lines.length &&
        /^\s*[-*+]\s+/.test(lines[i])
      ) {
        const itemText = lines[i].replace(
          /^\s*[-*+]\s+/,
          ""
        );

        listHtml += `<li>${inlineMarkdown(itemText)}</li>`;

        i++;
      }

      listHtml += "</ul>";

      htmlParts.push(listHtml);

      continue;
    }

    // --------------------------------------------------------------
    // Numbered list
    // --------------------------------------------------------------

    if (/^\s*\d+[.)]\s+/.test(line)) {
      let listHtml = '<ol class="md-list">';

      while (
        i < lines.length &&
        /^\s*\d+[.)]\s+/.test(lines[i])
      ) {
        const itemText = lines[i].replace(
          /^\s*\d+[.)]\s+/,
          ""
        );

        listHtml += `<li>${inlineMarkdown(itemText)}</li>`;

        i++;
      }

      listHtml += "</ol>";

      htmlParts.push(listHtml);

      continue;
    }

    // --------------------------------------------------------------
    // Blockquote
    // --------------------------------------------------------------

    if (/^\s*>\s?/.test(line)) {
      let quoteHtml = '<blockquote class="md-quote">';

      while (
        i < lines.length &&
        /^\s*>\s?/.test(lines[i])
      ) {
        const quoteText = lines[i].replace(
          /^\s*>\s?/,
          ""
        );

        quoteHtml += `<p>${inlineMarkdown(quoteText)}</p>`;

        i++;
      }

      quoteHtml += "</blockquote>";

      htmlParts.push(quoteHtml);

      continue;
    }

    // --------------------------------------------------------------
    // Horizontal rule
    // --------------------------------------------------------------

    if (
      /^\s*([-*_])(?:\s*\1){2,}\s*$/.test(line)
    ) {
      htmlParts.push('<hr class="md-hr">');

      i++;
      continue;
    }

    // --------------------------------------------------------------
    // Blank line
    // --------------------------------------------------------------

    if (line.trim() === "") {
      i++;
      continue;
    }

    // --------------------------------------------------------------
    // Regular paragraph
    // --------------------------------------------------------------

    htmlParts.push(
      `<p class="md-p">${inlineMarkdown(line)}</p>`
    );

    i++;
  }

  return htmlParts.join("");
}

// ----------------------------------------------------------------------
// Markdown helpers
// ----------------------------------------------------------------------

function isMarkdownTableSeparator(line) {
  const cleaned = line.trim();

  if (!cleaned.startsWith("|")) {
    return false;
  }

  const cells = parseTableRow(cleaned);

  if (cells.length === 0) {
    return false;
  }

  return cells.every((cell) => {
    return /^:?-{3,}:?$/.test(cell.trim());
  });
}

function parseTableRow(line) {
  let cleaned = line.trim();

  if (cleaned.startsWith("|")) {
    cleaned = cleaned.substring(1);
  }

  if (cleaned.endsWith("|")) {
    cleaned = cleaned.substring(0, cleaned.length - 1);
  }

  return cleaned
    .split("|")
    .map((cell) => cell.trim());
}

function inlineMarkdown(text) {
  let result = text;

  // Inline code
  result = result.replace(
    /`([^`]+)`/g,
    "<code>$1</code>"
  );

  // Bold
  result = result.replace(
    /\*\*([^*]+)\*\*/g,
    "<strong>$1</strong>"
  );

  // Bold using __
  result = result.replace(
    /__([^_]+)__/g,
    "<strong>$1</strong>"
  );

  // Italic
  result = result.replace(
    /(^|[^\*])\*([^*\n]+)\*(?!\*)/g,
    "$1<em>$2</em>"
  );

  // Italic using _
  result = result.replace(
    /(^|[^_])_([^_\n]+)_(?!_)/g,
    "$1<em>$2</em>"
  );

  // Strikethrough
  result = result.replace(
    /~~([^~]+)~~/g,
    "<del>$1</del>"
  );

  // Links
  result = result.replace(
    /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
  );

  return result;
}

// ----------------------------------------------------------------------
// Message rendering
// ----------------------------------------------------------------------

function appendMessage({
  role,
  content,
  modeUsed,
  sources,
  webSources,
  isError,
}) {
  clearWelcomeMessage();

  const wrapper = document.createElement("div");

  wrapper.className = `message ${role}`;

  // --------------------------------------------------------------
  // Mode tag
  // --------------------------------------------------------------

  if (role === "assistant" && modeUsed) {
    const tag = document.createElement("div");

    tag.className = "mode-tag";

    tag.textContent =
      MODE_LABELS[modeUsed] || "💬 General AI";

    wrapper.appendChild(tag);
  }

  // --------------------------------------------------------------
  // Message bubble
  // --------------------------------------------------------------

  const bubble = document.createElement("div");

  bubble.className =
    "bubble" + (isError ? " error-bubble" : "");

  if (role === "assistant" && !isError) {
    bubble.innerHTML = renderMarkdown(content);
  } else {
    bubble.textContent = content || "";
  }

  wrapper.appendChild(bubble);

  // --------------------------------------------------------------
  // Document sources
  // --------------------------------------------------------------

  if (Array.isArray(sources) && sources.length > 0) {
    const box = document.createElement("div");

    box.className = "sources-box";

    const title = document.createElement("strong");

    title.textContent = "Sources:";

    box.appendChild(title);

    sources.forEach((s) => {
      const item = document.createElement("div");

      item.className = "source-item";

      const sourceTitle = document.createElement("div");

      sourceTitle.className = "source-title";

      const pageInfo = s.page
        ? `, page ${s.page}`
        : "";

      const score =
        typeof s.score === "number"
          ? ` · score ${s.score.toFixed(2)}`
          : "";

      sourceTitle.textContent =
        `${s.document_name || "Document"}${pageInfo}${score}`;

      const snippet = document.createElement("div");

      snippet.textContent = s.snippet || "";

      item.appendChild(sourceTitle);
      item.appendChild(snippet);

      box.appendChild(item);
    });

    wrapper.appendChild(box);
  }

  // --------------------------------------------------------------
  // Web sources
  // --------------------------------------------------------------

  if (
    Array.isArray(webSources) &&
    webSources.length > 0
  ) {
    const box = document.createElement("div");

    box.className = "sources-box";

    const title = document.createElement("strong");

    title.textContent = "Sources:";

    box.appendChild(title);

    webSources.forEach((s) => {
      const item = document.createElement("div");

      item.className = "source-item";

      const link = document.createElement("a");

      link.href = s.url || "#";
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.className = "source-title";
      link.textContent = s.title || s.url || "Web source";

      const snippet = document.createElement("div");

      snippet.textContent = s.snippet || "";

      item.appendChild(link);
      item.appendChild(snippet);

      box.appendChild(item);
    });

    wrapper.appendChild(box);
  }

  el.chatWindow.appendChild(wrapper);

  el.chatWindow.scrollTop =
    el.chatWindow.scrollHeight;
}

// ----------------------------------------------------------------------
// HTML escaping
// ----------------------------------------------------------------------

function escapeHtml(str) {
  const div = document.createElement("div");

  div.textContent = String(str ?? "");

  return div.innerHTML;
}

// ----------------------------------------------------------------------
// Loading
// ----------------------------------------------------------------------

function setLoading(isLoading) {
  if (el.loadingIndicator) {
    el.loadingIndicator.classList.toggle(
      "hidden",
      !isLoading
    );
  }

  if (el.sendBtn) {
    el.sendBtn.disabled = isLoading;
  }

  if (isLoading && el.chatWindow) {
    el.chatWindow.scrollTop =
      el.chatWindow.scrollHeight;
  }
}

// ----------------------------------------------------------------------
// Image attachment / Vision
// ----------------------------------------------------------------------

if (el.attachImageBtn && el.visionFileInput) {
  el.attachImageBtn.addEventListener(
    "click",
    () => {
      el.visionFileInput.click();
    }
  );

  el.visionFileInput.addEventListener(
    "change",
    () => {
      const file =
        el.visionFileInput.files?.[0];

      if (!file) return;

      const allowed = [
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/gif",
      ];

      if (!allowed.includes(file.type)) {
        alert(
          "Only PNG, JPEG, WEBP, or GIF images are supported."
        );

        el.visionFileInput.value = "";

        return;
      }

      if (file.size > 10 * 1024 * 1024) {
        alert(
          "Image exceeds the maximum allowed size of 10 MB."
        );

        el.visionFileInput.value = "";

        return;
      }

      state.pendingImage = file;

      el.imageAttachThumb.src =
        URL.createObjectURL(file);

      el.imageAttachName.textContent =
        file.name;

      el.imageAttachPreview.classList.remove(
        "hidden"
      );

      el.chatInput.placeholder =
        "Ask something about this image (optional)...";

      el.chatInput.focus();
    }
  );
}

function clearPendingImage() {
  if (state.pendingImage) {
    state.pendingImage = null;
  }

  if (el.visionFileInput) {
    el.visionFileInput.value = "";
  }

  if (el.imageAttachPreview) {
    el.imageAttachPreview.classList.add(
      "hidden"
    );
  }

  if (el.imageAttachThumb) {
    el.imageAttachThumb.src = "";
  }

  if (el.chatInput) {
    el.chatInput.placeholder =
      "Ask a question, or attach an image to ask about it...";
  }
}

if (el.imageAttachRemove) {
  el.imageAttachRemove.addEventListener(
    "click",
    clearPendingImage
  );
}

async function submitImageAnalysis() {
  const file = state.pendingImage;

  if (!file) return;

  const question =
    el.chatInput.value.trim();

  clearWelcomeMessage();

  appendMessage({
    role: "user",
    content:
      question ||
      "🖼️ (image attached)",
  });

  state.history.push({
    role: "user",
    content:
      question ||
      "[Attached an image]",
  });

  el.chatInput.value = "";

  autoResizeTextarea();

  const thumbUrl =
    URL.createObjectURL(file);

  clearPendingImage();

  setLoading(true);

  const formData = new FormData();

  formData.append("file", file);
  formData.append("question", question);

  if (state.sessionId) {
    formData.append(
      "session_id",
      state.sessionId
    );
  }

  try {
    const response = await fetch(
      API.visionAnalyze,
      {
        method: "POST",
        body: formData,
        credentials: "same-origin",
      }
    );

    if (handleUnauthorized(response)) {
      return;
    }

    const data = await safeJson(response);

    if (!response.ok) {
      throw new Error(
        data.error ||
          data.detail ||
          "Image analysis failed."
      );
    }

    state.sessionId =
      data.session_id;

    appendMessage({
      role: "assistant",
      content: data.answer,
      modeUsed: "vision",
    });

    state.history.push({
      role: "assistant",
      content: data.answer,
    });

    await loadChatHistory();
  } catch (err) {
    appendMessage({
      role: "assistant",
      content:
        `Something went wrong: ${err.message}`,
      isError: true,
    });
  } finally {
    setLoading(false);

    URL.revokeObjectURL(thumbUrl);
  }
}

// ----------------------------------------------------------------------
// Chat submit
// ----------------------------------------------------------------------

if (el.chatForm) {
  el.chatForm.addEventListener(
    "submit",
    async (e) => {
      e.preventDefault();

      if (state.pendingImage) {
        await submitImageAnalysis();
        return;
      }

      const question =
        el.chatInput.value.trim();

      if (!question) return;

      appendMessage({
        role: "user",
        content: question,
      });

      state.history.push({
        role: "user",
        content: question,
      });

      el.chatInput.value = "";

      autoResizeTextarea();

      setLoading(true);

      try {
        const response = await fetch(
          API.chat,
          {
            method: "POST",
            headers: {
              "Content-Type":
                "application/json",
            },
            credentials:
              "same-origin",
            body: JSON.stringify({
              question,
              mode: state.mode,
              history:
                state.history
                  .slice(0, -1)
                  .slice(-10),
              session_id:
                state.sessionId,
            }),
          }
        );

        if (handleUnauthorized(response)) {
          return;
        }

        const data =
          await safeJson(response);

        if (!response.ok) {
          throw new Error(
            data.error ||
              data.detail ||
              "The request failed."
          );
        }

        state.sessionId =
          data.session_id;

        appendMessage({
          role: "assistant",
          content: data.answer,
          modeUsed: data.mode_used,
          sources: data.sources,
          webSources:
            data.web_sources,
        });

        state.history.push({
          role: "assistant",
          content: data.answer,
        });

        await loadChatHistory();
      } catch (err) {
        appendMessage({
          role: "assistant",
          content:
            `Something went wrong: ${err.message}`,
          isError: true,
        });
      } finally {
        setLoading(false);
      }
    }
  );
}

// ----------------------------------------------------------------------
// Safe JSON response
// ----------------------------------------------------------------------

async function safeJson(response) {
  try {
    return await response.json();
  } catch {
    return {
      error:
        "The server returned an invalid response.",
    };
  }
}

// ----------------------------------------------------------------------
// New Chat
// ----------------------------------------------------------------------

function startNewChat() {
  state.history = [];
  state.sessionId = null;

  el.chatWindow.innerHTML = `
    <div class="welcome-message">
      <h2>How can I help you today?</h2>
      <p>
        Ask me anything, or upload a PDF and ask questions about it.
      </p>
    </div>
  `;

  renderHistoryList();
}

if (el.clearChatBtn) {
  el.clearChatBtn.addEventListener(
    "click",
    startNewChat
  );
}

if (el.newChatBtn) {
  el.newChatBtn.addEventListener(
    "click",
    startNewChat
  );
}

// ----------------------------------------------------------------------
// Chat history
// ----------------------------------------------------------------------

async function loadChatHistory() {
  try {
    const response =
      await fetch(
        API.chatHistory,
        {
          credentials:
            "same-origin",
        }
      );

    if (handleUnauthorized(response)) {
      return;
    }

    const sessions =
      await safeJson(response);

    if (!Array.isArray(sessions)) {
      state.sessions = [];
    } else {
      state.sessions = sessions;
    }

    renderHistoryList();
  } catch (err) {
    console.error(
      "Failed to load chat history",
      err
    );
  }
}

function renderHistoryList() {
  if (
    !state.sessions ||
    state.sessions.length === 0
  ) {
    el.historyList.innerHTML =
      `<p class="empty-hint">No chats yet.</p>`;

    return;
  }

  el.historyList.innerHTML = "";

  state.sessions.forEach(
    (session) => {
      const item =
        document.createElement("div");

      item.className =
        "history-item" +
        (
          session.session_id ===
          state.sessionId
            ? " active"
            : ""
        );

      const title =
        document.createElement("span");

      title.className =
        "history-title";

      title.title =
        session.title || "Chat";

      title.textContent =
        session.title || "Chat";

      const deleteButton =
        document.createElement("button");

      deleteButton.className =
        "delete-btn";

      deleteButton.dataset.id =
        session.session_id;

      deleteButton.title =
        "Delete chat";

      deleteButton.textContent =
        "🗑️";

      item.appendChild(title);
      item.appendChild(deleteButton);

      title.addEventListener(
        "click",
        () =>
          openChatSession(
            session.session_id
          )
      );

      deleteButton.addEventListener(
        "click",
        (e) => {
          e.stopPropagation();

          deleteChatSession(
            session.session_id
          );
        }
      );

      el.historyList.appendChild(item);
    }
  );
}

async function openChatSession(
  sessionId
) {
  try {
    const response =
      await fetch(
        `${API.chatHistory}/${encodeURIComponent(
          sessionId
        )}`,
        {
          credentials:
            "same-origin",
        }
      );

    if (handleUnauthorized(response)) {
      return;
    }

    if (!response.ok) {
      throw new Error(
        "Could not load that chat."
      );
    }

    const detail =
      await safeJson(response);

    state.sessionId =
      detail.session_id;

    state.history =
      Array.isArray(detail.messages)
        ? detail.messages.map(
            (m) => ({
              role: m.role,
              content: m.content,
            })
          )
        : [];

    clearWelcomeMessage();

    el.chatWindow.innerHTML = "";

    if (Array.isArray(detail.messages)) {
      detail.messages.forEach(
        (m) => {
          appendMessage({
            role: m.role,
            content: m.content,
            modeUsed:
              m.mode_used,
          });
        }
      );
    }

    renderHistoryList();
  } catch (err) {
    console.error(err);
  }
}

async function deleteChatSession(
  sessionId
) {
  try {
    const response =
      await fetch(
        `${API.chatHistory}/${encodeURIComponent(
          sessionId
        )}`,
        {
          method: "DELETE",
          credentials:
            "same-origin",
        }
      );

    if (handleUnauthorized(response)) {
      return;
    }

    if (!response.ok) {
      const data =
        await safeJson(response);

      throw new Error(
        data.error ||
          data.detail ||
          "Could not delete that chat."
      );
    }

    if (
      state.sessionId ===
      sessionId
    ) {
      startNewChat();
    }

    await loadChatHistory();
  } catch (err) {
    console.error(err);

    alert(
      `Could not delete chat: ${err.message}`
    );
  }
}

// ----------------------------------------------------------------------
// PDF Document Upload
// ----------------------------------------------------------------------

if (el.fileInput) {
  el.fileInput.addEventListener(
    "change",
    async () => {
      const file =
        el.fileInput.files?.[0];

      if (!file) return;

      // ------------------------------------------------------------
      // Client-side PDF validation
      // ------------------------------------------------------------

      const fileName =
        file.name.toLowerCase();

      const isPdf =
        file.type === "application/pdf" ||
        fileName.endsWith(".pdf");

      if (!isPdf) {
        el.uploadStatus.textContent =
          "✗ Please select a PDF file.";

        el.uploadStatus.className =
          "upload-status error";

        el.fileInput.value = "";

        return;
      }

      // ------------------------------------------------------------
      // Client-side size check
      // Backend still performs the real security validation.
      // This value matches the current config default of 20 MB.
      // ------------------------------------------------------------

      const maxPdfSize =
        20 * 1024 * 1024;

      if (file.size > maxPdfSize) {
        el.uploadStatus.textContent =
          "✗ PDF exceeds the current 20 MB upload limit.";

        el.uploadStatus.className =
          "upload-status error";

        el.fileInput.value = "";

        return;
      }

      el.uploadStatus.textContent =
        `Uploading "${file.name}"...`;

      el.uploadStatus.className =
        "upload-status";

      const formData =
        new FormData();

      formData.append(
        "file",
        file
      );

      try {
        const response =
          await fetch(
            API.upload,
            {
              method: "POST",
              body: formData,
              credentials:
                "same-origin",
            }
          );

        if (
          handleUnauthorized(response)
        ) {
          return;
        }

        const data =
          await safeJson(response);

        if (!response.ok) {
          throw new Error(
            data.error ||
              data.detail ||
              "Upload failed."
          );
        }

        const documentInfo =
          data.document || {};

        el.uploadStatus.textContent =
          `✓ ${
            documentInfo.document_name ||
            file.name
          } indexed (${
            documentInfo.num_chunks ??
            0
          } chunks)`;

        el.uploadStatus.className =
          "upload-status success";

        await loadDocuments();
      } catch (err) {
        el.uploadStatus.textContent =
          `✗ ${err.message}`;

        el.uploadStatus.className =
          "upload-status error";
      } finally {
        // Reset input so the same PDF can be selected again
        // if needed.
        el.fileInput.value = "";
      }
    }
  );
}

async function loadDocuments() {
  try {
    const response =
      await fetch(
        API.documents,
        {
          credentials:
            "same-origin",
        }
      );

    if (handleUnauthorized(response)) {
      return;
    }

    const documents =
      await safeJson(response);

    state.documents =
      Array.isArray(documents)
        ? documents
        : [];

    renderDocumentList();
  } catch (err) {
    console.error(
      "Failed to load documents",
      err
    );
  }
}

function renderDocumentList() {
  if (
    !state.documents ||
    state.documents.length === 0
  ) {
    el.documentList.innerHTML =
      `<p class="empty-hint">No documents uploaded yet.</p>`;

    return;
  }

  el.documentList.innerHTML = "";

  state.documents.forEach(
    (doc) => {
      const item =
        document.createElement("div");

      item.className =
        "document-item";

      const sizeKb =
        Math.round(
          (doc.size_bytes || 0) /
            1024
        );

      const info =
        document.createElement("div");

      info.className =
        "doc-info";

      const name =
        document.createElement("div");

      name.className =
        "doc-name";

      name.title =
        doc.document_name || "";

      name.textContent =
        doc.document_name ||
        "Unnamed document";

      const meta =
        document.createElement("div");

      meta.className =
        "doc-meta";

      meta.textContent =
        `${doc.num_chunks || 0} chunks · ${sizeKb} KB`;

      info.appendChild(name);
      info.appendChild(meta);

      const deleteButton =
        document.createElement("button");

      deleteButton.className =
        "delete-btn";

      deleteButton.dataset.id =
        doc.document_id;

      deleteButton.title =
        "Delete document";

      deleteButton.textContent =
        "🗑️";

      item.appendChild(info);
      item.appendChild(deleteButton);

      deleteButton.addEventListener(
        "click",
        () =>
          deleteDocument(
            doc.document_id
          )
      );

      el.documentList.appendChild(item);
    }
  );
}

async function deleteDocument(
  documentId
) {
  try {
    const response =
      await fetch(
        `${API.documents}/${encodeURIComponent(
          documentId
        )}`,
        {
          method: "DELETE",
          credentials:
            "same-origin",
        }
      );

    if (handleUnauthorized(response)) {
      return;
    }

    const data =
      await safeJson(response);

    if (!response.ok) {
      throw new Error(
        data.error ||
          data.detail ||
          "Failed to delete document."
      );
    }

    await loadDocuments();
  } catch (err) {
    alert(
      `Could not delete document: ${err.message}`
    );
  }
}

// ----------------------------------------------------------------------
// Health check
// ----------------------------------------------------------------------

async function checkHealth() {
  if (!el.healthIndicator) return;

  try {
    const response =
      await fetch(API.health);

    const data =
      await safeJson(response);

    el.healthIndicator.classList.remove(
      "ok",
      "degraded",
      "error"
    );

    if (
      response.ok &&
      data.status === "ok"
    ) {
      el.healthIndicator.classList.add(
        "ok"
      );

      el.healthIndicator.title =
        "All systems operational";
    } else {
      el.healthIndicator.classList.add(
        "degraded"
      );

      el.healthIndicator.title =
        (data.warnings || []).join(
          " | "
        ) ||
        "Degraded";
    }
  } catch (err) {
    el.healthIndicator.classList.remove(
      "ok",
      "degraded"
    );

    el.healthIndicator.classList.add(
      "error"
    );

    el.healthIndicator.title =
      "Backend unreachable";
  }
}

// ----------------------------------------------------------------------
// Mobile PDF picker support
// ----------------------------------------------------------------------

// Some mobile browsers behave better when the file input is triggered
// from a direct user interaction. The label in index.html already
// provides that interaction. This listener also supports direct clicks
// on the upload button if the browser requires it.

const uploadButton =
  document.querySelector(
    'label[for="file-input"]'
  );

if (
  uploadButton &&
  el.fileInput
) {
  uploadButton.addEventListener(
    "click",
    () => {
      // The label normally opens the native picker automatically.
      // Do not call click() here because that can cause duplicate
      // picker dialogs on some browsers.
    }
  );
}

// ----------------------------------------------------------------------
// Initialisation
// ----------------------------------------------------------------------

loadDocuments();
loadChatHistory();
checkHealth();

setInterval(
  checkHealth,
  30000
);