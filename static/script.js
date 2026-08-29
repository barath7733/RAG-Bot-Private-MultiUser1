// ----------------------------------------------------------------------
// Frontend logic for the AI Assistant + RAG Document Intelligence app.
// No API keys or secrets ever live in this file — every privileged call
// goes through the FastAPI backend.
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
  history: [], // {role, content} — recent context sent with each request
  documents: [],
  sessionId: null, // current chat session id (null = not started yet)
  sessions: [],
  pendingImage: null, // File object attached but not yet sent
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


// ---------------------------- Auth guard ----------------------------

function handleUnauthorized(response) {
  if (response.status === 401) {
    window.location.href = "/login";
    return true;
  }

  return false;
}


el.logoutBtn.addEventListener("click", async () => {
  try {
    await fetch(API.logout, {
      method: "POST",
      credentials: "same-origin"
    });
  } finally {
    window.location.href = "/login";
  }
});


// ---------------------------- Mode selector ----------------------------

el.modeToggle.addEventListener("click", (e) => {
  const btn = e.target.closest(".mode-btn");

  if (!btn) return;

  document
    .querySelectorAll(".mode-btn")
    .forEach((b) => b.classList.remove("active"));

  btn.classList.add("active");

  state.mode = btn.dataset.mode;
});


// ---------------------------- Chat ----------------------------

function autoResizeTextarea() {
  el.chatInput.style.height = "auto";

  el.chatInput.style.height =
    Math.min(el.chatInput.scrollHeight, 160) + "px";
}


el.chatInput.addEventListener(
  "input",
  autoResizeTextarea
);


el.chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    el.chatForm.requestSubmit();
  }
});


function clearWelcomeMessage() {
  const welcome =
    el.chatWindow.querySelector(".welcome-message");

  if (welcome) welcome.remove();
}


const MODE_LABELS = {
  rag: "📄 Document Mode",
  web: "🌐 Web Search",
  general: "💬 General AI",
  vision: "🖼️ Image Analysis",
};


// ---------------------------- Markdown rendering ----------------------------

function renderMarkdown(text) {
  if (!text) return "";

  let escaped = escapeHtml(text);

  const lines = escaped.split("\n");
  const htmlParts = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Markdown table
    if (
      line.trim().startsWith("|") &&
      lines[i + 1] &&
      /^\s*\|?[\s:-]+\|[\s:|-]*$/.test(lines[i + 1])
    ) {
      const headerCells = line
        .trim()
        .replace(/^\||\|$/g, "")
        .split("|")
        .map((c) => c.trim());

      let tableHtml =
        '<table class="md-table"><thead><tr>';

      headerCells.forEach(
        (c) =>
          (tableHtml +=
            `<th>${inlineMarkdown(c)}</th>`)
      );

      tableHtml +=
        "</tr></thead><tbody>";

      i += 2;

      while (
        i < lines.length &&
        lines[i].trim().startsWith("|")
      ) {
        const rowCells = lines[i]
          .trim()
          .replace(/^\||\|$/g, "")
          .split("|")
          .map((c) => c.trim());

        tableHtml += "<tr>";

        rowCells.forEach(
          (c) =>
            (tableHtml +=
              `<td>${inlineMarkdown(c)}</td>`)
        );

        tableHtml += "</tr>";

        i++;
      }

      tableHtml += "</tbody></table>";

      htmlParts.push(tableHtml);

      continue;
    }


    // Headings
    const headingMatch =
      line.match(/^(#{1,4})\s+(.*)$/);

    if (headingMatch) {
      const level = Math.min(
        headingMatch[1].length + 2,
        6
      );

      htmlParts.push(
        `<h${level} class="md-heading">${inlineMarkdown(
          headingMatch[2]
        )}</h${level}>`
      );

      i++;

      continue;
    }


    // Bullet list
    if (/^\s*[-*]\s+/.test(line)) {
      let listHtml =
        '<ul class="md-list">';

      while (
        i < lines.length &&
        /^\s*[-*]\s+/.test(lines[i])
      ) {
        listHtml +=
          `<li>${inlineMarkdown(
            lines[i].replace(
              /^\s*[-*]\s+/,
              ""
            )
          )}</li>`;

        i++;
      }

      listHtml += "</ul>";

      htmlParts.push(listHtml);

      continue;
    }


    // Numbered list
    if (/^\s*\d+[.)]\s+/.test(line)) {
      let listHtml =
        '<ol class="md-list">';

      while (
        i < lines.length &&
        /^\s*\d+[.)]\s+/.test(lines[i])
      ) {
        listHtml +=
          `<li>${inlineMarkdown(
            lines[i].replace(
              /^\s*\d+[.)]\s+/,
              ""
            )
          )}</li>`;

        i++;
      }

      listHtml += "</ol>";

      htmlParts.push(listHtml);

      continue;
    }


    // Blank line
    if (line.trim() === "") {
      i++;
      continue;
    }


    // Regular paragraph
    htmlParts.push(
      `<p class="md-p">${inlineMarkdown(line)}</p>`
    );

    i++;
  }

  return htmlParts.join("");
}


function inlineMarkdown(text) {
  return text
    .replace(
      /`([^`]+)`/g,
      "<code>$1</code>"
    )
    .replace(
      /\*\*([^*]+)\*\*/g,
      "<strong>$1</strong>"
    )
    .replace(
      /(?<!\*)\*([^*\n]+)\*(?!\*)/g,
      "<em>$1</em>"
    )
    .replace(
      /~~([^~]+)~~/g,
      "<del>$1</del>"
    )
    .replace(
      /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener">$1</a>'
    );
}


function appendMessage({
  role,
  content,
  modeUsed,
  sources,
  webSources,
  isError
}) {
  clearWelcomeMessage();

  const wrapper =
    document.createElement("div");

  wrapper.className =
    `message ${role}`;


  if (
    role === "assistant" &&
    modeUsed
  ) {
    const tag =
      document.createElement("div");

    tag.className = "mode-tag";

    tag.textContent =
      MODE_LABELS[modeUsed] ||
      "💬 General AI";

    wrapper.appendChild(tag);
  }


  const bubble =
    document.createElement("div");

  bubble.className =
    "bubble" +
    (isError ? " error-bubble" : "");


  if (
    role === "assistant" &&
    !isError
  ) {
    bubble.innerHTML =
      renderMarkdown(content);
  } else {
    bubble.textContent = content;
  }


  wrapper.appendChild(bubble);


  if (sources && sources.length > 0) {
    const box =
      document.createElement("div");

    box.className = "sources-box";

    box.innerHTML =
      "<strong>Sources:</strong>";


    sources.forEach((s) => {
      const item =
        document.createElement("div");

      item.className =
        "source-item";

      const pageInfo =
        s.page
          ? `, page ${s.page}`
          : "";

      item.innerHTML =
        `<div class="source-title">${escapeHtml(
          s.document_name
        )}${pageInfo} · score ${s.score.toFixed(
          2
        )}</div><div>${escapeHtml(
          s.snippet
        )}</div>`;

      box.appendChild(item);
    });


    wrapper.appendChild(box);
  }


  if (
    webSources &&
    webSources.length > 0
  ) {
    const box =
      document.createElement("div");

    box.className =
      "sources-box";

    box.innerHTML =
      "<strong>Sources:</strong>";


    webSources.forEach((s) => {
      const item =
        document.createElement("div");

      item.className =
        "source-item";


      const link =
        document.createElement("a");

      link.href = s.url;

      link.target = "_blank";

      link.rel =
        "noopener noreferrer";

      link.className =
        "source-title";

      link.textContent =
        s.title;

      link.style.color =
        "inherit";

      item.appendChild(link);


      const snippet =
        document.createElement("div");

      snippet.textContent =
        s.snippet;

      item.appendChild(snippet);

      box.appendChild(item);
    });


    wrapper.appendChild(box);
  }


  el.chatWindow.appendChild(wrapper);

  el.chatWindow.scrollTop =
    el.chatWindow.scrollHeight;
}


function escapeHtml(str) {
  const div =
    document.createElement("div");

  div.textContent = str;

  return div.innerHTML;
}


function setLoading(isLoading) {
  el.loadingIndicator.classList.toggle(
    "hidden",
    !isLoading
  );

  el.sendBtn.disabled =
    isLoading;

  if (isLoading) {
    el.chatWindow.scrollTop =
      el.chatWindow.scrollHeight;
  }
}


// ---------------------------- Image attach (vision) ----------------------------

el.attachImageBtn.addEventListener(
  "click",
  () =>
    el.visionFileInput.click()
);


el.visionFileInput.addEventListener(
  "change",
  () => {
    const file =
      el.visionFileInput.files[0];

    if (!file) return;


    const allowed = [
      "image/png",
      "image/jpeg",
      "image/webp",
      "image/gif"
    ];


    if (!allowed.includes(file.type)) {
      alert(
        "Only PNG, JPEG, WEBP, or GIF images are supported."
      );

      el.visionFileInput.value = "";

      return;
    }


    if (
      file.size >
      10 * 1024 * 1024
    ) {
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


function clearPendingImage() {
  state.pendingImage = null;

  el.visionFileInput.value = "";

  el.imageAttachPreview.classList.add(
    "hidden"
  );

  el.imageAttachThumb.src = "";

  el.chatInput.placeholder =
    "Ask a question, or attach an image to ask about it...";
}


el.imageAttachRemove.addEventListener(
  "click",
  clearPendingImage
);


async function submitImageAnalysis() {
  const file =
    state.pendingImage;

  const question =
    el.chatInput.value.trim();


  clearWelcomeMessage();


  appendMessage({
    role: "user",
    content:
      question ||
      "🖼️ (image attached)"
  });


  state.history.push({
    role: "user",
    content:
      question ||
      "[Attached an image]"
  });


  el.chatInput.value = "";

  autoResizeTextarea();


  const thumbUrl =
    URL.createObjectURL(file);

  clearPendingImage();

  setLoading(true);


  const formData =
    new FormData();

  formData.append(
    "file",
    file
  );

  formData.append(
    "question",
    question
  );


  if (state.sessionId) {
    formData.append(
      "session_id",
      state.sessionId
    );
  }


  try {
    const response =
      await fetch(
        API.visionAnalyze,
        {
          method: "POST",
          body: formData,
          credentials:
            "same-origin"
        }
      );


    if (
      handleUnauthorized(response)
    ) {
      return;
    }


    const data =
      await response.json();


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
      modeUsed: "vision"
    });


    state.history.push({
      role: "assistant",
      content: data.answer
    });


    loadChatHistory();


  } catch (err) {

    appendMessage({
      role: "assistant",
      content:
        `Something went wrong: ${err.message}`,
      isError: true
    });

  } finally {

    setLoading(false);

    URL.revokeObjectURL(
      thumbUrl
    );
  }
}


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
      content: question
    });


    state.history.push({
      role: "user",
      content: question
    });


    el.chatInput.value = "";

    autoResizeTextarea();

    setLoading(true);


    try {

      const response =
        await fetch(
          API.chat,
          {
            method: "POST",
            headers: {
              "Content-Type":
                "application/json"
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
                state.sessionId
            })
          }
        );


      if (
        handleUnauthorized(response)
      ) {
        return;
      }


      const data =
        await response.json();


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
          data.web_sources
      });


      state.history.push({
        role: "assistant",
        content: data.answer
      });


      loadChatHistory();


    } catch (err) {

      appendMessage({
        role: "assistant",
        content:
          `Something went wrong: ${err.message}`,
        isError: true
      });

    } finally {

      setLoading(false);
    }
  }
);


// ---------------------------- New Chat ----------------------------

function startNewChat() {

  state.history = [];

  state.sessionId = null;


  el.chatWindow.innerHTML = `
    <div class="welcome-message">
      <h2>How can I help you today?</h2>
      <p>Ask me anything, or upload a PDF and ask questions about it.</p>
    </div>
  `;


  renderHistoryList();
}


el.clearChatBtn.addEventListener(
  "click",
  startNewChat
);


el.newChatBtn.addEventListener(
  "click",
  startNewChat
);


// ---------------------------- Chat history ----------------------------

async function loadChatHistory() {

  try {

    const response =
      await fetch(
        API.chatHistory,
        {
          credentials:
            "same-origin"
        }
      );


    if (
      handleUnauthorized(response)
    ) {
      return;
    }


    const sessions =
      await response.json();


    state.sessions =
      sessions;


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
        document.createElement(
          "div"
        );


      item.className =
        "history-item" +
        (
          session.session_id ===
          state.sessionId
            ? " active"
            : ""
        );


      item.innerHTML = `
        <span
          class="history-title"
          title="${escapeHtml(
            session.title
          )}"
        >
          ${escapeHtml(
            session.title
          )}
        </span>

        <button
          class="delete-btn"
          data-id="${session.session_id}"
          title="Delete chat"
        >
          🗑️
        </button>
      `;


      item
        .querySelector(
          ".history-title"
        )
        .addEventListener(
          "click",
          () =>
            openChatSession(
              session.session_id
            )
        );


      item
        .querySelector(
          ".delete-btn"
        )
        .addEventListener(
          "click",
          (e) => {
            e.stopPropagation();

            deleteChatSession(
              session.session_id
            );
          }
        );


      el.historyList.appendChild(
        item
      );
    }
  );
}


async function openChatSession(
  sessionId
) {

  try {

    const response =
      await fetch(
        `${API.chatHistory}/${sessionId}`,
        {
          credentials:
            "same-origin"
        }
      );


    if (
      handleUnauthorized(response)
    ) {
      return;
    }


    if (!response.ok) {
      throw new Error(
        "Could not load that chat."
      );
    }


    const detail =
      await response.json();


    state.sessionId =
      detail.session_id;


    state.history =
      detail.messages.map(
        (m) => ({
          role: m.role,
          content: m.content
        })
      );


    clearWelcomeMessage();

    el.chatWindow.innerHTML = "";


    detail.messages.forEach(
      (m) => {

        appendMessage({
          role: m.role,
          content: m.content,
          modeUsed:
            m.mode_used
        });

      }
    );


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
        `${API.chatHistory}/${sessionId}`,
        {
          method: "DELETE",
          credentials:
            "same-origin"
        }
      );


    if (
      handleUnauthorized(response)
    ) {
      return;
    }


    if (!response.ok) {
      throw new Error(
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
  }
}


// ---------------------------- Document upload ----------------------------

el.fileInput.addEventListener(
  "change",
  async () => {

    const file =
      el.fileInput.files[0];

    if (!file) return;


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
              "same-origin"
          }
        );


      if (
        handleUnauthorized(response)
      ) {
        return;
      }


      const data =
        await response.json();


      if (!response.ok) {
        throw new Error(
          data.error ||
          data.detail ||
          "Upload failed."
        );
      }


      el.uploadStatus.textContent =
        `✓ ${data.document.document_name} indexed (${data.document.num_chunks} chunks)`;


      el.uploadStatus.className =
        "upload-status success";


      await loadDocuments();


    } catch (err) {

      el.uploadStatus.textContent =
        `✗ ${err.message}`;

      el.uploadStatus.className =
        "upload-status error";


    } finally {

      el.fileInput.value = "";
    }
  }
);


async function loadDocuments() {

  try {

    const response =
      await fetch(
        API.documents,
        {
          credentials:
            "same-origin"
        }
      );


    if (
      handleUnauthorized(response)
    ) {
      return;
    }


    const documents =
      await response.json();


    state.documents =
      documents;


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
        document.createElement(
          "div"
        );


      item.className =
        "document-item";


      const sizeKb =
        Math.round(
          doc.size_bytes / 1024
        );


      item.innerHTML = `
        <div class="doc-info">

          <div
            class="doc-name"
            title="${escapeHtml(
              doc.document_name
            )}"
          >
            ${escapeHtml(
              doc.document_name
            )}
          </div>

          <div class="doc-meta">
            ${doc.num_chunks} chunks · ${sizeKb} KB
          </div>

        </div>

        <button
          class="delete-btn"
          data-id="${doc.document_id}"
          title="Delete document"
        >
          🗑️
        </button>
      `;


      el.documentList.appendChild(
        item
      );
    }
  );


  el.documentList
    .querySelectorAll(
      ".delete-btn"
    )
    .forEach(
      (btn) => {

        btn.addEventListener(
          "click",
          () =>
            deleteDocument(
              btn.dataset.id
            )
        );

      }
    );
}


async function deleteDocument(
  documentId
) {

  try {

    const response =
      await fetch(
        `${API.documents}/${documentId}`,
        {
          method: "DELETE",
          credentials:
            "same-origin"
        }
      );


    if (
      handleUnauthorized(response)
    ) {
      return;
    }


    const data =
      await response.json();


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


// ---------------------------- Health check ----------------------------

async function checkHealth() {

  try {

    const response =
      await fetch(API.health);


    const data =
      await response.json();


    el.healthIndicator.classList.remove(
      "ok",
      "degraded",
      "error"
    );


    if (
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
        (data.warnings || [])
          .join(" | ") ||
        "Degraded";
    }


  } catch (err) {

    el.healthIndicator.classList.add(
      "error"
    );

    el.healthIndicator.title =
      "Backend unreachable";
  }
}


// ---------------------------- Init ----------------------------

loadDocuments();

loadChatHistory();

checkHealth();

setInterval(
  checkHealth,
  30000
);