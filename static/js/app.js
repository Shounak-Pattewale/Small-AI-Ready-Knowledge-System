/* Vanilla JS only. Each submitted question is ONE independent request to
 * POST /demo - no chat history is ever sent to the backend. All model/user
 * text is inserted via textContent, never innerHTML, so nothing returned by
 * the LLM (or typed by the user) can be interpreted as markup/script.
 */
(function () {
  "use strict";

  const MAX_QUESTION_LENGTH = 1000;

  const form = document.getElementById("chat-form");
  const textarea = document.getElementById("question");
  const sendButton = document.getElementById("send-button");
  const messages = document.getElementById("messages");
  const suggestions = document.getElementById("suggestions");
  const suggestionButtons = document.querySelectorAll(".suggestion");
  const exampleButtons = document.querySelectorAll(".example");
  const examplesPanel = document.getElementById("examples-panel");
  const examplesToggle = document.getElementById("examples-toggle");

  let requestInFlight = false;

  function scrollToBottom() {
    messages.scrollTo({ top: messages.scrollHeight, behavior: "smooth" });
  }

  function appendUserMessage(text) {
    const row = document.createElement("div");
    row.className = "msg msg--user";
    const bubble = document.createElement("div");
    bubble.className = "msg__bubble";
    bubble.textContent = text; // never innerHTML - user input is untrusted
    row.appendChild(bubble);
    messages.appendChild(row);
    scrollToBottom();
  }

  function appendSystemMessage(text) {
    const row = document.createElement("div");
    row.className = "msg msg--system";
    const bubble = document.createElement("div");
    bubble.className = "msg__bubble";
    bubble.textContent = text;
    row.appendChild(bubble);
    messages.appendChild(row);
    scrollToBottom();
  }

  function buildProvenanceCard(source, section, page) {
    const card = document.createElement("div");
    card.className = "provenance";

    const label = document.createElement("div");
    label.className = "provenance__label";
    label.textContent = "Source";
    card.appendChild(label);

    // Only render fields the backend actually returned - never a "Page: null" line.
    if (source) {
      const line = document.createElement("div");
      line.className = "provenance__line";
      line.textContent = source;
      card.appendChild(line);
    }
    if (section) {
      const line = document.createElement("div");
      line.className = "provenance__line";
      line.textContent = section;
      card.appendChild(line);
    }
    if (page !== null && page !== undefined) {
      const line = document.createElement("div");
      line.className = "provenance__line";
      line.textContent = "Page " + page;
      card.appendChild(line);
    }
    return card;
  }

  function appendAssistantAnswer(data) {
    const row = document.createElement("div");
    row.className = "msg msg--assistant";

    const group = document.createElement("div");
    group.className = "msg__group";

    const bubble = document.createElement("div");
    bubble.className = "msg__bubble";

    if (data.answered) {
      bubble.textContent = data.answer; // plain text only - no Markdown/HTML rendering, see CSS white-space: pre-wrap
      group.appendChild(bubble);
      if (data.source || data.section || data.page !== null) {
        group.appendChild(buildProvenanceCard(data.source, data.section, data.page));
      }
    } else {
      bubble.textContent = "I couldn't find enough information in the knowledge base to answer that.";
      group.appendChild(bubble);
    }

    row.appendChild(group);
    messages.appendChild(row);
    scrollToBottom();
  }

  function showTypingIndicator() {
    const row = document.createElement("div");
    row.className = "msg msg--assistant";
    row.id = "typing-indicator";
    const bubble = document.createElement("div");
    bubble.className = "msg__bubble typing";
    bubble.setAttribute("aria-label", "Assistant is typing");
    for (let i = 0; i < 3; i++) {
      bubble.appendChild(document.createElement("span"));
    }
    row.appendChild(bubble);
    messages.appendChild(row);
    scrollToBottom();
  }

  function removeTypingIndicator() {
    const el = document.getElementById("typing-indicator");
    if (el) el.remove();
  }

  function setLoading(isLoading) {
    requestInFlight = isLoading;
    sendButton.disabled = isLoading;
    textarea.disabled = isLoading;
    suggestionButtons.forEach(function (button) {
      button.disabled = isLoading;
    });
    exampleButtons.forEach(function (button) {
      button.disabled = isLoading;
    });
  }

  function hideSuggestions() {
    if (suggestions) suggestions.remove();
  }

  // Shared entry point for both typed submissions and sample-question clicks -
  // one request path, one place that appends the user bubble and calls /demo.
  // Returns true if the question was accepted and submitted.
  function askQuestion(question) {
    if (requestInFlight) return false;
    if (!question) return false;
    if (question.length > MAX_QUESTION_LENGTH) return false; // server remains authoritative either way

    hideSuggestions(); // initial-state-only, per session - never reappears after the first question
    appendUserMessage(question);
    submitQuestion(question);
    return true;
  }

  function errorMessageFor(status, body) {
    if (status === 429) {
      return (body && body.error) || "Too many requests. Please wait a moment and try again.";
    }
    if (status === 503) {
      return (body && body.error) || "The knowledge service is temporarily unavailable.";
    }
    if (status === 400 || status === 413) {
      return (body && body.error) || "That question couldn't be sent. Please try a shorter question.";
    }
    if (status === 500) {
      return "Something went wrong.";
    }
    return "Something went wrong. Please try again.";
  }

  async function submitQuestion(question) {
    showTypingIndicator();
    setLoading(true);
    try {
      const response = await fetch("/demo", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question }),
      });

      let body = null;
      try {
        body = await response.json();
      } catch (_) {
        body = null;
      }

      removeTypingIndicator();

      if (!response.ok) {
        appendSystemMessage(errorMessageFor(response.status, body));
        return;
      }

      appendAssistantAnswer(body);
    } catch (_networkError) {
      removeTypingIndicator();
      appendSystemMessage("Network error. Please check your connection and try again.");
    } finally {
      setLoading(false);
      textarea.focus();
    }
  }

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    const question = textarea.value.trim();
    if (!askQuestion(question)) return;
    textarea.value = "";
    textarea.style.height = "auto";
  });

  suggestionButtons.forEach(function (button) {
    button.addEventListener("click", function () {
      askQuestion(button.dataset.question);
    });
  });

  // Right-side example panel: same shared askQuestion() path, never hidden by
  // the conversation starting - unlike the fresh-chat suggestions above.
  exampleButtons.forEach(function (button) {
    button.addEventListener("click", function () {
      askQuestion(button.dataset.question);
    });
  });

  // Mobile-only collapsible drawer (hidden by CSS on wide viewports, where the
  // toggle button itself is not shown and the panel is always visible).
  if (examplesToggle && examplesPanel) {
    examplesToggle.addEventListener("click", function () {
      const isOpen = examplesPanel.classList.toggle("examples--open");
      examplesToggle.setAttribute("aria-expanded", String(isOpen));
    });
  }

  textarea.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  textarea.addEventListener("input", function () {
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + "px";
  });
})();
