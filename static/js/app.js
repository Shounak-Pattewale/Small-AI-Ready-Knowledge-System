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
    if (requestInFlight) return;

    const question = textarea.value.trim();
    if (!question) return;
    if (question.length > MAX_QUESTION_LENGTH) return; // server remains authoritative either way

    appendUserMessage(question);
    textarea.value = "";
    textarea.style.height = "auto";

    submitQuestion(question);
  });

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
