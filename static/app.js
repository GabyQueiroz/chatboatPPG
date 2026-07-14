const form = document.querySelector("#chat-form");
const input = document.querySelector("#query");
const messages = document.querySelector("#messages");
const send = document.querySelector("#send");
const suggestionsList = document.querySelector("#suggestions-list");

const conversation = [];

function addToHistory(role, content) {
  conversation.push({ role, content });
  if (conversation.length > 12) {
    conversation.splice(0, conversation.length - 12);
  }
}

function addMessage(role, text) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const body = document.createElement("p");
  body.textContent = text;
  article.appendChild(body);

  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;
  return article;
}

function updateMessage(article, text) {
  article.innerHTML = "";
  const body = document.createElement("p");
  body.textContent = text;
  article.appendChild(body);
  messages.scrollTop = messages.scrollHeight;
}

async function ask(query) {
  const response = await fetch("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, history: conversation.slice(-8) }),
  });

  if (!response.ok) {
    throw new Error("Falha ao consultar a API.");
  }

  return response.json();
}

function submitSuggestion(question) {
  input.value = question;
  form.requestSubmit();
}

async function loadSuggestions() {
  if (!suggestionsList) return;

  try {
    const response = await fetch(`/api/suggestions?t=${Date.now()}`);
    const data = await response.json();
    suggestionsList.innerHTML = "";
    (data.suggestions || []).forEach((question) => {
      const button = document.createElement("button");
      button.className = "suggestion-chip";
      button.type = "button";
      button.textContent = question;
      button.addEventListener("click", () => submitSuggestion(question));
      suggestionsList.appendChild(button);
    });
  } catch (error) {
    suggestionsList.innerHTML = "";
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = input.value.trim();
  if (!query) return;

  addMessage("user", query);
  addToHistory("user", query);
  input.value = "";
  send.disabled = true;
  send.textContent = "Consultando";
  const pending = addMessage("assistant", "Consultando os documentos...");

  try {
    const data = await ask(query);
    updateMessage(pending, data.results);
    addToHistory("assistant", data.results || "");
    loadSuggestions();
  } catch (error) {
    const message = "Não consegui consultar o chatbot agora. Verifique se o servidor e o Ollama estão ativos.";
    updateMessage(pending, message);
    addToHistory("assistant", message);
  } finally {
    send.disabled = false;
    send.textContent = "Enviar";
    input.focus();
  }
});

loadSuggestions();
