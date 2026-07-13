const form = document.querySelector("#chat-form");
const input = document.querySelector("#query");
const messages = document.querySelector("#messages");
const send = document.querySelector("#send");

function addMessage(role, text, sources = []) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const body = document.createElement("p");
  body.textContent = text;
  article.appendChild(body);

  if (sources.length) {
    const sourceBox = document.createElement("div");
    sourceBox.className = "sources";
    sourceBox.innerHTML = "<strong>Fontes consultadas</strong>";
    const list = document.createElement("ul");
    sources.slice(0, 4).forEach((item) => {
      const li = document.createElement("li");
      li.textContent = item.source;
      list.appendChild(li);
    });
    sourceBox.appendChild(list);
    article.appendChild(sourceBox);
  }

  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;
}

async function ask(query) {
  const response = await fetch("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });

  if (!response.ok) {
    throw new Error("Falha ao consultar a API.");
  }

  return response.json();
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = input.value.trim();
  if (!query) return;

  addMessage("user", query);
  input.value = "";
  send.disabled = true;
  send.textContent = "Consultando";

  try {
    const data = await ask(query);
    addMessage("assistant", data.results, data.sources || []);
  } catch (error) {
    addMessage("assistant", "Não consegui consultar o chatbot agora. Verifique se o servidor e o Ollama estão ativos.");
  } finally {
    send.disabled = false;
    send.textContent = "Enviar";
    input.focus();
  }
});
