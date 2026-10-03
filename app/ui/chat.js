"use strict";

const form = document.querySelector("#question-form");
const input = document.querySelector("#question");
const send = document.querySelector("#send");
const reset = document.querySelector("#new-chat");
const mobileReset = document.querySelector("#new-chat-mobile");
const welcome = document.querySelector("#welcome");
const conversation = document.querySelector("#conversation");
const scrollArea = document.querySelector("#chat-scroll");
const status = document.querySelector("#status");
const dialog = document.querySelector("#source-dialog");
let busy = false;
// Keep complete exchanges only. These character bounds also apply on the server.
const MAX_HISTORY_MESSAGES = 6;
const MAX_HISTORY_MESSAGE_CHARS = 6000;
const MAX_HISTORY_CHARS = 12000;
let chatHistory = [];
let historyRevision = 0;

function rememberExchange(question, answer, revision) {
  // Retrying an older failed question must not replace a newer conversation branch.
  if (revision !== historyRevision) return;
  chatHistory.push({role: "user", content: question.slice(0, MAX_HISTORY_MESSAGE_CHARS)},
    {role: "assistant", content: answer.slice(0, MAX_HISTORY_MESSAGE_CHARS)});
  while (chatHistory.length > MAX_HISTORY_MESSAGES ||
    chatHistory.reduce((total, item) => total + item.content.length, 0) > MAX_HISTORY_CHARS) {
    chatHistory.splice(0, 2);
  }
  historyRevision += 1;
}

let ticketBusy = false;
let pendingProposal = null;
let ticketContext = null;
const ticketDialog = document.querySelector("#ticket-dialog");
const ticketForm = document.querySelector("#ticket-form");
const ticketTitle = document.querySelector("#ticket-title");
const ticketDescription = document.querySelector("#ticket-description");
const ticketPriority = document.querySelector("#ticket-priority");
const ticketSubmit = document.querySelector("#submit-ticket");
const ticketError = document.querySelector("#ticket-error");
const ticketRetryNote = document.querySelector("#ticket-retry-note");
const ticketClose = document.querySelector("#close-ticket");
const ticketCancel = document.querySelector("#cancel-ticket");

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function updateComposer() {
  const locked = busy || ticketBusy;
  send.disabled = locked || !input.value.trim();
  const uncertainTicket = Boolean(pendingProposal?.attempt && !pendingProposal?.ticket);
  reset.disabled = locked || uncertainTicket;
  mobileReset.disabled = locked || uncertainTicket;
  input.readOnly = locked;
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 140)}px`;
  const count = document.querySelector("#character-count");
  count.hidden = input.value.length < 1800;
  count.textContent = `${input.value.length}/2000`;
}

function scrollToLatest() {
  scrollArea.scrollTop = scrollArea.scrollHeight;
}

function showSource(source) {
  document.querySelector("#source-title").textContent = source.source_file.replace(/\.[^.]+$/, "").replace(/[_-]/g, " ");
  document.querySelector("#source-meta").textContent = `${source.source_file} · Page ${source.page_number} · Source ${source.reference}`;
  document.querySelector("#source-text").textContent = source.text;
  dialog.showModal();
}

// Render a small Markdown subset using DOM text nodes, never document/LLM HTML.
function appendInline(parent, text, sources) {
  const pattern = /(\[\d+\]|\*\*[^*\n]+\*\*|\*[^*\n]+\*|`[^`\n]+`)/g;
  let previous = 0;
  for (const match of text.matchAll(pattern)) {
    parent.append(document.createTextNode(text.slice(previous, match.index)));
    const token = match[0];
    if (sources.has(token)) {
      const citation = element("button", "citation", token);
      citation.type = "button";
      citation.setAttribute("aria-label", `Open source ${token}`);
      citation.addEventListener("click", () => showSource(sources.get(token)));
      parent.append(citation);
    } else if (token.startsWith("**")) {
      appendInline(parent.appendChild(element("strong")), token.slice(2, -2), sources);
    } else if (token.startsWith("*")) {
      appendInline(parent.appendChild(element("em")), token.slice(1, -1), sources);
    } else if (token.startsWith("`")) {
      parent.append(element("code", "", token.slice(1, -1)));
    } else {
      parent.append(document.createTextNode(token));
    }
    previous = match.index + token.length;
  }
  parent.append(document.createTextNode(text.slice(previous)));
}

function renderAnswer(parent, answer, sources) {
  let paragraph = null;
  let list = null;
  for (const line of answer.split("\n")) {
    if (!line.trim()) { paragraph = null; list = null; continue; }
    const heading = line.match(/^#{1,6}\s+(.+)$/);
    const bullet = line.match(/^\s*(?:[-*]\s+|\d+[.)]\s+)(.+)$/);
    if (heading) {
      appendInline(parent.appendChild(element("h3")), heading[1], sources);
      paragraph = null; list = null;
    } else if (bullet) {
      const tag = /^\s*\d/.test(line) ? "ol" : "ul";
      if (!list || list.tagName.toLowerCase() !== tag) list = parent.appendChild(element(tag));
      appendInline(list.appendChild(element("li")), bullet[1], sources);
      paragraph = null;
    } else {
      if (!paragraph) paragraph = parent.appendChild(element("p"));
      else paragraph.append(element("br"));
      appendInline(paragraph, line, sources);
      list = null;
    }
  }
}

function collectSources(response) {
  const sources = new Map();
  for (const step of response.steps || []) {
    if (step.name !== "search_documents") continue;
    for (const source of step.result?.sources || []) {
      if (typeof source.reference === "string" && typeof source.text === "string" &&
          typeof source.source_file === "string" && Number.isInteger(source.page_number)) {
        sources.set(source.reference, source);
      }
    }
  }
  return sources;
}

function durationLabel(value) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) return "Unavailable";
  if (value > 0 && value < 1) return "<1 ms";
  return value < 1000 ? `${Math.round(value)} ms` : `${(value / 1000).toFixed(2)} s`;
}

function appendRequestActivity(card, response, requestId = "") {
  const trace = response?.trace;
  const steps = Array.isArray(response?.steps) ? response.steps : [];
  if (!trace && !requestId && !steps.length) return;
  const details = element("details", "request-activity");
  const summary = element("summary");
  const count = trace?.number_of_tool_calls ?? steps.length;
  summary.append(element("span", "activity-title", "Request activity"),
    element("span", "activity-preview", `${count} tool${count === 1 ? "" : "s"}${trace ? ` · ${durationLabel(trace.latency_ms)}` : ""}`));
  details.append(summary);
  const body = element("div", "activity-body");
  const header = element("div", "activity-heading");
  const failed = trace?.status === "failure" || response?.completed === false;
  header.append(element("span", `activity-status${failed ? " activity-failed" : ""}`, failed ? "Request stopped" : "Completed"));
  if (trace?.endpoint) header.append(element("code", "activity-endpoint", trace.endpoint));
  body.append(header);
  const metrics = element("dl", "activity-metrics");
  for (const [label, value] of [
    ["Processing time", durationLabel(trace?.latency_ms)],
    ["Model time", durationLabel(trace?.llm_latency_ms)],
    ["Retrieval time", trace?.embedding_model ? durationLabel(trace.retrieval_latency_ms) : "Not used"],
    ["Total tokens", Number.isInteger(trace?.total_tokens) ? trace.total_tokens.toLocaleString() : "Unavailable"],
  ]) {
    const metric = element("div");
    metric.append(element("dt", "", label), element("dd", "", value));
    metrics.append(metric);
  }
  body.append(metrics);
  const meta = element("dl", "activity-meta");
  for (const [label, value] of [
    ["Request ID", trace?.request_id || requestId || "Unavailable"],
    ["LLM model", trace?.llm_model || "Unavailable"],
    ["Embedding model", trace?.embedding_model || "Not used"],
  ]) {
    meta.append(element("dt", "", label), element("dd", "", value));
  }
  body.append(meta);
  const events = Array.isArray(trace?.events) ? trace.events : steps.map(step => ({
    kind: "tool", name: step.name, status: step.result?.error ? "failure" : "success",
  }));
  const timeline = element("ol", "activity-timeline");
  const labels = {search_documents: "Search documents", get_ticket: "Read ticket",
    prepare_ticket: "Prepare ticket proposal", create_ticket: "Create ticket", escalate_ticket: "Escalate ticket"};
  let toolIndex = 0;
  for (const event of events) {
    const step = event.kind === "tool" ? steps[toolIndex++] : null;
    const item = element("li", `activity-event${event.status === "failure" ? " activity-event-failed" : ""}`);
    const title = element("div", "activity-event-heading");
    title.append(element("strong", "", event.kind === "llm" ? "Model call" : labels[event.name] || "Tool call"),
      element("span", "", durationLabel(event.latency_ms)));
    item.append(title);
    if (event.kind === "tool") item.append(element("code", "activity-tool-name", event.name));
    const outcome = event.kind === "llm" ? (event.status === "failure" ? "Provider request failed" : "Response received") :
      event.status === "failure" ? "Failed" : "Succeeded";
    item.append(element("p", "activity-outcome", outcome));
    if (step?.name === event.name) {
      if (typeof step.result?.error === "string") item.append(element("p", "activity-note", step.result.error));
      else if (step.name === "search_documents") {
        if (typeof step.input?.question === "string") {
          const query = element("p", "activity-note", `Search query: ${step.input.question}`);
          query.dir = "auto";
          item.append(query);
        }
        const sources = Array.isArray(step.result?.sources) ? step.result.sources : [];
        item.append(element("p", "activity-note", `${sources.length} source excerpt${sources.length === 1 ? "" : "s"} returned`));
        const links = element("div", "activity-source-links");
        for (const source of sources) {
          if (!collectSources({steps: [step]}).has(source.reference)) continue;
          const link = element("button", "text-button", `${source.reference} ${source.source_file} · Page ${source.page_number}`);
          link.type = "button";
          link.addEventListener("click", () => showSource(source));
          links.append(link);
        }
        item.append(links);
      } else if (step.name === "prepare_ticket") {
        item.append(element("p", "activity-note", "Proposal prepared for review. No ticket created."));
      } else if (Number.isInteger(step.result?.id)) {
        item.append(element("p", "activity-note", `Ticket #${step.result.id} · ${step.result.status || ""}`));
      }
    }
    timeline.append(item);
  }
  if (events.length) body.append(timeline);
  else body.append(element("p", "activity-note", "No execution events are available for this request."));
  body.append(element("p", "activity-footnote", "Recorded actions, shown after the request finishes. Retrieval includes embedding and database search; it is part of the search tool time. Processing time excludes browser/network time. Tokens include usage reported by the provider."));
  details.append(body);
  card.append(details);
}

function assistantMessage() {
  const message = element("article", "message assistant-message");
  const label = element("div", "message-label");
  const avatar = element("span", "avatar", "✳");
  avatar.setAttribute("aria-hidden", "true");
  label.append(avatar, document.createTextNode("Support Agent"));
  const card = element("div", "answer-card");
  message.append(label, card);
  conversation.append(message);
  return card;
}

function showThinking(card) {
  card.replaceChildren();
  card.classList.remove("error-card");
  const thinking = element("div", "thinking");
  const dots = element("span", "thinking-dots");
  dots.setAttribute("aria-hidden", "true");
  dots.append(element("span"), element("span"), element("span"));
  thinking.append(dots, document.createTextNode("Looking into your question…"));
  card.append(thinking);
}

function showResponse(card, response, question) {
  card.replaceChildren();
  const sources = collectSources(response);
  const proposal = response.completed && validTicketDraft(response.ticket_proposal);
  if (!response.completed) {
    card.append(element("p", "incomplete-note", "The assistant could not finish this request. You can review any retrieved sources below and try again."));
  }
  const answer = element("div", "answer-text");
  answer.dir = "auto";
  renderAnswer(answer, proposal ? "I've prepared a ticket proposal from our conversation. Review it before confirming." : response.answer, sources);
  card.append(answer);
  appendRequestActivity(card, response);
  if (sources.size) {
    const section = element("div", "sources");
    section.append(element("p", "sources-label", "EXPLORE THE SOURCES"));
    const list = element("div", "source-list");
    for (const source of sources.values()) {
      const button = element("button", "source-card");
      button.type = "button";
      const name = element("span", "source-name", source.source_file.replace(/\.[^.]+$/, "").replace(/[_-]/g, " "));
      name.append(element("span", "source-page", `Page ${source.page_number}`));
      button.append(element("span", "source-number", source.reference), name, element("span", "source-open", "↗"));
      button.addEventListener("click", () => showSource(source));
      list.append(button);
    }
    section.append(list);
    card.append(section);
  } else if (!proposal) {
    card.append(element("p", "no-sources", "No document sources were used for this answer."));
  }
  if (proposal) {
    showTicketProposal(card, response.ticket_proposal, question, response);
    return;
  }
  const footer = element("div", "answer-footer");
  footer.append(element("span", "", sources.size ? "Check the cited sources for the policy details." : "Ask a policy question to explore your documents."));
  const copy = element("button", "text-button", "Copy answer");
  copy.type = "button";
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(response.answer);
      copy.textContent = "Copied";
      setTimeout(() => { copy.textContent = "Copy answer"; }, 2000);
    } catch { copy.textContent = "Could not copy"; }
  });
  footer.append(copy);
  card.append(footer);
  const support = element("div", "support-action");
  support.append(element("span", "", "Need a hand from IT?"));
  const help = element("button", "support-button", "Still need help? ↗");
  help.type = "button";
  const context = {question, response, card, help, attempt: null, draft: null, ticket: null};
  help.addEventListener("click", () => openTicket(context));
  support.append(help);
  card.append(support);
}

function validTicketDraft(payload) {
  return payload && typeof payload.title === "string" && payload.title.trim().length > 0 && payload.title.length <= 200 &&
    typeof payload.description === "string" && payload.description.trim().length > 0 && payload.description.length <= 5000 &&
    ["low", "medium", "high"].includes(payload.priority);
}

function renderProposal(context) {
  context.details.replaceChildren(element("h3", "", context.draft.title));
  const priority = element("span", "ticket-badge", `Priority: ${context.draft.priority}`);
  const description = element("p", "ticket-saved-description", context.draft.description);
  description.dir = "auto";
  context.details.append(priority, description);
  const active = pendingProposal === context && !context.ticket;
  context.confirm.disabled = busy || ticketBusy || !active || Boolean(context.conflict);
  context.confirm.textContent = ticketBusy && active ? "Creating ticket…" : context.attempt ? "Retry creation" : "Confirm ticket";
  context.confirm.hidden = Boolean(context.ticket);
  context.cancel.disabled = busy || ticketBusy || !active || Boolean(context.attempt);
  context.cancel.hidden = Boolean(context.ticket);
  context.help.disabled = busy || ticketBusy || (!context.ticket && (!active || Boolean(context.attempt)));
}

function dismissProposal(context, text = "Proposal cancelled. No ticket was created.") {
  if (!context || context.attempt || ticketBusy) return;
  if (pendingProposal === context) pendingProposal = null;
  context.note.textContent = text;
  renderProposal(context);
  updateComposer();
}

function showTicketProposal(card, draft, question, response) {
  dismissProposal(pendingProposal, "A newer proposal replaced this one.");
  const section = element("section", "ticket-proposal");
  section.setAttribute("aria-label", "Ticket proposal awaiting confirmation");
  const label = element("p", "sources-label", "REVIEW BEFORE CREATING");
  section.append(label);
  const details = element("div", "ticket-receipt-details");
  const note = element("p", "proposal-note", "Review the details. Click Confirm ticket or reply yes / כן to create it.");
  note.setAttribute("role", "status");
  const actions = element("div", "ticket-actions");
  const confirm = element("button", "primary-button", "Confirm ticket");
  const help = element("button", "secondary-button", "Edit details");
  const cancel = element("button", "secondary-button", "Cancel proposal");
  for (const button of [confirm, help, cancel]) button.type = "button";
  const context = {question, response, card, help, confirm, cancel, note, details, section, label, draft: {...draft},
    viaAgent: true, attempt: null, ticket: null, historyRevision};
  pendingProposal = context;
  confirm.addEventListener("click", () => confirmProposal(context));
  help.addEventListener("click", () => openTicket(context));
  cancel.addEventListener("click", () => dismissProposal(context));
  actions.append(cancel, help, confirm);
  section.append(details, note, actions);
  card.append(section);
  renderProposal(context);
}

async function sendTicketRequest(context) {
  context.agentResponse = null;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), context.viaAgent ? 120000 : 20000);
  try {
    const result = await fetch(context.viaAgent ? "/agent" : "/tickets", {
      method: "POST", headers: {"Content-Type": "application/json", "Idempotency-Key": context.attempt.key},
      body: JSON.stringify(context.viaAgent ? {
        message: "Create the reviewed ticket with exactly the approved fields.",
        allowed_actions: ["create_ticket"], idempotency_key: context.attempt.key,
        approved_ticket: context.attempt.payload,
      } : context.attempt.payload), signal: controller.signal,
    });
    const data = await result.json().catch(() => null);
    if (context.viaAgent) context.agentResponse = {...data, requestId: result.headers.get("X-Request-ID")};
    if (!result.ok) {
      if (result.status === 422) {
        context.attempt = null;
        throw new Error("Some ticket details were rejected. Review the fields and confirm again.");
      }
      if (result.status === 409) {
        context.conflict = true;
        throw new Error("This request conflicts with an existing ticket. Contact IT before opening another request.");
      }
      throw new Error("We could not confirm ticket creation. Retry this same request to check or complete it.");
    }
    // A real create_ticket tool result is authoritative, even if the final LLM answer failed.
    const ticket = context.viaAgent ? data?.steps?.find(step => step.name === "create_ticket" && !step.result?.error)?.result : data;
    if (!ticket || !Number.isInteger(ticket.id) || ticket.id < 1 || !validTicketDraft(ticket) ||
        !["open", "in_progress", "closed"].includes(ticket.status) ||
        ["title", "description", "priority"].some(field => ticket[field] !== context.attempt.payload[field])) {
      throw new Error("We could not verify a created ticket. Retry this same request; no new request will be opened.");
    }
    return ticket;
  } finally { clearTimeout(timeout); }
}

function creationError(error) {
  return error.name === "AbortError" ? "Confirmation timed out. Keep this chat open and retry the same request." :
    error instanceof TypeError ? "Connection lost. Keep this chat open and retry the same request." : error.message;
}

function ticketCreated(context, ticket, approval = "Yes, create the reviewed ticket.") {
  context.ticket = ticket;
  context.help.textContent = `View ticket #${ticket.id} ↗`;
  const receipt = `Ticket #${ticket.id} · ${ticket.status.replace(/_/g, " ")} · ${ticket.priority} priority`;
  if (context.viaAgent) {
    pendingProposal = null;
    context.note.textContent = receipt;
    context.section.setAttribute("aria-label", "Created ticket");
    context.label.textContent = "CREATED TICKET";
    context.creationActivity?.remove();
    const receiptCard = assistantMessage();
    receiptCard.append(element("p", "ticket-confirmation", receipt));
    appendRequestActivity(receiptCard, context.agentResponse, context.agentResponse?.requestId);
    rememberExchange(approval, receipt, context.historyRevision);
    renderProposal(context);
  } else {
    context.card.append(element("p", "ticket-confirmation", receipt));
  }
  status.textContent = receipt;
}

async function confirmProposal(context, approval) {
  if (busy || ticketBusy || pendingProposal !== context || context.ticket || context.conflict) return;
  if (!validTicketDraft(context.draft)) { openTicket(context); return; }
  if (!context.attempt) {
    const message = element("article", "message");
    message.append(element("div", "user-message", approval || "Confirm ticket"));
    conversation.append(message);
  }
  context.attempt ||= {key: crypto.randomUUID(), payload: {...context.draft}};
  ticketBusy = true;
  context.note.textContent = "Creating your approved ticket…";
  updateComposer();
  renderProposal(context);
  try {
    ticketCreated(context, await sendTicketRequest(context), approval);
  } catch (error) {
    context.note.textContent = creationError(error);
    context.creationActivity?.remove();
    context.creationActivity = element("div", "creation-activity");
    appendRequestActivity(context.creationActivity, {...context.agentResponse, completed: false}, context.agentResponse?.requestId);
    context.section.append(context.creationActivity);
  } finally {
    ticketBusy = false;
    renderProposal(context);
    updateComposer();
    scrollToLatest();
  }
}

function draftTicket(context) {
  const sources = [...collectSources(context.response).values()].map(source =>
    `${source.reference} ${source.source_file}, page ${source.page_number}`
  ).join("\n");
  // A suggestion is not evidence that the user actually tried a troubleshooting step.
  const description = `Issue reported:\n${context.question}\n\nAssistant response (review before sending; not confirmed as tried):\n${context.response.answer}` +
    (sources ? `\n\nRetrieved sources:\n${sources}` : "");
  return {title: context.question.slice(0, 200), description: description.slice(0, 5000), priority: "medium"};
}

function readTicketFields() {
  return {title: ticketTitle.value.trim(), description: ticketDescription.value.trim(), priority: ticketPriority.value};
}

function updateTicketControls() {
  const locked = ticketBusy || Boolean(ticketContext?.attempt);
  ticketTitle.readOnly = locked;
  ticketDescription.readOnly = locked;
  ticketPriority.disabled = locked;
  ticketSubmit.disabled = ticketBusy || Boolean(ticketContext?.conflict);
  ticketClose.disabled = ticketBusy;
  ticketCancel.disabled = ticketBusy;
  ticketSubmit.textContent = ticketBusy ? "Creating ticket…" :
    ticketContext?.attempt ? "Retry creation" : "Confirm & create ticket";
  ticketRetryNote.hidden = !ticketContext?.attempt || ticketBusy || Boolean(ticketContext?.conflict);
  updateComposer();
}

function renderTicketReceipt(context) {
  const ticket = context.ticket;
  document.querySelector("#ticket-heading").textContent = `Ticket #${ticket.id} created`;
  ticketForm.hidden = true;
  document.querySelector("#ticket-receipt").hidden = false;
  const details = document.querySelector("#ticket-receipt-details");
  details.replaceChildren();
  details.append(element("h3", "", ticket.title));
  const badges = element("div", "ticket-badges");
  badges.append(element("span", "ticket-badge", `Status: ${ticket.status.replace(/_/g, " ")}`),
    element("span", "ticket-badge", `Priority: ${ticket.priority}`));
  details.append(badges);
  const description = element("p", "ticket-saved-description", ticket.description);
  description.dir = "auto";
  details.append(description);
}

function openTicket(context) {
  if (busy || ticketBusy) return;
  if (context.viaAgent && !context.ticket && pendingProposal !== context) return;
  ticketContext = context;
  ticketError.hidden = !context.errorText;
  ticketError.textContent = context.errorText || "";
  if (context.ticket) {
    renderTicketReceipt(context);
  } else {
    context.draft ||= draftTicket(context);
    document.querySelector("#ticket-heading").textContent = "Let’s get you some help.";
    ticketForm.hidden = false;
    document.querySelector("#ticket-receipt").hidden = true;
    ticketTitle.value = context.draft.title;
    ticketDescription.value = context.draft.description;
    ticketPriority.value = context.draft.priority;
  }
  updateTicketControls();
  ticketDialog.showModal();
}

function closeTicket() {
  if (ticketBusy) return;
  if (ticketContext && !ticketContext.ticket && !ticketContext.attempt) {
    ticketContext.draft = readTicketFields();
    if (ticketContext.viaAgent) renderProposal(ticketContext);
  }
  ticketDialog.close();
}

async function createTicket(event) {
  event.preventDefault();
  const context = ticketContext;
  if (!context || context.ticket || context.conflict || ticketBusy) return;
  if (context.viaAgent && pendingProposal !== context) return;
  ticketError.hidden = true;
  context.errorText = null;
  const payload = context.attempt?.payload || readTicketFields();
  if (!payload.title || !payload.description || payload.title.length > 200 || payload.description.length > 5000) {
    ticketError.textContent = "Enter a title of up to 200 characters and a description of up to 5,000 characters.";
    ticketError.hidden = false;
    return;
  }
  // Freeze both key and approved body. A timeout may occur after the database committed.
  context.attempt ||= {key: crypto.randomUUID(), payload};
  context.draft = payload;
  ticketBusy = true;
  updateTicketControls();
  if (context.viaAgent) renderProposal(context);
  try {
    ticketCreated(context, await sendTicketRequest(context));
    renderTicketReceipt(context);
  } catch (error) {
    context.errorText = creationError(error);
    ticketError.textContent = context.errorText;
    ticketError.hidden = false;
    if (context.viaAgent) context.note.textContent = context.errorText;
  } finally {
    ticketBusy = false;
    updateTicketControls();
    if (context.viaAgent) renderProposal(context);
  }
}

const errorMessages = {
  422: "The question or conversation context was rejected. Try a shorter question or start a new chat.",
  502: "The AI service could not complete the request. Please try again shortly.",
  503: "The assistant is temporarily unavailable. Please try again when the service is ready.",
  504: "The request took too long. Please try again.",
};

async function ask(question, retryCard = null, snapshot = null) {
  if (busy || ticketBusy || !question.trim() || question.length > 2000) return;
  const reply = question.trim().toLowerCase().replace(/[.!?]+$/, "").trim();
  if (pendingProposal?.attempt && (retryCard || !["yes", "כן", "confirm", "confirm ticket", "מאשר", "מאשרת", "אשר"].includes(reply))) {
    pendingProposal.note.textContent = "Creation is not confirmed yet. Keep this chat open and retry this same request.";
    scrollToLatest();
    return;
  }
  if (!retryCard && pendingProposal) {
    if (["yes", "כן", "confirm", "confirm ticket", "מאשר", "מאשרת", "אשר"].includes(reply)) {
      input.value = "";
      await confirmProposal(pendingProposal, question);
      return;
    }
    if (pendingProposal.attempt) {
      pendingProposal.note.textContent = "Creation is not confirmed yet. Retry this same request before continuing the conversation.";
      scrollToLatest();
      return;
    }
    if (["no", "לא", "cancel", "בטל", "בטלי"].includes(reply)) {
      dismissProposal(pendingProposal);
      input.value = "";
      updateComposer();
      return;
    }
    dismissProposal(pendingProposal, "This proposal is inactive after your new message. Ask for an updated proposal to confirm.");
  }
  snapshot ||= {history: chatHistory.map(item => ({...item})), revision: historyRevision};
  busy = true;
  welcome.hidden = true;
  conversation.hidden = false;
  if (!retryCard) {
    const message = element("article", "message");
    const text = element("div", "user-message", question);
    text.dir = "auto";
    message.append(text);
    conversation.append(message);
  }
  const card = retryCard || assistantMessage();
  showThinking(card);
  input.value = "";
  updateComposer();
  status.textContent = "The assistant is working on your question.";
  scrollToLatest();
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120000);
  let failureResponse = null;
  let requestId = "";
  try {
    const result = await fetch("/agent", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({message: question, history: snapshot.history, allowed_actions: []}), signal: controller.signal,
    });
    requestId = result.headers.get("X-Request-ID") || "";
    const response = await result.json().catch(() => null);
    if (!result.ok) {
      failureResponse = response;
      throw new Error(errorMessages[result.status] || "Something went wrong. Please try again.");
    }
    if (!response || typeof response.answer !== "string" || !Array.isArray(response.steps) || typeof response.completed !== "boolean") {
      throw new Error("The assistant returned an unexpected response. Please try again.");
    }
    if (response.completed && response.answer.trim()) {
      const rememberedAnswer = validTicketDraft(response.ticket_proposal) ?
        `Ticket proposal awaiting explicit confirmation:\n${JSON.stringify(response.ticket_proposal)}` : response.answer;
      rememberExchange(question, rememberedAnswer, snapshot.revision);
    }
    showResponse(card, response, question);
    status.textContent = response.completed ? "Answer ready. You can open its sources." : "The assistant could not finish this request.";
  } catch (error) {
    card.replaceChildren();
    card.classList.add("error-card");
    const message = error.name === "AbortError" ? "The request took too long. Please try again." :
      error instanceof TypeError ? "Could not reach the assistant. Check your connection and try again." : error.message;
    card.append(element("div", "answer-text", message));
    const retry = element("button", "text-button", "Try again ↗");
    retry.type = "button";
    retry.addEventListener("click", () => ask(question, card, snapshot));
    card.append(retry);
    appendRequestActivity(card, {...failureResponse, completed: false}, requestId);
    status.textContent = message;
  } finally {
    clearTimeout(timeout);
    busy = false;
    if (pendingProposal) renderProposal(pendingProposal);
    updateComposer();
    scrollToLatest();
    input.focus({preventScroll: true});
  }
}

form.addEventListener("submit", event => { event.preventDefault(); ask(input.value.trim()); });
input.addEventListener("input", updateComposer);
input.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!send.disabled) form.requestSubmit();
  }
});
document.querySelectorAll(".suggestion").forEach(button => {
  button.addEventListener("click", () => {
    input.value = button.dataset.question;
    updateComposer();
    input.focus();
  });
});
function newChat() {
  if (busy || ticketBusy) return;
  if (pendingProposal?.attempt && !pendingProposal.ticket) return;
  dismissProposal(pendingProposal);
  pendingProposal = null;
  chatHistory = [];
  historyRevision += 1;
  conversation.replaceChildren();
  conversation.hidden = true;
  welcome.hidden = false;
  input.value = "";
  status.textContent = "New chat started.";
  updateComposer();
  scrollArea.scrollTop = 0;
  input.focus();
}
reset.addEventListener("click", newChat);
mobileReset.addEventListener("click", newChat);
document.querySelector("#close-source").addEventListener("click", () => dialog.close());
dialog.addEventListener("click", event => {
  if (event.target === dialog) {
    const bounds = dialog.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
  }
});
ticketForm.addEventListener("submit", createTicket);
ticketClose.addEventListener("click", closeTicket);
ticketCancel.addEventListener("click", closeTicket);
document.querySelector("#ticket-done").addEventListener("click", closeTicket);
ticketDialog.addEventListener("cancel", event => { event.preventDefault(); closeTicket(); });
ticketDialog.addEventListener("close", () => { ticketContext = null; });
updateComposer();
