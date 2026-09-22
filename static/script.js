const $ = (id) => document.getElementById(id);

function renderTerms(list, elementId) {
  const el = $(elementId);
  el.innerHTML = "";
  if (!list || !list.length) {
    el.innerHTML = "<li>No strong terms identified.</li>";
    return;
  }
  list.forEach(item => {
    const li = document.createElement("li");
    li.innerHTML = `<strong>${escapeHtml(item.term)}</strong> <span>(${item.impact})</span>`;
    el.appendChild(li);
  });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderLinguistic(features) {
  const labels = {
    word_count: "Word count",
    sentence_count: "Sentences",
    average_sentence_length: "Avg. sentence length",
    uppercase_word_ratio: "Uppercase ratio",
    exclamation_marks: "Exclamation marks",
    question_marks: "Question marks"
  };
  $("linguistic").innerHTML = "";
  Object.entries(labels).forEach(([key, label]) => {
    const div = document.createElement("div");
    div.className = "stat";
    div.innerHTML = `<span>${label}</span><strong>${escapeHtml(features[key])}</strong>`;
    $("linguistic").appendChild(div);
  });
}

$("analyzeBtn").addEventListener("click", async () => {
  $("status").textContent = "Analyzing...";
  $("status").className = "status";

  const payload = {
    title: $("title").value.trim(),
    text: $("text").value.trim()
  };

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    const data = await response.json();

    if (!response.ok) throw new Error(data.error || "Analysis failed.");

    $("result").classList.remove("hidden");
    $("credibility").textContent = data.credibility;
    $("confidence").textContent = data.confidence + "%";
    $("confidenceBar").style.width = data.confidence + "%";
    $("prediction").textContent = data.prediction;
    $("realProb").textContent = data.real_probability + "%";
    $("fakeProb").textContent = data.fake_probability + "%";

    renderTerms(data.explanation.supports_real, "realTerms");
    renderTerms(data.explanation.supports_fake, "fakeTerms");
    renderLinguistic(data.linguistic_features);
    $("notice").textContent = data.notice;
    $("status").textContent = "Analysis completed.";
  } catch (err) {
    $("status").textContent = err.message;
    $("status").className = "status error";
  }
});

$("clearBtn").addEventListener("click", () => {
  $("title").value = "";
  $("text").value = "";
  $("result").classList.add("hidden");
  $("status").textContent = "";
});
