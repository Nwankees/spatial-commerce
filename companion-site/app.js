(() => {
  const config = window.WILL_IT_FIT_CONFIG || {};
  const apiBase = String(config.apiBaseUrl || "").replace(/\/$/, "");
  const note = document.querySelector("#service-note");

  const updateCard = (name, info) => {
    const node = document.querySelector(`[data-service="${name}"]`);
    if (!node || !info) return;
    const configured = Boolean(info.configured);
    node.textContent = configured
      ? `${info.provider} · configured`
      : `${info.provider} · safe local fallback`;
    node.classList.toggle("live", configured);
  };

  fetch(`${apiBase}/api/v1/sponsor/status`, { headers: { Accept: "application/json" } })
    .then((response) => {
      if (!response.ok) throw new Error("status unavailable");
      return response.json();
    })
    .then((status) => {
      updateCard("persistence", status.persistence);
      updateCard("memory", status.memory);
      updateCard("voice", status.voice);
      note.textContent = "Live configuration status from the Will It Fit? backend.";
    })
    .catch(() => {
      note.textContent = "The demo site remains available while optional backend services are offline.";
    });
})();
