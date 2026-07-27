(() => {
  "use strict";

  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const editor = document.getElementById("editor");
  const statusLine = document.getElementById("status-line");

  const tabAuto = document.getElementById("tab-auto");
  const tabRefine = document.getElementById("tab-refine");
  const autoView = document.getElementById("auto-view");
  const refineView = document.getElementById("refine-view");
  const autoPreview = document.getElementById("auto-preview");
  const refineResult = document.getElementById("refine-result");

  const newImageBtn = document.getElementById("new-image-btn");
  const downloadBtn = document.getElementById("download-btn");
  const reprocessBtn = document.getElementById("reprocess-btn");

  const modeIncludeBtn = document.getElementById("mode-include-btn");
  const modeExcludeBtn = document.getElementById("mode-exclude-btn");
  const modeBoxBtn = document.getElementById("mode-box-btn");
  const undoBtn = document.getElementById("undo-btn");
  const resetBtn = document.getElementById("reset-btn");
  const restoreAutoBtn = document.getElementById("restore-auto-btn");
  const altThumbs = document.getElementById("alt-thumbs");

  const canvas = document.getElementById("refine-canvas");
  const ctx = canvas.getContext("2d");

  let sessionId = null;
  let baseImage = null; // HTMLImageElement of the original upload
  let clickMode = "include"; // include | exclude | box
  let points = []; // { x, y, label }
  let box = null; // { x1, y1, x2, y2 }
  let history = []; // snapshots of { points, box } for undo
  let dragStart = null;
  let hasResult = false;

  function setStatus(text, busy = false) {
    statusLine.innerHTML = busy ? `<span class="spinner"></span>${text}` : text;
  }

  function resetState() {
    sessionId = null;
    baseImage = null;
    points = [];
    box = null;
    history = [];
    hasResult = false;
    altThumbs.innerHTML = "";
    downloadBtn.disabled = true;
  }

  // ---------- Upload ----------

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    const file = e.dataTransfer.files?.[0];
    if (file) uploadFile(file);
  });
  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (file) uploadFile(file);
  });

  async function uploadFile(file) {
    resetState();
    dropzone.classList.add("hidden");
    editor.classList.remove("hidden");
    switchTab("auto");
    setStatus("Uploading and removing background…", true);

    const form = new FormData();
    form.append("file", file);

    try {
      const res = await fetch("/api/upload-and-auto", { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      sessionId = data.session_id;
      hasResult = true;
      autoPreview.src = `${data.result_url}?t=${Date.now()}`;
      downloadBtn.disabled = false;
      setStatus("");
    } catch (err) {
      console.error(err);
      setStatus("Something went wrong processing this image.");
      backToDropzone();
    }
  }

  function backToDropzone() {
    resetState();
    editor.classList.add("hidden");
    dropzone.classList.remove("hidden");
    fileInput.value = "";
  }

  newImageBtn.addEventListener("click", backToDropzone);

  reprocessBtn.addEventListener("click", async () => {
    if (!sessionId) return;
    setStatus("Reprocessing…", true);
    const res = await fetch("/api/auto", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    });
    if (res.ok) {
      const data = await res.json();
      autoPreview.src = `${data.result_url}?t=${Date.now()}`;
    }
    setStatus("");
  });

  downloadBtn.addEventListener("click", async () => {
    if (!sessionId || !hasResult) return;
    const res = await fetch(`/api/download/${sessionId}`);
    if (!res.ok) return;
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `cutout-${sessionId.slice(0, 8)}.png`;
    a.click();
    URL.revokeObjectURL(url);
  });

  // ---------- Tabs ----------

  function switchTab(name) {
    const refine = name === "refine";
    tabAuto.classList.toggle("active", !refine);
    tabRefine.classList.toggle("active", refine);
    autoView.classList.toggle("hidden", refine);
    refineView.classList.toggle("hidden", !refine);
    if (refine) enterRefineMode();
  }

  tabAuto.addEventListener("click", () => switchTab("auto"));
  tabRefine.addEventListener("click", () => switchTab("refine"));

  async function enterRefineMode() {
    if (!sessionId || baseImage) return;
    setStatus("Preparing refine mode…", true);
    await fetch(`/api/prepare-refine/${sessionId}`, { method: "POST" });
    baseImage = new Image();
    baseImage.onload = () => {
      canvas.width = baseImage.naturalWidth;
      canvas.height = baseImage.naturalHeight;
      redrawCanvas();
      setStatus("");
    };
    baseImage.src = `/api/original/${sessionId}?t=${Date.now()}`;
    refineResult.src = `/api/preview/${sessionId}?t=${Date.now()}`;
  }

  // ---------- Refine canvas ----------

  function redrawCanvas(previewBox) {
    if (!baseImage) return;
    ctx.drawImage(baseImage, 0, 0, canvas.width, canvas.height);

    for (const p of points) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
      ctx.fillStyle = p.label === 1 ? "#34d399" : "#f87171";
      ctx.strokeStyle = "#0f1115";
      ctx.lineWidth = 2;
      ctx.fill();
      ctx.stroke();
    }

    const rect = previewBox || box;
    if (rect) {
      ctx.strokeStyle = "#5b8cff";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.strokeRect(
        Math.min(rect.x1, rect.x2),
        Math.min(rect.y1, rect.y2),
        Math.abs(rect.x2 - rect.x1),
        Math.abs(rect.y2 - rect.y1)
      );
      ctx.setLineDash([]);
    }
  }

  function canvasCoords(evt) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: Math.round((evt.clientX - rect.left) * scaleX),
      y: Math.round((evt.clientY - rect.top) * scaleY),
    };
  }

  function pushHistory() {
    history.push({ points: points.map((p) => ({ ...p })), box: box ? { ...box } : null });
  }

  canvas.addEventListener("mousedown", (e) => {
    if (!baseImage) return;
    if (clickMode === "box") {
      dragStart = canvasCoords(e);
    }
  });

  canvas.addEventListener("mousemove", (e) => {
    if (!baseImage || clickMode !== "box" || !dragStart) return;
    const current = canvasCoords(e);
    redrawCanvas({ x1: dragStart.x, y1: dragStart.y, x2: current.x, y2: current.y });
  });

  canvas.addEventListener("mouseup", (e) => {
    if (!baseImage || clickMode !== "box" || !dragStart) return;
    const current = canvasCoords(e);
    const rect = { x1: dragStart.x, y1: dragStart.y, x2: current.x, y2: current.y };
    dragStart = null;
    if (Math.abs(rect.x2 - rect.x1) < 4 || Math.abs(rect.y2 - rect.y1) < 4) {
      redrawCanvas();
      return;
    }
    pushHistory();
    box = rect;
    redrawCanvas();
    runSegment();
  });

  canvas.addEventListener("click", (e) => {
    if (!baseImage || clickMode === "box") return;
    const { x, y } = canvasCoords(e);
    pushHistory();
    points.push({ x, y, label: clickMode === "include" ? 1 : 0 });
    redrawCanvas();
    runSegment();
  });

  function setClickMode(mode) {
    clickMode = mode;
    modeIncludeBtn.classList.toggle("active", mode === "include");
    modeExcludeBtn.classList.toggle("active", mode === "exclude");
    modeBoxBtn.classList.toggle("active", mode === "box");
    canvas.classList.toggle("mode-exclude", mode === "exclude");
    canvas.classList.toggle("mode-box", mode === "box");
  }

  modeIncludeBtn.addEventListener("click", () => setClickMode("include"));
  modeExcludeBtn.addEventListener("click", () => setClickMode("exclude"));
  modeBoxBtn.addEventListener("click", () => setClickMode("box"));

  undoBtn.addEventListener("click", async () => {
    if (!history.length) return;
    const prev = history.pop();
    points = prev.points;
    box = prev.box;
    redrawCanvas();
    if (!points.length && !box) {
      await resetRefine();
    } else {
      await runSegment();
    }
  });

  resetBtn.addEventListener("click", resetRefine);

  async function resetRefine() {
    if (!sessionId) return;
    points = [];
    box = null;
    history = [];
    altThumbs.innerHTML = "";
    redrawCanvas();
    const res = await fetch(`/api/reset/${sessionId}`, { method: "POST" });
    if (res.ok) {
      const data = await res.json();
      refineResult.src = `${data.preview_url}?t=${Date.now()}`;
    }
  }

  restoreAutoBtn.addEventListener("click", async () => {
    if (!sessionId) return;
    points = [];
    box = null;
    history = [];
    altThumbs.innerHTML = "";
    redrawCanvas();
    const res = await fetch(`/api/restore-auto/${sessionId}`, { method: "POST" });
    if (res.ok) {
      const data = await res.json();
      refineResult.src = `${data.result_url}?t=${Date.now()}`;
      autoPreview.src = `${data.result_url}?t=${Date.now()}`;
    }
  });

  async function runSegment(maskIndex) {
    if (!sessionId) return;
    setStatus("Refining…", true);
    const payload = { session_id: sessionId, points, box };
    if (maskIndex !== undefined) payload.mask_index = maskIndex;

    try {
      const res = await fetch("/api/segment", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      hasResult = true;
      downloadBtn.disabled = false;
      refineResult.src = `${data.result_url}?t=${Date.now()}`;
      autoPreview.src = `${data.result_url}?t=${Date.now()}`;
      renderAlternatives(data.alternatives, maskIndex);
      setStatus("");
    } catch (err) {
      console.error(err);
      setStatus("Could not refine the cutout.");
    }
  }

  function renderAlternatives(alternatives, selectedIndex) {
    altThumbs.innerHTML = "";
    if (!alternatives || !alternatives.length) return;
    alternatives.forEach((alt) => {
      const wrap = document.createElement("button");
      wrap.type = "button";
      wrap.className = "alt-thumb" + (alt.selected ? " selected" : "");
      wrap.title = `Score ${alt.score}`;
      const img = document.createElement("img");
      img.src = `data:image/png;base64,${alt.thumbnail}`;
      wrap.appendChild(img);
      wrap.addEventListener("click", () => runSegment(alt.index));
      altThumbs.appendChild(wrap);
    });
  }
})();
