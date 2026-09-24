/* This asset is global so instant navigation from any document can open home. */
(() => {
  let dispose = () => {};

  function mountHome() {
    dispose();
    dispose = () => {};
    const root = document.querySelector(".flowcean-home");
    if (!root) return;

    const lifetime = new AbortController();
    const { signal } = lifetime;
    const figure = root.querySelector(".home-experiment");
    const play = root.querySelector(".home-play");
    const playLabel = play.querySelector(".home-play-label");
    const copy = root.querySelector(".home-copy");
    const copyStatus = root.querySelector(".home-copy-status");
    const temperature = root.querySelector("[data-temperature]");
    const modeLabel = root.querySelector("[data-mode-label]");
    const timeLabel = root.querySelector("[data-time]");
    const gauge = root.querySelector("[data-gauge]");
    const cursor = root.querySelector("[data-cursor]");
    const point = root.querySelector("[data-point]");
    const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
    let playing = !motion.matches;
    let visible = false;
    let points = null;
    let duration = 12;
    let time = 0;
    let frame = 0;
    let previous = null;
    let lastPaint = 0;
    let index = 0;

    function updateButton() {
      const label = playing ? "Pause" : "Play";
      playLabel.textContent = label;
      play.dataset.playing = String(playing);
      play.setAttribute("aria-label", `${label} simulation replay`);
    }

    function paint() {
      while (index < points.length - 2 && points[index + 1][0] <= time) index++;
      const a = points[index];
      const b = points[index + 1];
      const fraction = Math.min(1, Math.max(0, (time - a[0]) / (b[0] - a[0])));
      const value = a[1] + (b[1] - a[1]) * fraction;
      const mode = time >= b[0] ? b[2] : a[2];
      const x = 38 + time / duration * 502;
      const y = 170 - (value - 20) / 3 * 140;
      temperature.textContent = value.toFixed(2);
      timeLabel.textContent = time.toFixed(2);
      modeLabel.textContent = mode === "heating" ? "Heating" : "Cooling";
      figure.dataset.mode = mode;
      gauge.setAttribute("cx", String(8 + (value - 20) / 3 * 214));
      cursor.setAttribute("d", `M${x} 24V174`);
      point.setAttribute("cx", String(x));
      point.setAttribute("cy", String(y));
    }

    function tick(now) {
      frame = 0;
      let wrapped = false;
      if (previous !== null) {
        time += (now - previous) / 2000;
        if (time >= duration) {
          time %= duration;
          index = 0;
          wrapped = true;
        }
      }
      previous = now;
      if (wrapped || now - lastPaint >= 50) {
        paint();
        lastPaint = now;
      }
      frame = requestAnimationFrame(tick);
    }

    function syncPlayback() {
      cancelAnimationFrame(frame);
      frame = 0;
      previous = null;
      if (points && playing && visible && !document.hidden && !signal.aborted) {
        frame = requestAnimationFrame(tick);
      }
    }

    const observer = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
      syncPlayback();
    }, { threshold: 0.1 });
    observer.observe(figure);

    dispose = () => {
      lifetime.abort();
      observer.disconnect();
      cancelAnimationFrame(frame);
    };

    play.addEventListener("click", () => {
      if (!points) return;
      playing = !playing;
      updateButton();
      syncPlayback();
    }, { signal });
    document.addEventListener("visibilitychange", syncPlayback, { signal });
    motion.addEventListener("change", () => {
      // A new reduced-motion preference always stops replay. Resuming is explicit.
      if (motion.matches) {
        playing = false;
        updateButton();
        syncPlayback();
      }
    }, { signal });

    copy.disabled = false;
    copy.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText("pip install flowcean");
        if (!signal.aborted) copyStatus.textContent = "Copied to clipboard.";
      } catch {
        if (!signal.aborted) copyStatus.textContent = "Copy unavailable. Select and copy the command above.";
      }
    }, { signal });

    async function loadTrace() {
      try {
        const response = await fetch(figure.dataset.source, { signal });
        if (!response.ok) throw new Error("Trace unavailable");
        const data = await response.json();
        if (signal.aborted) return;
        if (data.duration !== 12 || !Array.isArray(data.samples) || data.samples.length < 2 || !Array.isArray(data.events)) {
          throw new Error("Invalid trace");
        }
        const rows = data.samples.concat(data.events.map(event => [event.time, event.temperature, event.target]));
        rows.sort((a, b) => a[0] - b[0]);
        if (rows.some((row, i) => !Number.isFinite(row[0]) || !Number.isFinite(row[1]) ||
          !["heating", "cooling"].includes(row[2]) || (i && row[0] <= rows[i - 1][0])) ||
          rows[0][0] !== 0 || rows[rows.length - 1][0] !== data.duration) {
          throw new Error("Invalid trace samples");
        }
        points = rows;
        duration = data.duration;
        play.disabled = false;
        updateButton();
        syncPlayback();
      } catch {
        if (signal.aborted) return;
        playing = false;
        updateButton();
        play.disabled = true;
        root.querySelector(".home-replay-status").textContent = "Precomputed Flowcean simulation · replay unavailable";
      }
    }
    loadTrace();
  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(mountHome);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mountHome, { once: true });
  } else {
    mountHome();
  }
})();
