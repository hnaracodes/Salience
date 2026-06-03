/**
 * Threadmind scroll ball — GSAP ScrollTrigger choreography.
 * Ball rolls down the rail, rotates, and morphs color per section phase.
 */
(function () {
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const journey = document.querySelector(".journey");
  const ballHost = document.getElementById("scrollBallHost");
  const ballInner = document.getElementById("scrollBallInner");
  const sphere = document.getElementById("scrollBallSphere");
  const trail = document.getElementById("scrollBallTrail");
  const markers = document.querySelectorAll(".section-marker");

  if (!journey || !ballHost || reduced || typeof gsap === "undefined") {
    if (ballHost) ballHost.style.top = "0";
    document.querySelectorAll(".chat-bubble, .cost-card").forEach((el) => {
      el.classList.add("visible", "revealed");
    });
    return;
  }

  gsap.registerPlugin(ScrollTrigger);

  const sections = [
    { id: "hero", phase: "intro", marker: 0 },
    { id: "why-subscribe", phase: "reason", marker: 1 },
    { id: "how-it-works", phase: "compute", marker: 2 },
    { id: "models-preview", phase: "compute", marker: 3 },
    { id: "pricing-teaser", phase: "trust", marker: 4 },
    { id: "testimonials", phase: "trust", marker: 5 },
    { id: "faq", phase: "reason", marker: 6 },
    { id: "cta", phase: "intro", marker: 7 },
  ];

  const rail = document.querySelector(".scroll-rail-track");
  const railHeight = () => (rail ? rail.offsetHeight : window.innerHeight * 0.8);

  const tl = gsap.timeline({
    scrollTrigger: {
      trigger: journey,
      start: "top top",
      end: "bottom bottom",
      scrub: 1.2,
      onUpdate: (self) => updatePhase(self.progress),
    },
  });

  tl.to(ballHost, {
    top: () => railHeight(),
    ease: "none",
  }, 0);

  tl.to(ballInner, {
    rotateX: 720,
    rotateY: 1080,
    rotateZ: 360,
    ease: "none",
  }, 0);

  if (trail) {
    tl.to(trail, {
      height: () => railHeight(),
      ease: "none",
    }, 0);
  }

  function updatePhase(progress) {
    const idx = Math.min(
      sections.length - 1,
      Math.floor(progress * sections.length),
    );
    const sec = sections[idx];
    if (sphere && sec) sphere.dataset.phase = sec.phase;
    markers.forEach((m, i) => {
      m.classList.toggle("is-active", i === sec.marker);
    });
    document.querySelectorAll(".step").forEach((step, i) => {
      step.classList.toggle("is-active", sec.id === "how-it-works" && i <= idx % 4);
    });
  }

  sections.forEach((sec) => {
    const el = document.getElementById(sec.id);
    if (!el) return;
    ScrollTrigger.create({
      trigger: el,
      start: "top 65%",
      onEnter: () => revealSection(el),
      onEnterBack: () => revealSection(el),
    });
  });

  function revealSection(el) {
    el.querySelectorAll(".chat-bubble").forEach((b, i) => {
      setTimeout(() => b.classList.add("visible"), i * 120);
    });
    el.querySelectorAll(".cost-card").forEach((c, i) => {
      setTimeout(() => c.classList.add("revealed"), i * 100);
    });
  }

  revealSection(document.getElementById("hero"));
})();
