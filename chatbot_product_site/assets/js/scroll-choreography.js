/**
 * Threadmind scroll choreography — GSAP ScrollTrigger section washes,
 * spotlight, and staggered reveals. Peak sections pop copper/signal;
 * dip sections desaturate for measurable engagement contrast on capture.
 */
(function () {
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const root = document.documentElement;
  const journey = document.querySelector(".journey");
  if (!journey || reduced || typeof gsap === "undefined") return;

  gsap.registerPlugin(ScrollTrigger);

  const ENGAGE = {
    peak: { r: 196, g: 137, b: 90, wash: 0.22, sat: 1.12, bright: 1.08, spot: 0.9 },
    rise: { r: 109, g: 184, b: 138, wash: 0.12, sat: 1.05, bright: 1.02, spot: 0.45 },
    neutral: { r: 20, g: 18, b: 16, wash: 0.04, sat: 1, bright: 1, spot: 0.15 },
    dip: { r: 14, g: 13, b: 12, wash: 0, sat: 0.72, bright: 0.88, spot: 0 },
  };

  const sections = Array.from(journey.querySelectorAll("section[data-engage]"));

  function applyEngage(level, yPct) {
    const cfg = ENGAGE[level] || ENGAGE.neutral;
    root.style.setProperty("--wash-r", String(cfg.r));
    root.style.setProperty("--wash-g", String(cfg.g));
    root.style.setProperty("--wash-b", String(cfg.b));
    root.style.setProperty("--wash-opacity", String(cfg.wash));
    root.style.setProperty("--section-saturate", String(cfg.sat));
    root.style.setProperty("--section-brightness", String(cfg.bright));
    root.style.setProperty("--spotlight-strength", String(cfg.spot));
    root.style.setProperty("--spotlight-y", `${Math.round(yPct)}%`);
  }

  sections.forEach((sec) => {
    const level = sec.dataset.engage || "neutral";
    const cfg = ENGAGE[level] || ENGAGE.neutral;

    ScrollTrigger.create({
      trigger: sec,
      start: "top 70%",
      end: "bottom 30%",
      onEnter: () => applyEngage(level, 38),
      onEnterBack: () => applyEngage(level, 38),
      onLeave: () => {},
      onLeaveBack: () => {},
    });

    if (level === "peak" || level === "rise") {
      gsap.fromTo(
        sec.querySelectorAll("h1, h2"),
        { scale: 0.94, opacity: 0.7 },
        {
          scale: 1,
          opacity: 1,
          duration: 0.9,
          ease: "power3.out",
          scrollTrigger: {
            trigger: sec,
            start: "top 75%",
            toggleActions: "play none none reverse",
          },
        },
      );
    }

    if (level === "dip") {
      gsap.to(sec, {
        opacity: 0.88,
        scrollTrigger: {
          trigger: sec,
          start: "top 60%",
          end: "bottom 40%",
          scrub: 1.5,
        },
      });
    }

    sec.querySelectorAll(".cost-card, .model-card, .price-tier").forEach((card, i) => {
      gsap.fromTo(
        card,
        { y: 32, opacity: 0, rotateX: level === "peak" ? -6 : 0 },
        {
          y: 0,
          opacity: 1,
          rotateX: 0,
          duration: 0.75,
          delay: i * 0.08,
          ease: "power2.out",
          scrollTrigger: {
            trigger: card,
            start: "top 88%",
            toggleActions: "play none none reverse",
          },
        },
      );
    });

    if (level === "peak") {
      sec.querySelectorAll(".btn-primary").forEach((btn) => {
        gsap.fromTo(
          btn,
          { scale: 0.92 },
          {
            scale: 1,
            duration: 0.6,
            ease: "back.out(1.4)",
            scrollTrigger: {
              trigger: btn,
              start: "top 90%",
              toggleActions: "play none none reverse",
            },
          },
        );
      });
    }
  });

  const heroTitle = document.getElementById("hero-title");
  if (heroTitle) {
    gsap.fromTo(
      heroTitle,
      { y: 40, opacity: 0 },
      {
        y: 0,
        opacity: 1,
        duration: 1.1,
        ease: "power3.out",
        delay: 0.15,
      },
    );
  }

  const heroDemo = document.getElementById("hero-demo");
  if (heroDemo) {
    gsap.to(heroDemo, {
      y: -24,
      scrollTrigger: {
        trigger: "#hero",
        start: "top top",
        end: "bottom top",
        scrub: 1.8,
      },
    });
  }

  applyEngage("peak", 40);
})();
