"use client";

import { useEffect } from "react";

import { hashTargetId } from "../lib/hash-target";

export function HashAnchorScroll() {
  useEffect(() => {
    const targetId = hashTargetId(window.location.hash);
    if (!targetId) return;

    let cancelled = false;
    let frame: number | null = null;
    void document.fonts.ready.then(() => {
      if (cancelled) return;
      frame = window.requestAnimationFrame(() => {
        const target = document.getElementById(targetId);
        if (!(target instanceof HTMLElement)) return;
        target.focus({ preventScroll: true });
        target.scrollIntoView({ block: "start" });
      });
    });

    return () => {
      cancelled = true;
      if (frame !== null) window.cancelAnimationFrame(frame);
    };
  }, []);

  return null;
}
