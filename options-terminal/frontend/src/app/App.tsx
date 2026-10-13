// Desk shell (front_end_design.md §4): identity bar, function keys, command line,
// message line, scrolling work area, suggested-functions bar. Keyboard first (§7).
import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import { usePolled } from "../api/usePolled.ts";
import { checkSlotList } from "../api/validate.ts";
import { formatUtc } from "../format/units.ts";
import { Catalog, slotHref, sortSlots } from "../views/Catalog.tsx";
import { SlotView } from "../views/SlotView.tsx";
import { parseHash, type Route } from "./router.ts";

function useRoute(): Route {
  const [route, setRoute] = useState(() => parseHash(window.location.hash));
  useEffect(() => {
    const onHash = () => setRoute(parseHash(window.location.hash));
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  return route;
}

export function App() {
  const route = useRoute();
  const slots = usePolled("/v1/slots", checkSlotList);
  const [asOf, setAsOf] = useState<string | null>(null);
  const [message, setMessage] = useState("TYPE A SLOT NUMBER OR ID AND PRESS <GO> (ENTER)");
  const [flash, setFlash] = useState(false);
  const cmdRef = useRef<HTMLInputElement>(null);
  const onAsOf = useCallback((v: string | null) => setAsOf(v), []);

  const say = (text: string) => {
    setMessage(text);
    setFlash(true);
    setTimeout(() => setFlash(false), 150);
  };

  const goCatalog = () => {
    window.location.hash = "#/";
    say("CATALOG");
  };

  const runCommand = (raw: string) => {
    const cmd = raw.trim().toLowerCase();
    if (!cmd) return;
    if (cmd === "cat" || cmd === "menu") return goCatalog();
    const ordered = sortSlots(slots.data ?? []);
    const slot = /^\d+$/.test(cmd) ? ordered[Number(cmd) - 1] : ordered.find((s) => s.slot_id === cmd);
    if (!slot) return say(`UNKNOWN COMMAND ${cmd.toUpperCase()}`);
    window.location.hash = slotHref(slot);
    say(slot.title.toUpperCase());
  };

  const onSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const input = cmdRef.current;
    if (!input) return;
    runCommand(input.value);
    input.value = "";
  };

  useEffect(() => {
    document.documentElement.classList.add("desk-page");
    const onKey = (e: KeyboardEvent) => {
      const input = cmdRef.current;
      if (e.key === "F1") {
        e.preventDefault();
        goCatalog();
      } else if (e.key === "Escape" && input) {
        input.value = "";
        input.blur();
        say("CANCELLED");
      } else if (
        input &&
        e.key.length === 1 &&
        !e.ctrlKey &&
        !e.metaKey &&
        !e.altKey &&
        !(e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement)
      ) {
        input.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className="desk">
      <header className="identity-bar">
        <span className="title">Options Terminal</span>
        <span className="dim">
          AS OF <span className="num">{asOf ? formatUtc(asOf) : "—"}</span>
        </span>
      </header>
      <nav className="fkeys" aria-label="Function keys">
        <button className="fkey nav" onClick={goCatalog}>
          F1 Catalog
        </button>
      </nav>
      <form className="cmd-line" onSubmit={onSubmit}>
        <span className="label">CMD &gt;</span>
        <input ref={cmdRef} aria-label="Command" autoComplete="off" spellCheck={false} />
      </form>
      <div className={`msg-line dim${flash ? " flash" : ""}`} role="status">
        {message}
      </div>
      <main className="work-area">
        {route.name === "catalog" && <Catalog slots={slots} />}
        {route.name === "slot" && (
          <SlotView slotId={route.slotId} scope={route.scope} slots={slots} onAsOf={onAsOf} />
        )}
        {route.name === "not_found" && (
          <div className="notice bad">
            UNKNOWN ADDRESS · <a href="#/">BACK TO CATALOG</a>
          </div>
        )}
      </main>
      <footer className="suggest-bar dim">
        <span className="label">Suggested</span>
        {sortSlots(slots.data ?? []).map((s, i) => (
          <button key={s.slot_id} onClick={() => runCommand(String(i + 1))}>
            {i + 1}) {s.slot_id.toUpperCase()}
          </button>
        ))}
      </footer>
    </div>
  );
}
