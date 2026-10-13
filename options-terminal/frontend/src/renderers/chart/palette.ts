// Series and scale colours (front_end_design.md §2, §5). Amber and cyan are the only series hues;
// green/red/yellow are reserved for status. Up to four series: amber, cyan, then dashed variants
// (secondary encoding, never a new hue). A fifth series is a design error: callers cap at four.
export const AMBER = "#fb8b1e";
export const CYAN = "#37d0e6";
const TRACK = [0x1a, 0x1a, 0x1a]; // --bar-track
const AMBER_RGB = [0xfb, 0x8b, 0x1e];

export interface SeriesStyle {
  color: string;
  dash?: string;
}

export const SERIES_STYLES: SeriesStyle[] = [
  { color: AMBER },
  { color: CYAN, dash: "6 4" },
  { color: AMBER, dash: "2 3" },
  { color: CYAN, dash: "10 3 2 3" },
];

/** Sequential single-hue scale from the track colour to amber (magnitude). t ∈ [0, 1]. */
export function heat(t: number): string {
  const c = Math.min(1, Math.max(0, t));
  const mix = TRACK.map((lo, i) => Math.round(lo + ((AMBER_RGB[i] as number) - lo) * (0.15 + 0.85 * c)));
  return `rgb(${mix[0]},${mix[1]},${mix[2]})`;
}
