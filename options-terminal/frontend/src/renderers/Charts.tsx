import type { Curve, SurfacePayload, Timeseries } from "../api/payloads.ts";
import { formatUtc, formatValue } from "../format/units.ts";
import { Heatmap } from "./chart/Heatmap.tsx";
import { LineChart } from "./chart/LineChart.tsx";

export function CurveChart({ payload, title }: { payload: Curve; title: string }) {
  const yUnit = payload.series[0]?.unit ?? payload.x.unit;
  return (
    <LineChart
      title={title}
      x={payload.x.values}
      xLabel={payload.x.label}
      series={payload.series.slice(0, 4)}
      formatX={(v) => formatValue(v, payload.x.unit)}
      formatY={(v) => formatValue(v, yUnit)}
      footer={
        payload.series.length > 4
          ? `${payload.x.label} on x. Showing 4 of ${payload.series.length} series; all are in the table and JSON export.`
          : `${payload.x.label} on x; gaps are missing values, never zero.`
      }
    />
  );
}

export function TimeseriesChart({ payload, title }: { payload: Timeseries; title: string }) {
  const xs = payload.t.map((t) => Date.parse(t));
  const yUnit = payload.series[0]?.unit;
  return (
    <LineChart
      title={title}
      x={xs}
      xLabel="Time (UTC)"
      series={payload.series.slice(0, 4)}
      formatX={(v) => formatUtc(new Date(v).toISOString())}
      formatY={(v) => (yUnit ? formatValue(v, yUnit) : String(v))}
      footer="Times are UTC; gaps are missing values, never zero."
    />
  );
}

export function SurfaceHeatmap({ payload, title }: { payload: SurfacePayload; title: string }) {
  return (
    <Heatmap
      title={title}
      x={payload.x.values}
      y={payload.y.values}
      z={payload.z.values}
      xLabel={payload.x.label}
      yLabel={payload.y.label}
      formatX={(v) => formatValue(v, payload.x.unit, { decimals: 2 })}
      formatY={(v) => formatValue(v, payload.y.unit, { decimals: 1 })}
      formatZ={(v) => formatValue(v, payload.z.unit, { decimals: 1 })}
      footer={`${payload.z.label} by ${payload.x.label} and ${payload.y.label}. Blank cells have no data and are not interpolated.`}
    />
  );
}
