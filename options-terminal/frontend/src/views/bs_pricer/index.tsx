// bs_pricer bespoke view (guide §7). Registered by its slot's display.default_view.
import { registerView, type BespokeProps } from "../registry.ts";
import { AssetView } from "./AssetView.tsx";
import { Monitor } from "./Monitor.tsx";

function BsPricerView(props: BespokeProps) {
  return props.scope.underlying ? <AssetView {...props} underlying={props.scope.underlying} /> : <Monitor {...props} />;
}

registerView("bs_pricer", BsPricerView);
