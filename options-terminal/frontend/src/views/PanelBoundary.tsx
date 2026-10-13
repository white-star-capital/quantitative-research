import { Component, type ReactNode } from "react";

// A rendering exception in one panel must not take down the page (FRONTEND.md F-19).
export class PanelBoundary extends Component<{ children: ReactNode }, { message: string | null }> {
  state = { message: null as string | null };

  static getDerivedStateFromError(err: unknown) {
    return { message: err instanceof Error ? err.message : String(err) };
  }

  render() {
    if (this.state.message !== null) {
      return <div className="notice bad">RENDER FAILED · {this.state.message}</div>;
    }
    return this.props.children;
  }
}
