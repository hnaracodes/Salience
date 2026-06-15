interface ViewerFrameProps {
  src: string
}

export function ViewerFrame({ src }: ViewerFrameProps) {
  // allow-scripts is required: the viewer renders a Three.js brain surface and
  // reads viewer_bundle.json via fetch.
  // allow-same-origin is intentionally omitted: the viewer is served from R2
  // (a different origin than this app), so the combination of allow-scripts +
  // allow-same-origin would grant the iframe's JS access to its own origin's
  // storage — unnecessary since the R2 viewer has no session cookies to protect.
  // The viewer fetches viewer_bundle.json via a relative path, which works
  // cross-origin from R2 because R2 CORS allows the Vercel origin.
  return (
    <iframe
      src={src}
      className="w-full h-[800px] rounded-lg border border-ink-700"
      title="UX Analysis Viewer"
      sandbox="allow-scripts"
    />
  )
}
