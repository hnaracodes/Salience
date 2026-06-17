interface ViewerFrameProps {
  src: string
}

export function ViewerFrame({ src }: ViewerFrameProps) {
  // allow-scripts + allow-same-origin: the viewer fetches sibling assets (viewer_bundle.json,
  // walkthrough.webm, brain_surface.js) via relative paths from the iframe document URL.
  // allow-same-origin is required so those requests use the MinIO/R2 origin, not an opaque
  // sandbox origin that breaks module imports and fetch.
  return (
    <iframe
      src={src}
      className="w-full h-[800px] rounded-lg border border-ink-700"
      title="UX Analysis Viewer"
      sandbox="allow-scripts allow-same-origin"
    />
  )
}
