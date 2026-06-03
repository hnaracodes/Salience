/**
 * Compact Three.js cortical surface viewer for UX session sidebar.
 * Loads brain/manifest.json + coords.bin + faces.bin + preds.bin from viewer base URL.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

function hotColor(t) {
  // yellow -> orange -> red (matches export warm heatmap)
  const r = 0.86 + 0.14 * t;
  const g = 0.16 + 0.7 * (1 - t) * (1 - t);
  const b = 0.12 + 0.2 * (1 - t);
  return new THREE.Color(r, g, b);
}

async function loadBinary(url, dtype) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load ${url}: ${res.status}`);
  const buf = await res.arrayBuffer();
  if (dtype === "float32") return new Float32Array(buf);
  if (dtype === "int32") return new Int32Array(buf);
  throw new Error(`Unknown dtype ${dtype}`);
}

/**
 * @param {HTMLElement} host
 * @param {string} brainBase - URL prefix for brain assets (e.g. "./brain")
 */
export async function createBrainSurface(host, brainBase) {
  const base = brainBase.replace(/\/$/, "");
  const manifest = await (await fetch(`${base}/manifest.json`)).json();
  const { n_timesteps, n_vertices, vmin, vmax } = manifest;

  const coords = await loadBinary(`${base}/coords.bin`, "float32");
  const faces = await loadBinary(`${base}/faces.bin`, "int32");
  const predsFlat = await loadBinary(`${base}/preds.bin`, "float32");

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x121110);

  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 2000);
  camera.position.set(0, 0, 320);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  host.innerHTML = "";
  host.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.enablePan = false;

  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(n_vertices * 3);
  for (let i = 0; i < n_vertices; i++) {
    positions[i * 3] = coords[i * 3];
    positions[i * 3 + 1] = coords[i * 3 + 2];
    positions[i * 3 + 2] = coords[i * 3 + 1];
  }
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(Array.from(faces));
  geometry.computeVertexNormals();

  const colors = new Float32Array(n_vertices * 3);
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

  const mesh = new THREE.Mesh(
    geometry,
    new THREE.MeshPhongMaterial({ vertexColors: true, shininess: 18, side: THREE.DoubleSide }),
  );
  scene.add(mesh);
  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const dir = new THREE.DirectionalLight(0xffffff, 0.85);
  dir.position.set(200, 300, 400);
  scene.add(dir);

  geometry.computeBoundingBox();
  const center = new THREE.Vector3();
  geometry.boundingBox.getCenter(center);
  controls.target.copy(center);

  let currentT = 0;
  let peakMeta = { idx: 0, val: 0 };

  function setTimestep(t) {
    const idx = Math.max(0, Math.min(n_timesteps - 1, Number(t) || 0));
    currentT = idx;
    const row = predsFlat.subarray(idx * n_vertices, (idx + 1) * n_vertices);
    let maxVal = -Infinity;
    let maxIdx = 0;
    for (let v = 0; v < n_vertices; v++) {
      const val = row[v];
      if (val > maxVal) {
        maxVal = val;
        maxIdx = v;
      }
      const norm = (val - vmin) / (vmax - vmin + 1e-8);
      const c = hotColor(Math.min(1, Math.max(0, norm)));
      colors[v * 3] = c.r;
      colors[v * 3 + 1] = c.g;
      colors[v * 3 + 2] = c.b;
    }
    geometry.attributes.color.needsUpdate = true;
    peakMeta = { idx: maxIdx, val: maxVal };
  }

  function resize() {
    const w = Math.max(host.clientWidth, 1);
    const h = Math.max(host.clientHeight, 1);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
  }

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }

  resize();
  setTimestep(0);
  animate();

  const ro = typeof ResizeObserver !== "undefined"
    ? new ResizeObserver(resize)
    : null;
  if (ro) ro.observe(host);
  else window.addEventListener("resize", resize);

  return {
    setTimestep,
    getTimestep: () => currentT,
    getPeakMeta: () => ({ ...peakMeta }),
    getNTimesteps: () => n_timesteps,
    dispose() {
      if (ro) ro.disconnect();
      renderer.dispose();
      geometry.dispose();
      mesh.material.dispose();
    },
  };
}
