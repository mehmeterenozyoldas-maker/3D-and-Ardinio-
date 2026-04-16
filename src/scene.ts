import * as THREE from "three/webgpu";
import {
  Fn,
  uniform,
  storage,
  instanceIndex,
  vec2,
  vec3,
  vec4,
  cos,
  dot,
  step,
  min,
  max,
  abs,
  float,
  fract,
  floor,
  positionLocal,
  select,
  pass,
  mrt,
  output,
  normalView,
  Loop,
  exp,
  log,
  negate
} from "three/tsl";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { ao } from "three/examples/jsm/tsl/display/GTAONode.js";
import { bloom } from "three/examples/jsm/tsl/display/BloomNode.js";
import { Pane } from "tweakpane";

THREE.Node.captureStackTrace = true;

/*
 *** Simplex noise function - ported from McEwan / Gustavson (Ashima Arts)
*/
const mod289 = (x: any): any =>
  x.sub(floor(x.mul(1.0 / 289.0)).mul(289.0));
const permute = (x: any): any =>
  mod289(x.mul(34.0).add(10.0).mul(x));
const taylorInvSqrt = (r: any): any =>
  float(1.79284291400159).sub(r.mul(0.85373472095314));

export const simplexNoise = Fn(([v]: [any]) => {
  const C = vec2(1.0 / 6.0, 1.0 / 3.0);
  const D = vec4(0.0, 0.5, 1.0, 2.0);

  const i  = floor(vec3(v).add(dot(vec3(v), vec3(C.y, C.y, C.y))));
  const x0 = vec3(v).sub(i).add(dot(i, vec3(C.x, C.x, C.x)));

  const g  = step(vec3(x0.y, x0.z, x0.x), x0);
  const l  = vec3(1.0).sub(g);
  const i1 = min(g, vec3(l.z, l.x, l.y));
  const i2 = max(g, vec3(l.z, l.x, l.y));

  const x1 = x0.sub(i1).add(C.x);
  const x2 = x0.sub(i2).add(C.y);
  const x3 = x0.sub(0.5);

  const im = mod289(i);
  const p  = permute(
    permute(
      permute(vec4(im.z, im.z, im.z, im.z).add(vec4(0.0, i1.z, i2.z, 1.0)))
        .add(vec4(im.y, im.y, im.y, im.y))
        .add(vec4(0.0, i1.y, i2.y, 1.0)),
    )
      .add(vec4(im.x, im.x, im.x, im.x))
      .add(vec4(0.0, i1.x, i2.x, 1.0)),
  );

  const n_  = float(1.0 / 7.0);
  const ns  = vec3(n_.mul(D.w), n_.mul(D.y).sub(1.0), n_.mul(D.z));

  const j   = p.sub(float(49.0).mul(floor(p.mul(ns.z).mul(ns.z))));
  const x_  = floor(j.mul(ns.z));
  const y_  = floor(j.sub(float(7.0).mul(x_)));
  const gx  = x_.mul(ns.x).add(ns.y);
  const gy  = y_.mul(ns.x).add(ns.y);
  const h   = float(1.0).sub(abs(gx)).sub(abs(gy));

  const b0  = vec4(gx.x, gx.y, gy.x, gy.y);
  const b1  = vec4(gx.z, gx.w, gy.z, gy.w);
  const s0  = floor(b0).mul(2.0).add(1.0);
  const s1  = floor(b1).mul(2.0).add(1.0);
  const sh  = step(h, vec4(0.0)).negate();

  const a0  = vec4(b0.x, b0.z, b0.y, b0.w).add(
    vec4(s0.x, s0.z, s0.y, s0.w).mul(vec4(sh.x, sh.x, sh.y, sh.y)),
  );
  const a1  = vec4(b1.x, b1.z, b1.y, b1.w).add(
    vec4(s1.x, s1.z, s1.y, s1.w).mul(vec4(sh.z, sh.z, sh.w, sh.w)),
  );

  const p0 = vec3(a0.x, a0.y, h.x);
  const p1 = vec3(a0.z, a0.w, h.y);
  const p2 = vec3(a1.x, a1.y, h.z);
  const p3 = vec3(a1.z, a1.w, h.w);

  const norm = taylorInvSqrt(
    vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)),
  );
  const p0n = p0.mul(norm.x);
  const p1n = p1.mul(norm.y);
  const p2n = p2.mul(norm.z);
  const p3n = p3.mul(norm.w);

  const m  = max(
    float(0.6).sub(vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3))),
    0.0,
  );
  const m2 = m.mul(m);

  return float(42.0)
    .mul(dot(m2.mul(m2), vec4(dot(p0n, x0), dot(p1n, x1), dot(p2n, x2), dot(p3n, x3))))
    .mul(0.5)
    .add(0.5);
});

interface FBMOptions {
  octaves?: number;
  lacunarity?: number;
  gain?: number;
}

export const makeFBM = (
  noiseFn: (p: any) => any,
  { octaves = 6, lacunarity = 2.0, gain = 0.5 }: FBMOptions = {},
) =>
  Fn(([p]: [any]) => {
    const value = float(0).toVar();
    const amplitude = float(0.5).toVar();
    const pos = vec3(p).toVar();

    Loop(octaves, () => {
      value.addAssign(noiseFn(pos).mul(amplitude));
      pos.mulAssign(lacunarity);
      amplitude.mulAssign(gain);
    });

    return value;
  });

const palette = Fn(
  ([t, a, b, c, d]: [any, any?, any?, any?, any?]) => {
    const _a = a ?? vec3(0.5, 0.5, 0.5);
    const _b = b ?? vec3(0.5, 0.5, 0.5);
    const _c = c ?? vec3(1.0, 1.0, 1.0);
    const _d = d ?? vec3(0.0, 0.33, 0.67);
    return _a.add(_b.mul(cos(_c.mul(t).add(_d).mul(6.28318))));
  },
);

const fbmNoise = makeFBM(simplexNoise);

export async function initScene(canvas: HTMLCanvasElement, paneContainer: HTMLElement) {
  if (!navigator.gpu) {
    throw new Error("WebGPU not supported");
  }

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x07080e);
  scene.fog = new THREE.Fog(0x07080e, 10, 60);

  const camera = new THREE.PerspectiveCamera(
    75,
    canvas.clientWidth / canvas.clientHeight,
    0.1,
    300,
  );
  camera.position.set(-4, 11, 15);
  camera.lookAt(0, -10, 0);

  const renderer = new THREE.WebGPURenderer({
    canvas,
    antialias: true,
  });
  renderer.setPixelRatio(window.devicePixelRatio || 2);
  renderer.setSize(canvas.clientWidth, canvas.clientHeight);
  const isAndroid = /android/i.test(navigator.userAgent);
  renderer.shadowMap.enabled = !isAndroid;
  renderer.shadowMap.type = THREE.PCFShadowMap;

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.enablePan = true;
  controls.enableZoom = true;
  controls.enabled = true;
  controls.enableRotate = false;
  controls.target.set(0, 2, 0);

  const options = {
    fieldRadius: 9,
    pylonRadius: 0.31,
    maxHeight: 8,
    noiseScale: 0.28,
    timeSpeed: 0.22,
    circularPattern: true,
    fogDensity: 0.04,
    animating: false,
    cameraControls: false,
    noiseType: "simplex" as "simplex" | "fbm",
  };

  const fieldRadiusU = uniform(options.fieldRadius);
  const pylonRadiusU = uniform(options.pylonRadius);
  const maxHeightU = uniform(options.maxHeight);
  const noiseScaleU = uniform(options.noiseScale);
  const timeSpeedU = uniform(options.timeSpeed);
  const noiseOffsetXU = uniform(1);
  const noiseOffsetYU = uniform(0);
  const noiseOffsetZU = uniform(0);
  const circularPatternU = uniform(1);

  function computeGrid(): number {
    const MAX_GRID = Math.floor(Math.sqrt(200_000));
    const desired = Math.round(options.fieldRadius / options.pylonRadius) + 1;
    return Math.min(MAX_GRID, Math.max(3, desired));
  }

  let GRID = 0;
  let COUNT = 0;

  let heightBuffer!: THREE.StorageBufferAttribute;
  let colorBuffer!: THREE.StorageBufferAttribute;
  let heightStorage!: any;
  let colorStorage!: any;
  let computeNode: any = null;
  let pylonMesh: THREE.InstancedMesh | null = null;

  const cyl = new THREE.CylinderGeometry(1, 1, 1, 24, 1);

  function buildComputeNode(count: number) {
    return Fn(() => {
      const idxF = float(instanceIndex);
      const col = idxF.mod(float(GRID));
      const row = floor(idxF.div(float(GRID)));
      const nx = col
        .div(float(GRID - 1))
        .mul(2.0)
        .sub(1.0);
      const nz = row
        .div(float(GRID - 1))
        .mul(2.0)
        .sub(1.0);

      const inside = nx.mul(nx).add(nz.mul(nz)).lessThanEqual(float(1));
      const active = inside.or(circularPatternU.lessThan(float(0.5)));

      const noiseIn = vec3(
        nx.mul(fieldRadiusU).mul(noiseScaleU).add(noiseOffsetXU),
        noiseOffsetYU,
        nz.mul(fieldRadiusU).mul(noiseScaleU).mul(0.65).add(noiseOffsetZU),
      ).toVar();
      const noiseFn = options.noiseType === "fbm" ? fbmNoise : simplexNoise;
      const n = noiseFn(noiseIn.mul(0.5)).pow(2);

      heightStorage
        .element(instanceIndex)
        .assign(select(active, n.mul(maxHeightU).add(0.04), float(0)));

      const t = fract(n.mul(0.08).mul(maxHeightU).add(0.5));
      const rgb = palette(
        t,
        vec3(0.5, 0.55, 0.5),
        vec3(0.5),
        vec3(0.5, 0.4, 0.3),
        vec3(0.2),
      );
      colorStorage
        .element(instanceIndex)
        .assign(vec4(select(active, rgb, vec3(0, 0, 0)), float(1)));
    })().compute(count);
  }

  function buildMaterial(): THREE.MeshStandardMaterial {
    const mat = new THREE.MeshStandardMaterial({
      roughness: 0.9,
      metalness: 0.9,
    });

    mat.positionNode = Fn(() => {
      const h = heightStorage.element(instanceIndex);
      const pos = positionLocal.toVar();
      pos.x.assign(pos.x.mul(pylonRadiusU));
      pos.z.assign(pos.z.mul(pylonRadiusU));
      pos.y.assign(pos.y.add(0.5).mul(h));
      return pos;
    })();

    mat.colorNode = Fn(
      () => colorStorage.element(instanceIndex),
    )();
    
    mat.roughnessNode = Fn(() => colorStorage.element(instanceIndex).b.add(.3))();
    mat.metalnessNode = Fn(() => colorStorage.element(instanceIndex).b.add(.3))();

    return mat;
  }

  function rebuild() {
    const newGRID = computeGrid();

    if (newGRID === GRID && pylonMesh) {
      updateInstanceMatrices();
      return;
    }

    GRID = newGRID;
    COUNT = GRID * GRID;

    heightBuffer = new THREE.StorageBufferAttribute(COUNT, 1);
    colorBuffer = new THREE.StorageBufferAttribute(COUNT, 4);
    heightStorage = storage(heightBuffer, "float", COUNT);
    colorStorage = storage(colorBuffer, "vec4", COUNT);

    computeNode = buildComputeNode(COUNT);

    if (pylonMesh) {
      scene.remove(pylonMesh);
      (pylonMesh.material as THREE.Material).dispose();
    }
    pylonMesh = new THREE.InstancedMesh(cyl, buildMaterial(), COUNT);
    pylonMesh.castShadow = true;
    pylonMesh.receiveShadow = true;
    pylonMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    scene.add(pylonMesh);

    updateInstanceMatrices();
  }

  function updateInstanceMatrices() {
    if (!pylonMesh) return;
    const mat = new THREE.Matrix4();
    for (let i = 0; i < COUNT; i++) {
      const col = i % GRID;
      const row = Math.floor(i / GRID);
      const nx = (col / (GRID - 1)) * 2 - 1;
      const nz = (row / (GRID - 1)) * 2 - 1;
      const dist = Math.sqrt(nx * nx + nz * nz);
      const x = nx * options.fieldRadius;
      const z = nz * options.fieldRadius;
      mat.setPosition(x, options.circularPattern && dist > 1 ? -500 : 0, z);
      pylonMesh.setMatrixAt(i, mat);
    }
    pylonMesh.instanceMatrix.needsUpdate = true;
  }

  rebuild();

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(400, 400),
    new THREE.MeshStandardMaterial({
      color: 0x0d0f18,
      roughness: 0.95,
      metalness: 0.05,
    }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -0.005;
  ground.receiveShadow = true;
  scene.add(ground);

  const sun = new THREE.DirectionalLight(0xffffff, 9);
  sun.castShadow = true;
  sun.position.set(30, 30, 42);
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.near = 0.05;
  sun.shadow.camera.far = 1000;
  sun.shadow.radius = 2;
  (sun.shadow.camera as THREE.OrthographicCamera).left = -50;
  (sun.shadow.camera as THREE.OrthographicCamera).right = 50;
  (sun.shadow.camera as THREE.OrthographicCamera).top = 50;
  (sun.shadow.camera as THREE.OrthographicCamera).bottom = -50;
  sun.shadow.camera.updateProjectionMatrix();
  const backlight = new THREE.DirectionalLight(0x998877, 6.5);
  backlight.position.set(-10, 8, -12);
  scene.add(backlight);
  scene.add(sun);
  scene.add(new THREE.AmbientLight(0x223355, 3));

  const scenePass = pass(scene, camera);
  scenePass.setMRT(mrt({ output, normal: normalView }));
  const sceneColor = scenePass.getTextureNode("output");
  const sceneNormal = scenePass.getTextureNode("normal");
  const sceneDepth = scenePass.getTextureNode("depth");

  const aoEffect = ao(sceneDepth, sceneNormal, camera);
  aoEffect.resolutionScale = 0.5;
  aoEffect.radius.value = 0.5;

  const aoColor = aoEffect.getTextureNode().r.mul(sceneColor);
  const bloomEffect = bloom(aoColor, .9, .5, 0.7);

  const postProcessing = new THREE.RenderPipeline(
    renderer,
    sceneColor.add(bloomEffect),
  );

  const pane = new Pane({ container: paneContainer });
  const f = pane.addFolder({ title: "Pylons" });

  f.addBinding(options, "fieldRadius", {
    label: "Field Radius",
    min: 2,
    max: 100,
    step: 0.5,
  }).on("change", (v: any) => {
    fieldRadiusU.value = v.value;
    rebuild();
  });

  f.addBinding(options, "circularPattern", { label: "Circular Pattern" }).on(
    "change",
    () => {
      circularPatternU.value = options.circularPattern ? 1 : 0;
      updateInstanceMatrices();
    },
  );

  f.addBinding(options, "pylonRadius", {
    label: "Pylon Radius",
    min: 0.05,
    max: 0.6,
    step: 0.01,
  }).on("change", (v: any) => {
    pylonRadiusU.value = v.value;
    rebuild();
  });

  f.addBinding(options, "maxHeight", {
    label: "Max Height",
    min: 0.5,
    max: 20,
    step: 0.1,
  }).on("change", (v: any) => {
    maxHeightU.value = v.value;
  });

  f.addBinding(options, "noiseScale", {
    label: "Noise Scale",
    min: 0.002,
    max: 1.0,
    step: 0.01,
  }).on("change", (v: any) => {
    noiseScaleU.value = v.value;
  });

  f.addBinding(options, "timeSpeed", {
    label: "Time Speed",
    min: 0,
    max: 2,
    step: 0.01,
  }).on("change", (v: any) => {
    timeSpeedU.value = v.value;
  });
  f.addBinding(options, "noiseType", {
    label: "Noise",
    options: { Simplex: "simplex", FBM: "fbm" },
  }).on("change", () => {
    computeNode = buildComputeNode(COUNT);
  });
  f.addBinding(options, "animating", { label: "Animating" });
  f.addBinding(options, "cameraControls", { label: "Camera Controls" }).on(
    "change",
    () => {
      controls.enableRotate = options.cameraControls;
    },
  );
  f.expanded = true;

  let isDragging = false;
  let dragX = 0,
    dragY = 0;
  let velX = 0,
    velY = 0;

  canvas.addEventListener("pointerdown", (e) => {
    isDragging = true;
    velX = velY = 0;
    dragX = e.clientX;
    dragY = e.clientY;
  });
  window.addEventListener("pointermove", (e) => {
    if (!isDragging) return;
    const dx = e.clientX - dragX;
    const dy = e.clientY - dragY;
    dragX = e.clientX;
    dragY = e.clientY;
    if (!options.cameraControls) {
      velX = dx * 0.01;
      velY = dy * 0.01;
      
      noiseOffsetXU.value -= velX;
      noiseOffsetZU.value -= velY;
    }
  });
  window.addEventListener("pointerup", () => {
    isDragging = false;
  });

  let serialWriter: any = null;

  const hardwareFolder = pane.addFolder({ title: "Hardware Integration" });
  hardwareFolder.addButton({ title: "Connect Arduino" }).on("click", async () => {
    try {
      const nav = navigator as any;
      if (!nav.serial) {
        alert("Web Serial API is not supported in this browser. Please use Chrome or Edge.");
        return;
      }
      const port = await nav.serial.requestPort();
      await port.open({ baudRate: 9600 });
      serialWriter = port.writable.getWriter();
      console.log("Arduino Connected!");
    } catch (e) {
      console.error("Serial connection failed:", e);
    }
  });

  let lastTime = performance.now();
  let lastSerialTime = performance.now();
  let animationId: number;

  async function animate() {
    const now = performance.now();
    const delta = (now - lastTime) / 1000;
    lastTime = now;
    if (options.animating) noiseOffsetYU.value += delta * options.timeSpeed;
    if (!isDragging && (velX !== 0 || velY !== 0)) {
      const decay = Math.pow(0.04, delta);
      velX *= decay;
      velY *= decay;
      noiseOffsetXU.value -= velX;
      noiseOffsetZU.value -= velY;
    }

    if (serialWriter && now - lastSerialTime > 50) {
      // Generate a vibration intensity (0.0 to 1.0) based on the noise field's movement
      const pulse = (Math.sin(noiseOffsetYU.value * 4.0) + Math.cos(noiseOffsetXU.value * 4.0)) * 0.25 + 0.5;
      const pwm = Math.max(0, Math.min(255, Math.floor(pulse * 255)));
      serialWriter.write(new Uint8Array([pwm])).catch((err: any) => console.error("Write error:", err));
      lastSerialTime = now;
    }

    controls.update();
    if (computeNode) await renderer.computeAsync(computeNode);
    postProcessing.render();
    animationId = requestAnimationFrame(animate);
  }

  animate();

  const handleResize = () => {
    camera.aspect = canvas.clientWidth / canvas.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(canvas.clientWidth, canvas.clientHeight);
  };
  window.addEventListener("resize", handleResize);

  return () => {
    cancelAnimationFrame(animationId);
    window.removeEventListener("resize", handleResize);
    pane.dispose();
    renderer.dispose();
  };
}
