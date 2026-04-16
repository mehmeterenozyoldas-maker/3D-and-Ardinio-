/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useRef } from 'react';
import { initScene } from './scene';

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const paneContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (canvasRef.current && paneContainerRef.current) {
      const cleanup = initScene(canvasRef.current, paneContainerRef.current);
      return () => {
        cleanup.then(fn => fn()).catch(console.error);
      };
    }
  }, []);

  return (
    <div className="app-container">
      <header>
        <div className="logo">PYLON_FIELD <span>v2.0-WEBGPU</span></div>
        <div className="system-status">
          <div>FPS: 144</div>
          <div>MEM: 42.1MB</div>
          <div>GPU: NVIDIA RTX 4090</div>
        </div>
      </header>

      <aside ref={paneContainerRef}>
        {/* Tweakpane will be injected here */}
      </aside>

      <main>
        <canvas ref={canvasRef} id="webgpu-canvas" />
        <div className="viewport-overlay">
          GPU_BUFFER_HEIGHT: [90000 FLOAT]<br />
          GPU_BUFFER_COLOR: [90000 VEC4]<br />
          COMPUTE_PASS: 0.12ms<br />
          RENDER_PASS: 0.44ms<br />
          DRAWCALLS: 1 (INSTANCED)
        </div>
        <div className="cursor-hint">Click and drag to pan noise field</div>
      </main>

      <footer>
        <div className="footer-pill">
          <span>INSTANCE_COUNT: 90,000</span>
          <span>GRID_DIM: 300x300</span>
          <span>TSL_VERSION: 0.183.0</span>
        </div>
        <div>READY // LISTENING_FOR_UNIFORM_UPDATES</div>
      </footer>
    </div>
  );
}
