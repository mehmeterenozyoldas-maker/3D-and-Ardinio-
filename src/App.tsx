/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useRef, useState } from 'react';
import { initScene } from './scene';
import { X, Copy, Check } from 'lucide-react';

const HARDWARE_DOCS = {
  single: {
    title: "Single Motor",
    bom: [
      "1x Arduino Uno or Nano",
      "1x Vibration Motor (Coin or Cylindrical)",
      "1x NPN Transistor (2N2222) or Logic-level MOSFET (IRLZ44N)",
      "1x 1kΩ Resistor (to transistor base/gate)",
      "1x 1N4001 Diode (flyback protection across motor)"
    ],
    code: `const int motorPin = 3; // PWM pin

void setup() {
  Serial.begin(9600);
  pinMode(motorPin, OUTPUT);
}

void loop() {
  if (Serial.available()) {
    analogWrite(motorPin, Serial.read());
  }
}`
  },
  quadrant: {
    title: "4-Zone Quadrant",
    bom: [
      "1x Arduino Uno or Nano",
      "4x Vibration Motors",
      "4x Transistors or MOSFETs",
      "4x 1kΩ Resistors",
      "4x 1N4001 Diodes"
    ],
    code: `// Ensure these are PWM capable pins
const int pins[] = {3, 5, 6, 9};
byte buf[4];

void setup() {
  Serial.begin(9600);
  for(int i = 0; i < 4; i++) pinMode(pins[i], OUTPUT);
}

void loop() {
  if (Serial.available() >= 5) {
    if (Serial.read() == 255) { // Sync byte
      Serial.readBytes(buf, 4);
      for(int i = 0; i < 4; i++) {
        analogWrite(pins[i], buf[i]);
      }
    }
  }
}`
  },
  grid9: {
    title: "9-Zone Grid (3x3)",
    bom: [
      "1x Arduino Mega 2560 (Requires at least 9 PWM pins)",
      "9x Vibration Motors",
      "9x Transistors or MOSFETs",
      "9x 1kΩ Resistors",
      "9x 1N4001 Diodes"
    ],
     code: `// Arduino Mega PWM pins
const int pins[] = {2, 3, 4, 5, 6, 7, 8, 9, 10}; 
byte buf[9];

void setup() {
  Serial.begin(9600);
  for(int i = 0; i < 9; i++) pinMode(pins[i], OUTPUT);
}

void loop() {
  if (Serial.available() >= 10) {
    if (Serial.read() == 255) { // Sync byte
      Serial.readBytes(buf, 9);
      for(int i = 0; i < 9; i++) {
        analogWrite(pins[i], buf[i]);
      }
    }
  }
}`
  }
};

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const paneContainerRef = useRef<HTMLDivElement>(null);
  const [showModal, setShowModal] = useState(false);
  const [activeTab, setActiveTab] = useState<'single' | 'quadrant' | 'grid9'>('single');
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const handler = () => setShowModal(true);
    window.addEventListener('open-hardware-modal', handler);
    return () => window.removeEventListener('open-hardware-modal', handler);
  }, []);

  const handleCopy = () => {
    navigator.clipboard.writeText(HARDWARE_DOCS[activeTab].code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

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
        <div id="motor-status-list" className="absolute top-5 left-5 bg-[#1c1f2b]/80 backdrop-blur border border-[rgba(255,255,255,0.1)] p-3 rounded font-mono text-[11px] text-[#e0e2eb] min-w-[150px] shadow-lg transition-all hidden z-10 pointer-events-none"></div>
        <div className="viewport-overlay">
          GPU_BUFFER_HEIGHT: [90000 FLOAT]<br />
          GPU_BUFFER_COLOR: [90000 VEC4]<br />
          COMPUTE_PASS: 0.12ms<br />
          RENDER_PASS: 0.44ms<br />
          DRAWCALLS: 1 (INSTANCED)
        </div>

        <div className="cursor-hint">Click and drag to pan noise field</div>
      </main>

      {showModal && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur z-50 flex items-center justify-center p-6">
          <div className="bg-[#12141d] border border-[rgba(255,255,255,0.1)] rounded-xl w-full max-w-4xl max-h-[90vh] flex flex-col shadow-2xl">
            <div className="p-4 border-b border-[rgba(255,255,255,0.1)] flex items-center justify-between">
              <h2 className="font-mono text-[var(--accent)] text-lg tracking-wider font-bold">Hardware Integration Setup</h2>
              <button className="text-white/50 hover:text-white transition-colors" onClick={() => setShowModal(false)}>
                <X size={20} />
              </button>
            </div>
            
            <div className="flex border-b border-[rgba(255,255,255,0.1)] overflow-x-auto">
              {(Object.keys(HARDWARE_DOCS) as Array<keyof typeof HARDWARE_DOCS>).map(key => (
                <button 
                  key={key}
                  onClick={() => setActiveTab(key)}
                  className={`px-6 py-3 font-mono text-sm border-b-2 transition-colors whitespace-nowrap ${activeTab === key ? 'border-[var(--accent)] text-[var(--accent)] bg-[rgba(94,109,247,0.1)]' : 'border-transparent text-[#8e9299] hover:text-white'}`}
                >
                  {HARDWARE_DOCS[key].title}
                </button>
              ))}
            </div>

            <div className="flex flex-col md:flex-row flex-1 min-h-0">
              <div className="w-full md:w-1/3 p-6 border-r border-[rgba(255,255,255,0.1)] bg-[#07080e]/50 overflow-y-auto">
                <h3 className="font-mono text-white mb-4 text-sm tracking-widest uppercase">Bill of Materials</h3>
                <ul className="space-y-3 font-sans text-sm text-[#8e9299]">
                  {HARDWARE_DOCS[activeTab].bom.map((item, idx) => (
                     <li key={idx} className="flex gap-2">
                       <span className="text-[var(--accent)]">•</span>
                       <span>{item}</span>
                     </li>
                  ))}
                </ul>
              </div>
              <div className="w-full md:w-2/3 flex flex-col min-h-0 bg-[#07080e]">
                <div className="p-3 border-b border-[rgba(255,255,255,0.1)] flex justify-between items-center bg-[#1c1f2b]">
                  <span className="font-mono text-xs text-[#8e9299]">arduino_firmware.ino</span>
                  <button 
                    onClick={handleCopy} 
                    className="flex items-center gap-2 px-3 py-1 bg-[#12141d] hover:bg-[var(--accent)] border border-[rgba(255,255,255,0.1)] rounded text-xs font-mono transition-all text-white"
                  >
                    {copied ? <Check size={14} /> : <Copy size={14} />}
                    {copied ? 'COPIED!' : 'COPY CODE'}
                  </button>
                </div>
                <div className="flex-1 overflow-auto p-4">
                  <pre className="font-mono text-xs text-[#e0e2eb] leading-relaxed">
                    <code>{HARDWARE_DOCS[activeTab].code}</code>
                  </pre>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

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
