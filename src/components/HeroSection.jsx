import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const loopPhases = [
  'Perception',
  'Knowledge Grounding',
  'Execution',
  'Validation'
];

function HeroSection({ onExploreClick }) {
  const [modelLoaded, setModelLoaded] = useState(true);
  const [activePhaseIndex, setActivePhaseIndex] = useState(0);

  useEffect(() => {
    if (!modelLoaded) return;

    const timer = setInterval(() => {
      setActivePhaseIndex((idx) => (idx + 1) % loopPhases.length);
    }, 2200);

    return () => clearInterval(timer);
  }, [modelLoaded]);

  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Background gradient instead of Spline (placeholder) */}
      <div className="absolute inset-0">
        <div className="absolute inset-0 bg-gradient-to-br from-black via-neutral-950 to-black" />
        {/* Animated background orbs */}
        <motion.div
          className="absolute top-1/4 left-1/4 w-96 h-96 rounded-full bg-auraCyan/5 blur-3xl"
          animate={{ x: [0, 40, 0], y: [0, -30, 0], scale: [1, 1.15, 1] }}
          transition={{ duration: 12, repeat: Infinity, ease: 'easeInOut' }}
        />
        <motion.div
          className="absolute bottom-1/3 right-1/4 w-72 h-72 rounded-full bg-auraCyan/3 blur-3xl"
          animate={{ x: [0, -30, 0], y: [0, 40, 0], scale: [1.1, 1, 1.1] }}
          transition={{ duration: 15, repeat: Infinity, ease: 'easeInOut' }}
        />
        {/* Overlay gradient tint */}
        <div className="absolute inset-0 bg-gradient-to-b from-black/70 via-auraBlack/40 to-black/90 pointer-events-none" />
      </div>

      {/* Content overlay */}
      <div className="relative z-10 max-w-6xl mx-auto px-6 md:px-10 lg:px-16 flex flex-col lg:flex-row items-center gap-12">
        {/* Left content */}
        <AnimatePresence>
          {modelLoaded && (
            <motion.div
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 40 }}
              transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
              className="flex-1 space-y-6"
            >
              <motion.h1
                className="font-display text-3xl md:text-4xl lg:text-5xl tracking-[0.18em] uppercase text-auraCyan drop-shadow-glow"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2, duration: 0.8 }}
              >
                AURA
              </motion.h1>

              <motion.p
                className="font-mono text-xs md:text-sm tracking-[0.28em] text-neutral-400 uppercase"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.38, duration: 0.8 }}
              >
                Autonomous Computational Oncology Research Framework
              </motion.p>

              <motion.p
                className="text-sm md:text-base text-neutral-200 max-w-xl"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.55, duration: 0.8 }}
              >
                AURA orchestrates an end-to-end cancer research loop – from gigapixel whole slide
                ingestion to hazard ratio estimation – as a fully autonomous, multi-agent system
                built for digital biology labs.
              </motion.p>

              <div className="flex flex-wrap gap-4 items-center pt-4">
                <motion.button
                  onClick={onExploreClick}
                  className="glass px-5 py-2 rounded-full font-mono text-xs tracking-[0.26em] uppercase text-auraCyan hover:text-black hover:bg-auraCyan transition-colors shadow-glow"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.7, duration: 0.7 }}
                >
                  Visualize Research Cycle
                </motion.button>

                <motion.div
                  className="flex items-center gap-2 text-[10px] text-neutral-400 font-mono"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.78, duration: 0.7 }}
                >
                  <span className="inline-flex h-2 w-2 rounded-full bg-auraCyan animate-pulse shadow-glow" />
                  LIVE SANDBOX · MOCK VISUALIZATION
                </motion.div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Right: circular loop indicator */}
        <motion.div
          className="flex-1 flex items-center justify-center"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={modelLoaded ? { opacity: 1, scale: 1 } : { opacity: 0, scale: 0.9 }}
          transition={{ duration: 0.9 }}
        >
          <div className="relative w-56 h-56 md:w-64 md:h-64">
            {/* Outer rotating ring */}
            <motion.div
              className="absolute inset-0 rounded-full border border-auraCyan/50"
              animate={{ rotate: 360 }}
              transition={{ repeat: Infinity, duration: 36, ease: 'linear' }}
            >
              <div className="absolute -top-1 left-1/2 h-2 w-2 -translate-x-1/2 rounded-full bg-auraCyan shadow-glow" />
            </motion.div>

            {/* Phase nodes */}
            {loopPhases.map((phase, idx) => {
              const angle = (idx / loopPhases.length) * Math.PI * 2 - Math.PI / 2;
              const radius = 96;
              const x = Math.cos(angle) * radius;
              const y = Math.sin(angle) * radius;
              const isActive = idx === activePhaseIndex;

              return (
                <motion.button
                  key={phase}
                  type="button"
                  className="absolute font-mono text-[10px] text-center uppercase"
                  style={{
                    left: `calc(50% + ${x}px)`,
                    top: `calc(50% + ${y}px)`,
                    transform: 'translate(-50%, -50%)'
                  }}
                  whileHover={{ scale: 1.08 }}
                >
                  <span
                    className={`inline-flex items-center justify-center px-2 py-1 rounded-full glass ${
                      isActive ? 'text-auraCyan shadow-glow' : 'text-neutral-300/80'
                    }`}
                  >
                    {phase}
                  </span>
                </motion.button>
              );
            })}

            {/* Center label */}
            <div className="absolute inset-10 rounded-full glass flex flex-col items-center justify-center">
              <span className="font-mono text-[10px] text-neutral-500 tracking-[0.26em] uppercase">
                Autonomous Loop
              </span>
              <span className="mt-1 font-display text-sm text-auraCyan">
                WSI → Prognosis
              </span>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}

export default HeroSection;
