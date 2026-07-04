import { motion } from 'framer-motion';

const tiles = [
  {
    id: 'wsi',
    label: 'Gigapixel Tissue Sample',
    description: 'Mock interactive tile representing a zoomable pathology whole slide.'
  },
  {
    id: 'km',
    label: 'Kaplan–Meier Survival Curve',
    description: 'Mock survival curve tile with multi-arm visualization.'
  },
  {
    id: 'genes',
    label: 'Gene–Pathway Correlation',
    description: 'Mock network tile representing pathway-centric feature correlations.'
  },
  {
    id: 'console',
    label: 'Discovery Console',
    description:
      'Retro-futuristic sandbox console where autonomous research instances are visually "spawned".'
  }
];

function BentoGridSection() {
  return (
    <section className="py-20 md:py-28 bg-auraBlack">
      <div className="max-w-6xl mx-auto px-6 md:px-10 lg:px-16">
        <motion.h2
          className="font-display text-2xl md:text-3xl tracking-[0.24em] uppercase text-auraCyan mb-6"
          initial={{ opacity: 0, y: 18 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.4 }}
          transition={{ duration: 0.7 }}
        >
          Mock Analytics &amp; Sandbox
        </motion.h2>

        <motion.p
          className="text-sm md:text-base text-neutral-300 max-w-2xl mb-10"
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.4 }}
          transition={{ duration: 0.7, delay: 0.16 }}
        >
          This grid renders AURA&apos;s analytical capabilities entirely as front-end mock visualization:
          synthetic cohorts, illustrative survival curves, and conceptual gene–pathway maps driven
          by static JSON structures.
        </motion.p>

        <div className="grid md:grid-cols-4 gap-4 md:gap-6 auto-rows-[minmax(140px,1fr)]">
          {tiles.map((tile, idx) => (
            <motion.div
              key={tile.id}
              className={`glass rounded-xl p-4 md:p-5 border border-auraCyan/20 cursor-pointer hover:border-auraCyan/70 hover:shadow-glow transition-all ${
                tile.id === 'wsi' ? 'md:col-span-2 md:row-span-2' : ''
              } ${tile.id === 'km' ? 'md:col-span-2' : ''}`}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.25 }}
              transition={{ duration: 0.6, delay: idx * 0.08 }}
              whileHover={{ y: -4, scale: 1.02 }}
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="font-mono text-[10px] tracking-[0.26em] uppercase text-neutral-500">
                    {tile.id.toUpperCase()}
                  </p>
                  <h3 className="mt-2 font-display text-sm md:text-base text-auraCyan">
                    {tile.label}
                  </h3>
                  <p className="mt-3 text-[11px] md:text-xs text-neutral-200">
                    {tile.description}
                  </p>
                </div>
                <div className="flex flex-col items-end">
                  <span className="inline-flex h-1.5 w-8 bg-neutral-800 rounded-full overflow-hidden mb-1">
                    <span className="h-full w-4 bg-auraCyan/70" />
                  </span>
                  <span className="font-mono text-[9px] text-neutral-500">
                    MOCK DATA
                  </span>
                </div>
              </div>

              {/* Placeholder inner canvas */}
              <div className="mt-4 h-24 md:h-28 rounded-lg bg-gradient-to-br from-neutral-900 via-black to-neutral-800 border border-neutral-800/50 flex items-center justify-center">
                <span className="font-mono text-[10px] text-neutral-500">
                  Visualization placeholder · driven by static JSON
                </span>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default BentoGridSection;
