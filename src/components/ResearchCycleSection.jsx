import { motion } from 'framer-motion';

const steps = [
  {
    title: 'Whole Slide Ingestion',
    subtitle: 'Perception',
    description:
      'AURA streams gigapixel pathology whole slide images from oncology data lakes, constructing a multi-scale view of tissue microenvironments.'
  },
  {
    title: 'Feature Grounding',
    subtitle: 'Knowledge Grounding',
    description:
      'Multi-agent perception layers lock onto prognostic morphologies, aligning them with genomic signatures and prior evidence graphs.'
  },
  {
    title: 'Autonomous Trials',
    subtitle: 'Execution',
    description:
      'Simulated treatment arms and survival scenarios are executed inside a sandbox, generating synthetic cohorts while respecting clinical constraints.'
  },
  {
    title: 'Survival Analytics',
    subtitle: 'Validation',
    description:
      'Kaplan–Meier curves, hazard ratios, and confidence intervals are computed as AURA converges on stable prognostic hypotheses.'
  }
];

function ResearchCycleSection() {
  return (
    <section className="relative py-24 md:py-32 bg-gradient-to-b from-black via-auraBlack to-black">
      <div className="max-w-6xl mx-auto px-6 md:px-10 lg:px-16">
        <motion.h2
          className="font-display text-2xl md:text-3xl tracking-[0.24em] uppercase text-auraCyan mb-6"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.4 }}
          transition={{ duration: 0.8 }}
        >
          The Autonomous Research Cycle
        </motion.h2>

        <motion.p
          className="text-sm md:text-base text-neutral-300 max-w-2xl mb-10"
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.4 }}
          transition={{ duration: 0.8, delay: 0.15 }}
        >
          Each iteration of AURA in silico corresponds to a closed-loop oncology experiment –
          perception of tissue states, grounding of signals, execution of simulated trials, and
          validation of survival hypotheses using mock, client-side data structures.
        </motion.p>

        <div className="grid md:grid-cols-2 gap-8">
          {steps.map((step, idx) => (
            <motion.div
              key={step.title}
              className="glass rounded-2xl p-5 md:p-6 border border-auraCyan/20 hover:border-auraCyan/60 transition-colors shadow-sm hover:shadow-glow"
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.25 }}
              transition={{ duration: 0.65, delay: idx * 0.12 }}
            >
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="font-mono text-[10px] tracking-[0.24em] uppercase text-neutral-500">
                    Step {idx + 1} · {step.subtitle}
                  </p>
                  <h3 className="mt-2 font-display text-base md:text-lg text-auraCyan">
                    {step.title}
                  </h3>
                  <p className="mt-3 text-xs md:text-sm text-neutral-200">
                    {step.description}
                  </p>
                </div>
                <div className="flex flex-col items-end text-[10px] font-mono text-neutral-500">
                  <span className="inline-flex h-1.5 w-10 bg-neutral-800 rounded-full overflow-hidden">
                    <span className="h-full w-4 bg-auraCyan/70" />
                  </span>
                  <span className="mt-1">Phase {idx + 1}</span>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default ResearchCycleSection;
