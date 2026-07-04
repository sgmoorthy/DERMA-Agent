import { motion } from "framer-motion";

const articles = [
  {
    title: "Documentation hub",
    href: "/docs/index.html",
    summary:
      "A single landing page for the walkthrough, deployment guide, blog index, and core repository docs.",
  },
  {
    title: "Closed-loop agentic discovery",
    href: "/blog/closed-loop-agentic-discovery.html",
    summary:
      "Why DermaMind.ai should be viewed as a scientific workflow engine instead of a single predictive model.",
  },
  {
    title: "Architecture deep dive",
    href: "/blog/architecture-deep-dive.html",
    summary:
      "A layer-by-layer walkthrough of perception, graph grounding, execution, validation, and presentation.",
  },
  {
    title: "Research math and safety",
    href: "/blog/research-math-and-safety.html",
    summary:
      "How BH/FDR correction, attention-style slide pooling, and sandboxed execution strengthen the framework.",
  },
  {
    title: "Project walkthrough",
    href: "/walkthrough.md",
    summary:
      "A concise artifact summarizing the recent research-grade changes, validation steps, and follow-up ideas.",
  },
];

function FurtherReadingSection() {
  return (
    <section className="py-20 md:py-28 bg-gradient-to-b from-black via-auraBlack to-black">
      <div className="max-w-6xl mx-auto px-6 md:px-10 lg:px-16">
        <motion.h2
          className="font-display text-2xl md:text-3xl tracking-[0.24em] uppercase text-auraCyan mb-6"
          initial={{ opacity: 0, y: 18 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.4 }}
          transition={{ duration: 0.7 }}
        >
          Further Reading
        </motion.h2>

        <motion.p
          className="text-sm md:text-base text-neutral-300 max-w-3xl mb-10"
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.4 }}
          transition={{ duration: 0.7, delay: 0.12 }}
        >
          Explore the framework from multiple angles: the closed-loop discovery
          philosophy, the layered architecture, and the research math and safety
          principles that shape how DermaMind.ai makes and reports scientific
          claims.
        </motion.p>

        <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-6">
          {articles.map((article, idx) => (
            <motion.a
              key={article.title}
              href={article.href}
              className="glass rounded-2xl p-5 md:p-6 border border-auraCyan/20 hover:border-auraCyan/60 transition-colors shadow-sm hover:shadow-glow block"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.25 }}
              transition={{ duration: 0.6, delay: idx * 0.08 }}
              whileHover={{ y: -4, scale: 1.01 }}
            >
              <p className="font-mono text-[10px] tracking-[0.24em] uppercase text-neutral-500">
                Documentation
              </p>
              <h3 className="mt-3 font-display text-base md:text-lg text-auraCyan">
                {article.title}
              </h3>
              <p className="mt-3 text-xs md:text-sm text-neutral-200">
                {article.summary}
              </p>
              <span className="mt-5 inline-block font-mono text-[10px] uppercase tracking-[0.22em] text-auraCyan">
                Open article →
              </span>
            </motion.a>
          ))}
        </div>
      </div>
    </section>
  );
}

export default FurtherReadingSection;
