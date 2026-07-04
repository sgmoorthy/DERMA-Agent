import { useRef } from "react";
import HeroSection from "./components/HeroSection";
import ResearchCycleSection from "./components/ResearchCycleSection";
import BentoGridSection from "./components/BentoGridSection";
import FurtherReadingSection from "./components/FurtherReadingSection";

function App() {
  const researchRef = useRef(null);

  const scrollToResearch = () => {
    if (researchRef.current) {
      researchRef.current.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }
  };

  return (
    <div className="min-h-screen bg-auraBlack text-white">
      {/* Fixed navigation dock (bottom) */}
      <div className="fixed bottom-4 left-1/2 z-40 -translate-x-1/2 glass rounded-full px-6 py-3 flex items-center gap-4 shadow-glow">
        <button
          onClick={scrollToResearch}
          className="font-mono text-xs tracking-[0.26em] uppercase text-auraCyan hover:text-white transition-colors"
        >
          Research Cycle
        </button>
        <div className="h-1 w-16 bg-neutral-800 rounded-full overflow-hidden">
          <div className="h-full w-8 bg-auraCyan animate-pulse" />
        </div>
        <span className="font-mono text-[10px] text-neutral-400">
          AURA · Autonomous Oncology Loop
        </span>
      </div>

      <main className="relative">
        <HeroSection onExploreClick={scrollToResearch} />
        <div ref={researchRef}>
          <ResearchCycleSection />
        </div>
        <BentoGridSection />
        <FurtherReadingSection />
      </main>
    </div>
  );
}

export default App;
