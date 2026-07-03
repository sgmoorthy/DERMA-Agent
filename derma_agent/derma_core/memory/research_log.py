import os
import json
from datetime import datetime

# Handle crewai-soul imports gracefully with safe fallback
try:
    from crewai_soul import SoulMemory
    CREWAI_SOUL_AVAILABLE = True
except ImportError:
    CREWAI_SOUL_AVAILABLE = False

class ResearchNarrative:
    def __init__(self, agent_name: str, fallback_path: str = "data/research_narrative_log.json"):
        self.agent_name = agent_name
        self.fallback_path = fallback_path
        self.local_thoughts = []
        
        if CREWAI_SOUL_AVAILABLE:
            try:
                self.memory = SoulMemory(agent_name=agent_name)
            except Exception:
                # If initialization fails, use fallback
                self.memory = None
        else:
            self.memory = None
            
        # Ensure fallback storage path exists
        if not self.memory:
            os.makedirs(os.path.dirname(self.fallback_path), exist_ok=True)
            self._load_local_thoughts()
            
    def _load_local_thoughts(self):
        if os.path.exists(self.fallback_path):
            try:
                with open(self.fallback_path, 'r') as f:
                    self.local_thoughts = json.load(f)
            except Exception:
                self.local_thoughts = []
                
    def _save_local_thoughts(self):
        try:
            with open(self.fallback_path, 'w') as f:
                json.dump(self.local_thoughts, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving narrative fallback log: {e}")

    def log_discovery(self, hypothesis: str, validation_result: str):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "hypothesis": hypothesis,
            "result": validation_result,
            "log": f"Hypothesis: {hypothesis} | Result: {validation_result}"
        }
        
        if self.memory:
            try:
                self.memory.save_thought(log_entry["log"])
                return
            except Exception:
                pass
                
        # Fallback to local memory & persistent file
        self.local_thoughts.append(log_entry)
        self._save_local_thoughts()
        
    def get_history(self) -> list:
        if self.memory:
            try:
                return self.memory.get_all_thoughts()
            except Exception:
                pass
        return [item["log"] for item in self.local_thoughts]
