from langchain_openai import ChatOpenAI
import spacy, os, json


class AtomicFactGenerator(object):
    def __init__(self, llm: ChatOpenAI, demon_dir):
        self.nlp = spacy.load("en_core_web_sm")
        self.few_shot_path = os.path.join(demon_dir, "demons.json")
        self.llm = llm

        with open(self.few_shot_path, 'r') as f:
            self.demons = json.load(f)

    def get_init_atomic_facts_from_sentence(self, sentences):
        atomic_facts = {}
        for sent in sentences:
            prompt = f""" """

            response = self.llm.invoke(prompt)

            facts = [fact.strip() for fact in response.content.split(",") if fact.strip()]
            atomic_facts[sent] = facts

        return atomic_facts
