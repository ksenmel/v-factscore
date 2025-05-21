import re
from pysbd import Segmenter

from factscore.api_requests import APICompletions

SENTENCE_INSTRUCT_PROMPT = """Task: Given the following sentence, break it into individual, independent facts. Ensure that each statement does not rely on context from other statements. Replace all pronouns (e.g., 'he,' 'she,' 'it,' 'they') with the corresponding nouns or proper names to make the meaning clear without additional context. Do not change anything in the citations. If the sentence is inadequate or doesn't contain any information, answer "No facts to extract".

Example 1:
Input Sentence: "Albert Einstein developed the theory of relativity, which revolutionized modern physics."
Output:
- Albert Einstein developed the theory of relativity
- The theory of relativity revolutionized modern physics

Example 2:
Input Sentence: "Michael Collins (October 31, 1930 – April 28, 2021) is a retired American astronaut and test pilot who was the Command Module Pilot for the Apollo 11 mission in 1969."
- Michael Collins was born on October 31, 1930
- Michael Collins died on April 28, 2021
- Michael Collins is a retired American astronaut
- Michael Collins is a retired test pilot
- Michael Collins was the Command Module Pilot for the Apollo 11 mission in 1969

Example 4:
Input sentence: In addition to his acting roles, Bateman has written and directed two short films and is currently in development on his feature debut.
Output:
- Bateman has acting roles.
- Bateman has written two short films.
- Bateman has directed two short films.
- Bateman is currently in development on his feature debut.
"""

GENERATION_INSTRUCT_PROMPT = """Task: Given the following passage, break it into individual, independent facts. Ensure that each statement does not rely on context from other statements. Replace all pronouns (e.g., 'he,' 'she,' 'it,' 'they') with the corresponding nouns or proper names to make the meaning clear without additional context. Do not change anything in the citations. If the passage is inadequate or doesn't contain any information, answer "No facts to extract".
Example:
Input passage: "Albert Einstein (1879–1955) was a German-born theoretical physicist best known for developing the theory of relativity, one of the two pillars of modern physics. Einstein was born in Ulm, in the Kingdom of Württemberg in the German Empire. He studied physics and mathematics in Switzerland and later worked as a patent examiner while developing his groundbreaking theories. In 1905, he published four major papers, including the special theory of relativity, during what is now called his Annus Mirabilis (miracle year).
Output:
- Albert Einstein was born in 1879.
- Albert Einstein died in 1955.
- Albert Einstein was a German-born theoretical physicist.
- Albert Einstein is best known for developing the theory of relativity.
- The theory of relativity is one of the two pillars of modern physics.
- Albert Einstein was born in Ulm.
- Ulm was in the Kingdom of Württemberg.
- The Kingdom of Württemberg was part of the German Empire.
- Albert Einstein studied and mathematics in Switzerland.
- Albert Einstein developed his groundbreaking theories while working as a patent examiner.
- In 1905, Albert Einstein published four major papers including the special theory of relativity.
- The year 1905 is called Albert Einstein’s Annus Mirabilis (miracle year).
"""

class GenerationAtomicFactGenerator():
    demos = GENERATION_INSTRUCT_PROMPT

    def __init__(self, llm: APICompletions):
        self.llm = llm
        self.segmenter = Segmenter(language="en")
    
    def run(self, generation):
        """
        Converts the generation into a set of atomic facts.
        """
        assert isinstance(generation, str), "generation must be a string"

        return self.get_atomic_facts_from_generation(generation)
    
    async def get_atomic_facts_from_generation(self, generation):
        prompt = (
            self.demos
            + f"""Now process the following passage:\nInput passage: "{generation}"\nOutput\n:"""
            )

        response = await self.llm.generate([prompt])

        gen_to_facts = {}

        if response is not None:
            gen_to_facts[generation] = await self.text_to_facts(response[0])

            return gen_to_facts
    
    async def text_to_facts(self, text):
        """
        Breaks LLM's output into facts removing all LLM extra notes
        """
        facts = text.split("- ")[1:]
        facts = [
            fact.strip()[:-1]
            if len(fact) > 0 and fact.strip()[-1] == "\n"
            else fact.strip()
            for fact in facts
        ]
        facts = [re.sub(r"\n\n.*", "", fact, flags=re.DOTALL).strip() for fact in facts]
        if len(facts) > 0:
            if facts[-1][-1] != ".":
                facts[-1] = facts[-1] + "."
        return facts



class AtomicFactGenerator():
    demos = SENTENCE_INSTRUCT_PROMPT

    def __init__(self, llm: APICompletions):
        self.llm = llm
        self.segmenter = Segmenter(language="en")

    def run(self, generation):
        """
        Converts the generation into a set of atomic facts.
        """
        assert isinstance(generation, str), "generation must be a string"
        paragraphs = [
            para.strip() for para in generation.split("\n") if len(para.strip()) > 0
        ]
        return self.get_atomic_facts_from_paragraph(paragraphs)

    async def get_atomic_facts_from_paragraph(self, paragraphs):
        sentences = []
        para_breaks = []  # store indices of paragraph ends

        for para_idx, paragraph in enumerate(paragraphs):
            if para_idx > 0:
                para_breaks.append(len(sentences))

            curr_sentences = self.segmenter.segment(paragraph)  # list[str]
            sentences += curr_sentences

        atomic_facts_pairs = await self.get_atomic_facts_from_sentence(sentences)

        return atomic_facts_pairs, para_breaks

    async def get_atomic_facts_from_sentence(self, sentences):
        prompts = []
        for sentence in sentences:
            prompt = (
                self.demos
                + f"""Now process the following sentence:\nInput sentence: "{sentence}"\nOutput\n:"""
            )
            prompts.append(prompt)

        responses = await self.llm.generate(prompts)

        sent_to_facts = {}
        if responses is not None:
            for i, output in enumerate(responses):
                sent_to_facts[sentences[i]] = await self.text_to_facts(output)
            return sent_to_facts

    async def text_to_facts(self, text):
        """
        Breaks LLM's output into facts removing all LLM extra notes
        """
        facts = text.split("- ")[1:]
        facts = [
            fact.strip()[:-1]
            if len(fact) > 0 and fact.strip()[-1] == "\n"
            else fact.strip()
            for fact in facts
        ]
        facts = [re.sub(r"\n\n.*", "", fact, flags=re.DOTALL).strip() for fact in facts]
        if len(facts) > 0:
            if facts[-1][-1] != ".":
                facts[-1] = facts[-1] + "."
        return facts
