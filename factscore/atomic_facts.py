import asyncio
import re
from pysbd import Segmenter

from factscore.api_requests import APICompletions

SENTENCE_INSTRUCT_PROMPT = """Task: Given the following sentence, break it into individual, independent facts. Ensure that each statement is self-contained and does not rely on context from other statements. Replace all pronouns (e.g., 'he,' 'she,' 'it,' 'they') with the corresponding nouns or proper names to make the meaning clear without additional context. Do not change anything in the citations.

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


class AtomicFactGenerator(object):
    demos = SENTENCE_INSTRUCT_PROMPT

    def __init__(self, llm: APICompletions):
        self.llm = llm
        self.segmenter = Segmenter(language="en")

    def run(self, generation):
        """
        Convert the generation into a set of atomic facts.
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
                + f"""Now process the following sentence:\nInput sentence: "{sentence}"\nOutput\n:""")
            prompts.append(prompt)

        responses = await self.llm.generate(prompts)

        sent_to_facts = {}  # dict {sentence: facts from the sentence}
        if responses is not None:
            for i, output in enumerate(responses):
                sent_to_facts[sentences[i]] = await self.text_to_facts(output)
            return sent_to_facts

    async def text_to_facts(self, text):
        """
        Breaks LLM's output into facts and remove from them any llm notes
        (sometimes llm returns outputs like "<fact>\n\n<note of the llm>",
        that is inappropriate because we want just the fact without any extra information)
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
