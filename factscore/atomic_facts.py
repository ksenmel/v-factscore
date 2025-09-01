import re
from pysbd import Segmenter

from factscore.api_requests import APICompletions

SENTENCE_INSTRUCT_PROMPT = """Task: Given the following sentence, break it into individual, independent facts. Ensure that each statement does not rely on context from other statements. Replace all pronouns (e.g., 'he,' 'she,' 'it,' 'they') with the corresponding nouns or proper names to make the meaning clear without additional context. Do not change anything in the citations. If the sentence is inadequate or doesn't contain any information, answer "No facts to extract". Follow the style of output in example.

Example 1:
Input Sentence: "Michael Collins (October 31, 1930 – April 28, 2021) is a retired American astronaut and test pilot who was the Command Module Pilot for the Apollo 11 mission in 1969."
- Michael Collins was born on October 31, 1930
- Michael Collins died on April 28, 2021
- Michael Collins is a retired American astronaut
- Michael Collins is a retired test pilot
- Michael Collins was the Command Module Pilot for the Apollo 11 mission in 1969

Example 2:
Input sentence: In addition to his acting roles, Bateman has written and directed two short films and is currently in development on his feature debut.
Output:
- Bateman has acting roles.
- Bateman has written two short films.
- Bateman has directed two short films.
- Bateman is currently in development on his feature debut.
"""

GENERATION_INSTRUCT_PROMPT = """Task: Given the following passage, break it into individual, independent facts. Ensure that each statement does not rely on context from other statements. Replace all pronouns (e.g., 'he,' 'she,' 'it,' 'they') with the corresponding nouns or proper names to make the meaning clear without additional context. Do not change anything in the citations. If the passage is inadequate or doesn't contain any information, answer "No facts to extract". Follow the style of output in example.
Example:
Input passage: "Yuri Gagarin (9 March 1934 – 27 March 1968) was a Soviet pilot and cosmonaut who, aboard the first successful crewed spaceflight, became the first person to journey into outer space. Travelling on Vostok 1, Gagarin completed one orbit of Earth on 12 April 1961, with his flight taking 108 minutes. By achieving this major milestone for the Soviet Union amidst the Space Race, he became an international celebrity and was awarded many medals and titles, including his country's highest distinction: Hero of the Soviet Union."
Output:
- Yuri Gagarin was born on 9 March 1934.
- Yuri Gagarin died on 27 March 1968.
- Yuri Gagarin was a Soviet pilot.
- Yuri Gagarin was a Soviet cosmonaut.
- Yuri Gagarin flew aboard the first successful crewed spaceflight. 
- Yuri Gagarin was the first person to journey into outer space.
- Yuri Gagarin completed one orbit of Earth on 12 April 1961.
- Yuri Gagarin completed one orbit of Earth on 12 April 1961 on Vostok 1.
- Yuri Gagarin's orbit of Earth on 12 April 1961 took 108 minutes.
- Yuri Gagarin’s achievement was a major milestone for the Soviet Union.
- Yuri Gagarin’s spaceflight occurred during the Space Race.
- Yuri Gagarin became an international celebrity after the flight on 12 April 1961.
- Yuri Gagarin was awarded many medals and titles.
- Yuri Gagarin was awarded as Hero of the Soviet Union.
- Hero of the Soviet Union is the country’s highest distinction.

"""


class GenerationAtomicFactGenerator:
    """
    Converts text generations into atomic facts using a language model.

    This class:
    - Uses instructional prompts to guide fact extraction
    - Leverages a language model to decompose text into atomic facts
    - Cleans and formats the extracted facts
    - Provides text segmentation capabilities

    Attributes:
        demos (str): Instructional prompt template for fact extraction
        llm (APICompletions): Language model client for fact extraction
        segmenter (Segmenter): Text segmentation component
    """

    demos = GENERATION_INSTRUCT_PROMPT

    def __init__(self, llm: APICompletions):
        """
        Initialize the fact generator with a language model.

        Args:
            llm: Language model client for generating completions
        """
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

        response, failed, costs = await self.llm.generate([prompt])

        gen_to_facts = {}

        if response is not None:
            gen_to_facts[generation] = await self.text_to_facts(response[0])

            return gen_to_facts, costs

    async def text_to_facts(self, text):
        """
        Breaks LLM's output into facts removing all LLM extra notes
        """
        facts = text.split("- ")[1:]
        facts = [
            (
                fact.strip()[:-1]
                if len(fact) > 0 and fact.strip()[-1] == "\n"
                else fact.strip()
            )
            for fact in facts
        ]
        facts = [re.sub(r"\n\n.*", "", fact, flags=re.DOTALL).strip() for fact in facts]
        if len(facts) > 0:
            if facts[-1][-1] != ".":
                facts[-1] = facts[-1] + "."
        return facts


class AtomicFactGenerator:
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
            (
                fact.strip()[:-1]
                if len(fact) > 0 and fact.strip()[-1] == "\n"
                else fact.strip()
            )
            for fact in facts
        ]
        facts = [re.sub(r"\n\n.*", "", fact, flags=re.DOTALL).strip() for fact in facts]
        if len(facts) > 0:
            if facts[-1][-1] != ".":
                facts[-1] = facts[-1] + "."
        return facts
