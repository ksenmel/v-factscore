import asyncio
import re
import os

import sys
from pysbd import Segmenter
from factscore.api_requests import APICompletions

# how many examples should be?
SENTENCE_INSTRUCT_PROMPT = """
Task: Given the following sentence, break it into individual, independent facts.

Example 1:
Input sentence: Michael Collins (born October 31, 1930) is a retired American astronaut and test pilot who was the Command Module Pilot for the Apollo 11 mission in 1969.
Output:
- Michael Collins was born on October 31, 1930.
- Michael Collins is retired.
- Michael Collins is an American.
- Michael Collins was an astronaut.
- Michael Collins was a test pilot.
- Michael Collins was the Command Module Pilot.
- Michael Collins was the Command Module Pilot for the Apollo 11 mission.
- Michael Collins was the Command Module Pilot for the Apollo 11 mission in 1969.

Example 2:
Input sentence: He was an American composer, conductor, and musical director.
Output:
- He was an American.
- He was a composer.
- He was a conductor.
- He was a musical director.

Example 3:
Input sentence: She currently stars in the romantic comedy series, Love and Destiny, which premiered in 2019.
Output:
- She currently stars in Love and Destiny.
- Love and Destiny is a romantic comedy series.
- Love and Destiny premiered in 2019.

Example 4:
Input sentence: He is also a producer and engineer, having worked with a wide variety of artists, including Willie Nelson, Tim McGraw, and Taylor Swift.
Output:
- He is a producer.
- He is an engineer.
- He has worked with a wide variety of artists.
- Willie Nelson is an artist.
- He has worked with Willie Nelson.
- Tim McGraw is an artist.
- He has worked with Tim McGraw.
- Taylor Swift is an artist.
- He has worked with Taylor Swift.

Example 5:
Input sentence: He made his acting debut in the film The Moon is the Sun's Dream (1992), and continued to appear in small and supporting roles throughout the 1990s.
Output:
- He made his acting debut in the film.
- He made his acting debut in The Moon is the Sun's Dream.
- The Moon is the Sun's Dream is a film.
- The Moon is the Sun's Dream was released in 1992.
- After his acting debut, he appeared in small and supporting roles.
- After his acting debut, he appeared in small and supporting roles throughout the 1990s.

Example 6:
Input sentence: In 1963, Collins became one of the third group of astronauts selected by NASA, and he served as the back-up Command Module Pilot for the Gemini 7 mission.
Output:
- Collins became an astronaut.
- Collins became one of the third group of astronauts.
- Collins became one of the third group of astronauts selected by NASA.
- Collins became one of the third group of astronauts selected by NASA in 1963.
- He served as the Command Module Pilot.
- He served as the back-up Command Module Pilot.
- He served as the Command Module Pilot for the Gemini 7 mission.

Example 7:
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
        self.segmenter = Segmenter(language="en", clean=False)

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
                + f"""
                                   Now process the following sentence:\nInput sentence: "{sentence}"\nOutput:
                                   """
            )
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


if __name__ == "__main__":
    llm = APICompletions(
        base_url="",
        model_name="deepseek-ai/DeepSeek-R1",
    )
    generator = AtomicFactGenerator(llm)

    result = asyncio.run(
        generator.run(
            'Elvis Presley, often referred to as the "King of Rock and Roll," was one of the most influential and iconic musicians in the history of popular music. Born on January 8, 1935, in Tupelo, Mississippi, Elvis grew up in a working-class family. His rise to stardom began in the mid-1950s, when he signed with Sun Records in Memphis. His early recordings blended various musical genres, including country, blues, and gospel, and created a new sound that captivated audiences.\n\nElvis\'s first hit single, "Heartbreak Hotel," released in 1956, catapulted him to national fame. His charismatic stage presence, unique voice, and style revolutionized the music industry. His performances, often provocative and energetic, were a sharp contrast to the more conservative music scene of the time, making him a symbol of youth rebellion and cultural change.'
        )
    )

    atomic_facts, para_breaks = result

    print(atomic_facts)
    print(para_breaks)
