import json

from langchain_openai import ChatOpenAI
from pysbd import Segmenter


class AtomicFactGenerator(object):
    def __init__(self, llm: ChatOpenAI, prompt_config: str):
        self.is_bio = True
        self.llm = llm

        with open(prompt_config, "r") as f:
            self.demos = json.load(f)

    def run(self, generation):
        assert isinstance(generation, str), "generation must be a string"
        paragraphs = [
            para.strip() for para in generation.split("\n") if len(para.strip()) > 0
        ]
        return self.get_atomic_facts_from_paragraph(paragraphs)

    def get_atomic_facts_from_paragraph(self, paragraphs):
        sentences = []
        para_breaks = []  # store indices of paragraph ends

        for para_idx, paragraph in enumerate(paragraphs):
            if para_idx > 0:
                para_breaks.append(len(sentences))

            curr_sentences = split_into_sentences(paragraph)

            sentences += curr_sentences

        atoms = self.get_init_atomic_facts_from_sentence(sentences, n=2)
        print(atoms)

        atomic_facts_pairs = []
        for i, sent in enumerate(sentences):
            atomic_facts_pairs.append((sent, atoms[i]))

        return atomic_facts_pairs, para_breaks

    def get_init_atomic_facts_from_sentence(self, sentences, n):
        demos = self.demos
        atoms = []
        for sentence in sentences:
            prompt = ""

            for i in range(n):
                demo_sentence = list(demos.keys())[i]
                prompt += "Please breakdown the following sentence into independent facts: {}\n".format(
                    demo_sentence
                )

                for fact in demos[demo_sentence]:
                    prompt += "- {}\n".format(fact)
                prompt += "\n"

            prompt = (
                prompt
                + "Please breakdown the following sentence into independent facts: {}\n".format(
                    sentence
                )
            )

            response = self.llm.invoke(prompt)

            facts = [fact.strip(" -") for fact in str(response.content).split("\n-")]
            atomic_facts = [fact.strip(" .") for fact in facts]
            atoms.append(atomic_facts)

        return atoms


def split_into_sentences(text: str) -> list[str]:
    segmenter = Segmenter(language="en", clean=False)
    sents = segmenter.segment(text)
    return sents


def main():
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        model="gpt-3.5-turbo"
    )
    generator = AtomicFactGenerator(llm, "factscore/utils/demos.json")
    atomic_facts, para_breaks = generator.run(
        'Elvis Presley, often referred to as the "King of Rock and Roll," was one of the most influential and iconic musicians in the history of popular music. Born on January 8, 1935, in Tupelo, Mississippi, Elvis grew up in a working-class family. His rise to stardom began in the mid-1950s, when he signed with Sun Records in Memphis. His early recordings blended various musical genres, including country, blues, and gospel, and created a new sound that captivated audiences.\n\nElvis\'s first hit single, "Heartbreak Hotel," released in 1956, catapulted him to national fame. His charismatic stage presence, unique voice, and style revolutionized the music industry. His performances, often provocative and energetic, were a sharp contrast to the more conservative music scene of the time, making him a symbol of youth rebellion and cultural change.'
    )

    print(atomic_facts)
    print(para_breaks)


if __name__ == "__main__":
    main()
