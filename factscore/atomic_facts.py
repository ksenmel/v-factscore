from langchain_openai import ChatOpenAI
import spacy


class AtomicFactGenerator(object):
    def __init__(self, llm: ChatOpenAI):
        self.nlp = spacy.load("en_core_web_sm")
        self.llm = llm
        self.is_bio = True

    def get_init_atomic_facts_from_sentence(self, sentences):
        atomic_facts = {}
        for sent in sentences:
            prompt = (
                    """ Please breakdown the following sentence into independent facts: He made his acting debut in the film The Moon is the Sun's Dream (1992), and continued to appear in small and supporting roles throughout the 1990s.
- He made his acting debut in the film.
- He made his acting debut in The Moon is the Sun's Dream.
- The Moon is the Sun's Dream is a film.
- The Moon is the Sun's Dream was released in 1992.
- After his acting debut, he appeared in small and supporting roles.
- After his acting debut, he appeared in small and supporting roles throughout the 1990s.\nHe is also a successful producer and engineer, having worked with a wide variety of artists, including Willie Nelson, Tim McGraw, and Taylor Swift.
- He is successful.
- He is a producer.
- He is a engineer.
- He has worked with a wide variety of artists.
- Willie Nelson is an artist.
- He has worked with Willie Nelson.
- Tim McGraw is an artist.
- He has worked with Tim McGraw.
- Taylor Swift is an artist.
- He has worked with Taylor Swift.\nPlease breakdown the following sentence into independent facts: """
                    + sent
            )

            response = self.llm.invoke(prompt)

            facts = [
                fact.strip() for fact in response.content.split(",") if fact.strip()
            ]
            atomic_facts[sent] = facts

        return atomic_facts

