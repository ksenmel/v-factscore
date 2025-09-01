from flair.data import Sentence
from flair.nn import Classifier


class EntitiesRetriever:
    def __init__(self):
        self.tagger = Classifier.load("ner")

    def run(self, atoms):
        assert isinstance(atoms, list), "generation must be a list"

        return self.get_entities(atoms)

    def get_entities(self, atoms):
        atoms_ents = {}

        for atom in atoms:
            ents = []
            sent = Sentence(atom)
            self.tagger.predict(sent)

            for label in sent.get_labels():
                if str(label.value) == "PER":
                    ents.append(label.data_point.text)

            atoms_ents[atom] = ents

        return atoms_ents
