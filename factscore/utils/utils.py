from pysbd import Segmenter


def split_into_sentences(text: str) -> list[str]:
    segmenter = Segmenter(language='en', clean=False)
    sents = segmenter.segment(text)
    return sents
