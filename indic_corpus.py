"""indic_corpus dataset."""

import tensorflow_datasets as tfds

# TODO(indic_corpus): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(indic_corpus): BibTeX citation
_CITATION = """
@inproceedings{kakwani2020indicnlpsuite,
    title={{IndicNLPSuite: Monolingual Corpora, Evaluation Benchmarks and Pre-trained Multilingual Language Models for Indian Languages}},
    author={Divyanshu Kakwani and Anoop Kunchukuttan and Satish Golla and Gokul N.C. and Avik Bhattacharyya and Mitesh M. Khapra and Pratyush Kumar},
    year={2020},
    booktitle={Findings of EMNLP},
}
"""

VERSION = tfds.core.Version("1.0.0")

RELEASE_NOTES = {
    "1.0.0": "Initial release.",
}

INDIC_LANGS = [
    "as", "bn", "en", "gu", "hi", "kn",
    "ml", "mr", "or", "pa", "ta", "te"
]

class IndicCorpusConfig(tfds.core.BuilderConfig):
    """BuilderConfig for IndicCorpus"""

    def __init__(self, name, languages, **kwargs):
        super(IndicCorpusConfig, self).__init__(name=name, version=VERSION, **kwargs)

        self.languages = languages


class IndicCorpus(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for indic_corpus dataset."""

    BUILDER_CONFIGS = [
        IndicCorpusConfig(
            "indic_corp",
            languages=INDIC_LANGS,
            description="Indic Corp has 12 languages and is generated from web sources"
        )
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(indic_corpus): Specifies the tfds.core.DatasetInfo object
        features = {
            "text": tfds.features.Text()
        }
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(features),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            # supervised_keys=("image", "label"),  # Set to `None` to disable
            homepage="https://indicnlp.ai4bharat.org/home/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(indic_corpus): Downloads the data and defines the splits
        path = dl_manager.download_and_extract("https://storage.googleapis.com/ai4bharat-public-indic-nlp-corpora/indiccorp/all_langs/monoling.zip")

        # TODO(indic_corpus): Returns the Dict[split names, Iterator[Key, Example]]
        splits = []
        for lang in self.builder_config.languages:
            splits.extend([
                tfds.core.SplitGenerator(
                    name=lang,
                    gen_kwargs=dict(
                        split=lang,
                        path=path/f'monoling/{lang}.txt'
                    )

                ),
                tfds.core.SplitGenerator(
                    name=f'{lang}-validation',
                    gen_kwargs=dict(
                        split=f'{lang}-validation',
                        path=path/f'monoling/{lang}-validation.txt'
                    )
                )
            ])
        return splits

    def _generate_examples(self, split, path):
        """Yields examples."""
        beam = tfds.core.lazy_imports.apache_beam

        def _process_file(path):
            with open(path, 'r') as f:
                lines = f.read()
                lines = lines.split('\n')[:-1]
                for id_, line in enumerate(lines):
                    yield id_, {
                        "text": line
                    }

        return (
            beam.Create(path)
            | beam.Map(_process_file)
        )