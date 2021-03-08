# Use other format instead? Json etc

from dataclasses import dataclass
from collections import defaultdict


@dataclass
class Config:
    epoched = defaultdict(lambda: defaultdict(lambda: 'file', {'ER': None}))
    crop = defaultdict(lambda: {})
    # folder = '/volatile/Multifracs/meg/{subject}/'
    folder = '/media/merlin/Media/ScaledTime/MEG/{subject}/'
    filename = '{subject}_ScaledTime_{run}_{extension}.fif'
    timings_file = '{subject}_timings.json'

    def __post_init__(self):

        # self.epoched['eb180237']['ER'] = None

        # self.epoched['ag170045']['RS01'] = 2
        # self.epoched['ag170045']['ER'] = 2

        # self.crop['eb180237']['ER'] = (50.0, None)
        self.crop['gl180335']['ER'] = (30.0, None)


config = Config()
