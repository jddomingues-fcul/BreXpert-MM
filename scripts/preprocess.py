import logging
from argparse import ArgumentParser

from dtos.breast_cancer_dataset import BreastCancerDataset
from preprocessing.advanced_mri_lesions import AdvancedMRILesions
from preprocessing.breast_lesion_usg import BreastLesionUSG
from preprocessing.breast_micro_calc import BreastMicroCalc
from preprocessing.bus_bra import BUSBRA
from preprocessing.cbis_ddsm import CbisDDSM
from preprocessing.embed import Embed
from preprocessing.inbreast import Inbreast
from preprocessing.la_breast import LABreast
from preprocessing.mama_mia import MamaMia
from preprocessing.oasbud import Oasbud
from preprocessing.rsna_bcd import RsnaBCD
from preprocessing.tompei_cmmd import TompeiCMMD

if __name__ == "__main__":
    dataset_mapping = {
        "advanced-mri-lesions": lambda: AdvancedMRILesions(),
        "breast-lesions-usg": lambda: BreastLesionUSG(),
        "cbis-ddsm": lambda: CbisDDSM(),
        "embed": lambda: Embed(),
        "rsna-bcd": lambda: RsnaBCD(),
        "tompei-cmmd": lambda: TompeiCMMD(),
        "breast-micro-calc": lambda: BreastMicroCalc(),
        "mama-mia": lambda: MamaMia(),
        "oasbud": lambda: Oasbud(),
        "inbreast": lambda: Inbreast(),
        "la-breast": lambda: LABreast(),
        "bus-bra": lambda: BUSBRA(),
    }

    args = ArgumentParser()
    args.add_argument(
        "--dataset", type=str, required=True, choices=dataset_mapping.keys()
    )
    args.add_argument("--debug", action="store_true")
    args = args.parse_args()

    logging.basicConfig(
        filename=f"logs/{args.dataset}_processing.log",
        filemode="a",
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG if args.debug else logging.INFO,
    )

    dataset: BreastCancerDataset = dataset_mapping[args.dataset]()
    dataset.set_dataset_name(args.dataset)
    dataset.process_info()
    dataset.save_csv()
