import logging
import os
from argparse import ArgumentParser

from preprocessing.bus_cot import BUSCoT

from dtos.breast_cancer_dataset import BreastCancerDataset
from preprocessing.advanced_mri_lesions import AdvancedMRILesions
from preprocessing.breast_lesion_usg import BreastLesionUSG
from preprocessing.breast_micro_calc import BreastMicroCalc
from preprocessing.cbis_ddsm import CbisDDSM
from preprocessing.embed import Embed
from preprocessing.inbreast import Inbreast
from preprocessing.la_breast import LABreast
from preprocessing.mama_mia import MamaMia
from preprocessing.oasbud import Oasbud
from preprocessing.tompei_cmmd import TompeiCMMD

if __name__ == "__main__":
    dataset_mapping = {
        "advanced-mri-lesions": lambda: AdvancedMRILesions(),
        "breast-lesions-usg": lambda: BreastLesionUSG(),
        "bus-cot": lambda: BUSCoT(),
        "cbis-ddsm": lambda: CbisDDSM(),
        "embed": lambda: Embed(),
        "tompei-cmmd": lambda: TompeiCMMD(),
        "breast-micro-calc": lambda: BreastMicroCalc(),
        "mama-mia": lambda: MamaMia(),
        "oasbud": lambda: Oasbud(),
        "inbreast": lambda: Inbreast(),
        "la-breast": lambda: LABreast(),
    }

    args = ArgumentParser()
    args.add_argument(
        "--dataset", type=str, required=True, choices=dataset_mapping.keys()
    )
    args.add_argument("--debug", action="store_true")
    args = args.parse_args()

    os.makedirs("logs", exist_ok=True)
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
