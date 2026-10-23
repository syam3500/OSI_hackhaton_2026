from __future__ import annotations

from functools import lru_cache
from io import BytesIO
import re

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, Draw, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
DESCRIPTORS = {
    "mol_weight": "Molecular weight (g/mol)",
    "logp": "Crippen logP",
    "tpsa": "TPSA (Å²)",
    "hbd": "H-bond donors",
    "hba": "H-bond acceptors",
    "rotatable_bonds": "Rotatable bonds",
    "formal_charge": "Formal charge",
}


def valid_cas(value: str) -> bool:
    if not re.fullmatch(r"[1-9]\d{1,6}-\d{2}-\d", value):
        return False
    digits = value.replace("-", "")
    return sum(index * int(digit) for index, digit in enumerate(digits[-2::-1], 1)) % 10 == int(digits[-1])


def extract_cas(value) -> list[str]:
    found = re.findall(r"(?<!\d)\d{2,7}-\d{2}-\d(?!\d)", str(value))
    return sorted({number for number in found if valid_cas(number)})


@lru_cache(maxsize=25000)
def structure(smiles: str) -> dict | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        original = Chem.MolFromSmiles(smiles.strip())
        if original is None or original.GetNumAtoms() == 0:
            return None
        molecule = rdMolStandardize.Cleanup(original)
        molecule = rdMolStandardize.LargestFragmentChooser(preferOrganic=True).choose(molecule)
        molecule = rdMolStandardize.Uncharger().uncharge(molecule)
        Chem.SanitizeMol(molecule)
        return {
            "smiles_raw": Chem.MolToSmiles(original),
            "inchikey": Chem.MolToInchiKey(original),
            "smiles_std": Chem.MolToSmiles(molecule),
            "inchikey_std": Chem.MolToInchiKey(molecule),
            "mol_weight": Descriptors.MolWt(molecule),
            "logp": Crippen.MolLogP(molecule),
            "tpsa": rdMolDescriptors.CalcTPSA(molecule),
            "hbd": rdMolDescriptors.CalcNumHBD(molecule),
            "hba": rdMolDescriptors.CalcNumHBA(molecule),
            "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(molecule),
            "formal_charge": Chem.GetFormalCharge(molecule),
        }
    except (ValueError, RuntimeError):
        return None


def fingerprint(smiles: str):
    return GENERATOR.GetFingerprint(Chem.MolFromSmiles(smiles))


def fingerprint_matrix(smiles: list[str]) -> np.ndarray:
    matrix = np.zeros((len(smiles), 2048), dtype=np.float32)
    for index, value in enumerate(smiles):
        DataStructs.ConvertToNumpyArray(fingerprint(value), matrix[index])
    return matrix


def molecule_image(smiles: str) -> bytes:
    molecule = Chem.MolFromSmiles(smiles)
    image = Draw.MolToImage(molecule, size=(440, 250))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
