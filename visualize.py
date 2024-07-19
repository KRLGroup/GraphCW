import numpy as np
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import Chem
import matplotlib.pyplot as plt
from PIL import Image
import io

from rdkit.Chem import rdFMCS
from rdkit.Chem import AllChem
CLR_MIN = (0.5, 0.5, 1)
CLR_MAX = (1, 0.5, 0.5)


def visualize_batch(mols, attrs, valLabels=[],
        colLabels=[], rowLabels=[], attrLabels=None, size=300, globalScores=False, align=False, text_size=14):
    if align:
        try:
            mcs = rdFMCS.FindMCS(mols)
            template = Chem.MolFromSmarts(mcs.smartsString)
            AllChem.Compute2DCoords(template)
            for m in mols:
                _ = AllChem.GenerateDepictionMatching2DStructure(m,template)
        except:
            pass
    rows, cols = len(mols), len(attrs)
    inch_s = size/100
    fig, axs = plt.subplots(
        rows, cols,
        figsize=(cols*inch_s, rows*inch_s)
    )
    if rows == 1:   axs = axs[None,:]
    elif cols == 1: axs = axs[:, None]

    # If there are colLabels missing, use Methods name
    if not colLabels:
        colLabels = list(attrs.keys())

    # If there are rowLabels missing, use SMILES
    if not rowLabels:
        rowLabels = [Chem.MolToSmiles(x) for x in mols]

    for c, method in enumerate(attrs.keys()):
        if globalScores:
            normScale = max([np.max(np.abs(x))
                             for x in attrs[method]])
        else: normScale = None

        for r, mol in enumerate(mols):
            if r == 0: axs[r,c].set_title(colLabels[c], fontsize=text_size)

            # Draw Vertical Description
            if c == 0:
                # If rowLabels are too long, cut them
                if len(rowLabels[r]) > 30: rowLabels[r] = rowLabels[r][:27] + '...'
                axs[r,c].text(
                    -0.1, 0.5, rowLabels[r], rotation=90,
                    size=text_size, va="center", transform=axs[r,c].transAxes
                )
            # Draw Values on the far right
            if len(valLabels) > 0 and c == cols-1:
                axs[r,c].text(
                    1.1, 0.5, valLabels[r], 
                    size=text_size, va="center", transform=axs[r,c].transAxes
                )

            img = visualize_attrs(
                mol,
                attrs[method][r],
                size=size,
                # normScale=normScale
            )
            axs[r,c].imshow(img)
            axs[r,c].axis('off')

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf)
    img = Image.open(buf)
    return img

def visualize_attrs(mol, attr, size=300, normScale=None):
    # Normalization
    if normScale: M = normScale
    else: M = np.max(np.abs(attr))
    attr /= M

    atom_clrs = {}
    bond_clrs = {}
    hit_ats = []
    hit_bonds = []
    for i, at in enumerate(attr):
        hit_ats.append(i)
        atom_clrs[i] = attr2clr(at)

    for i, bond in enumerate(mol.GetBonds()):
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        at = (attr[a2] + attr[a1]) / 2
        hit_bonds.append(i)
        bond_clrs[i] = attr2clr(at)

    d = rdMolDraw2D.MolDraw2DCairo(size, size)
    rdMolDraw2D.PrepareAndDrawMolecule(
        d, mol,
        highlightAtoms=hit_ats,
        highlightAtomColors=atom_clrs,
        highlightBonds=hit_bonds,
        highlightBondColors=bond_clrs
    )

    d.FinishDrawing()
    img_bytes = d.GetDrawingText()
    img = Image.open(io.BytesIO(img_bytes))
    return img

def attr2clr(a):
    CM = np.array(CLR_MAX)   # Max
    Cm = np.array(CLR_MIN)   # Min
    Cw = np.array((1, 1, 1)) # white
    GradPos = CM - Cw
    GradNeg = Cm - Cw
    
    if a > 0: clr = Cw + GradPos * a
    else: clr = Cw + GradNeg * -a

    return tuple(clr)
