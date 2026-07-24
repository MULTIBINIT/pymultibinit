import numpy as np

from pymultibinit.pyeffpot.ddb_parser_complete import read_ddb


def test_parser_reads_abinit_symrel_order_and_tnons(tmp_path):
    ddb_path = tmp_path / "symrel_tnons.DDB"
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=int)
    symrel_fortran = "".join(f"{value:5d}" for value in rotation.reshape(-1, order="F"))
    ddb_path.write_text(
        "\n".join(
            [
                " **** DERIVATIVE DATABASE ****    ",
                "+DDB, Version number  20230401",
                "",
                " parser convention regression",
                "",
                "    usepaw         0",
                "     natom         1",
                "      nkpt         1",
                "    nsppol         1",
                "      nsym         1",
                "    ntypat         1",
                "    occopt         1",
                "     nband         1",
                "     acell  1.00000000000000D+00  1.00000000000000D+00  1.00000000000000D+00",
                "       amu  1.00000000000000D+00",
                "     rprim  1.00000000000000D+00  0.00000000000000D+00  0.00000000000000D+00",
                "            0.00000000000000D+00  1.00000000000000D+00  0.00000000000000D+00",
                "            0.00000000000000D+00  0.00000000000000D+00  1.00000000000000D+00",
                " dfpt_sciss  0.00000000000000D+00",
                f"    symrel     {symrel_fortran}",
                "     tnons  2.50000000000000D-01  0.00000000000000D+00  5.00000000000000D-01",
                "     typat         1",
                "       wtk  1.00000000000000D+00",
                "      xred  0.00000000000000D+00  0.00000000000000D+00  0.00000000000000D+00",
                "     znucl  1.00000000000000D+00",
                "      zion  1.00000000000000D+00",
                " **** Database of total energy derivatives ****",
                " ",
                " Number of data blocks=    1",
                "",
                " 2nd derivatives (non-stat.)  - # elements :           1",
                " qpt  5.00000000E-01  0.00000000E+00  0.00000000E+00   1.0",
                "   1    1    1    1   1.00000000000000D+00   0.00000000000000D+00",
                "",
            ]
        ),
        encoding="utf-8",
    )

    unitcell = read_ddb(str(ddb_path))

    assert np.array_equal(unitcell.symrel[0], rotation)
    assert np.allclose(unitcell.tnons[0], [0.25, 0.0, 0.5])
    assert np.linalg.norm(unitcell.zeff) == 0.0


def test_parser_extracts_gamma_elastic_and_internal_strain_terms(tmp_path):
    ddb_path = tmp_path / "strain_terms.DDB"
    ddb_path.write_text(
        "\n".join(
            [
                " **** DERIVATIVE DATABASE ****    ",
                "+DDB, Version number  20230401",
                "",
                " strain convention regression",
                "",
                "    usepaw         0",
                "     natom         1",
                "      nkpt         1",
                "    nsppol         1",
                "      nsym         1",
                "    ntypat         1",
                "    occopt         1",
                "     nband         1",
                "     acell  1.00000000000000D+00  1.00000000000000D+00  1.00000000000000D+00",
                "       amu  1.00000000000000D+00",
                "     rprim  1.00000000000000D+00  0.00000000000000D+00  0.00000000000000D+00",
                "            0.00000000000000D+00  1.00000000000000D+00  0.00000000000000D+00",
                "            0.00000000000000D+00  0.00000000000000D+00  1.00000000000000D+00",
                " dfpt_sciss  0.00000000000000D+00",
                "    symrel         1    0    0    0    1    0    0    0    1",
                "     tnons  0.00000000000000D+00  0.00000000000000D+00  0.00000000000000D+00",
                "     typat         1",
                "       wtk  1.00000000000000D+00",
                "      xred  0.00000000000000D+00  0.00000000000000D+00  0.00000000000000D+00",
                "     znucl  1.00000000000000D+00",
                "      zion  1.00000000000000D+00",
                " **** Database of total energy derivatives ****",
                " ",
                " Number of data blocks=    1",
                "",
                " 2nd derivatives (non-stat.)  - # elements :           4",
                " qpt  0.00000000E+00  0.00000000E+00  0.00000000E+00   1.0",
                "   1    1    1    1   1.00000000000000D+00   0.00000000000000D+00",
                "   1    4    1    4   2.00000000000000D+00   0.00000000000000D+00",
                "   2    4    3    5   5.00000000000000D-01   0.00000000000000D+00",
                "   2    1    1    4   4.00000000000000D+00   0.00000000000000D+00",
                "",
            ]
        ),
        encoding="utf-8",
    )

    unitcell = read_ddb(str(ddb_path))

    assert unitcell.elastic_constants[0, 0] == 2.0
    assert unitcell.elastic_constants[1, 5] == 0.5
    assert unitcell.elastic_constants[5, 1] == 0.0
    assert unitcell.strain_coupling[0, 1, 0] == 0.0
    assert np.allclose(unitcell.strain_coupling.sum(axis=2), 0.0)
