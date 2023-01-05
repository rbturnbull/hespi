from typer.testing import CliRunner

from hespi import main

runner = CliRunner()


def test_help():
    result = runner.invoke(main.app, ["--help"])
    assert result.exit_code == 0
    assert "output" in result.stdout


def test_adjust_case():
    assert "Acanthochlamydaceae" == main.adjust_case("family", "ACANTHOCHLAMYDACEAE")
    assert "Abutilon" == main.adjust_case("genus", "abutilon")
    assert "zostericolum" == main.adjust_case("species", "Zostericolum")    