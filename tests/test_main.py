from typer.testing import CliRunner

from hespi.main import app

runner = CliRunner()


def test_app():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "output" in result.stdout
