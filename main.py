import typer

app = typer.Typer()


@app.command()
def generate(): ...


if __name__ == "__main__":
    app()
