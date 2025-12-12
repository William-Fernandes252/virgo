from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown

from virgo import Virgo
from virgo.graph import builder
from virgo.schemas import MarkdownArticle

app = typer.Typer()
console = Console()


@app.command()
def generate(
    input: Annotated[
        str, typer.Argument(..., help="The input question to generate an article for.")
    ],
):
    """Generate an article using the Virgo assistant."""
    virgo = Virgo(builder)

    with console.status("[bold green]Generating article...[/bold green]"):
        res = virgo.generate(input)

    formatted_article: MarkdownArticle | None = res.get("formatted_article")
    if formatted_article:
        markdown_content = formatted_article.to_markdown()
        console.print(Markdown(markdown_content))
    else:
        # Fallback to raw message if formatting failed
        from typing import cast

        from langchain_core.messages import BaseMessage

        typer.secho(
            cast(BaseMessage, res["messages"][-1]).content, fg=typer.colors.YELLOW
        )


if __name__ == "__main__":
    app()
