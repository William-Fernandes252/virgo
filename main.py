from typing import Annotated, cast

import typer
from langchain_core.messages import BaseMessage, HumanMessage

from virgo.graph import builder

app = typer.Typer()


@app.command()
def generate(
    input: Annotated[
        str, typer.Argument(..., help="The input question to generate an article for.")
    ],
):
    """Generate an article using the Virgo assistant."""
    graph = builder.compile()

    message = HumanMessage(content=input)
    res = graph.invoke({"messages": [message]})  # type: ignore[arg-type]

    typer.secho(cast(BaseMessage, res["messages"][-1]).content, fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
