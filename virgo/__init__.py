from langgraph.graph import StateGraph


class Virgo:
    """The main Virgo assistant class."""

    def __init__(self, graph_builder: StateGraph):
        """Initialize the Virgo assistant."""
        self.builder = graph_builder
        self.graph = self.builder.compile()

    def generate(self, input_question: str):
        """Generate an article based on the input question."""
        from langchain_core.messages import HumanMessage

        message = HumanMessage(content=input_question)
        return self.graph.invoke({"messages": [message]})  # type: ignore[arg-type]
