"""ComposioLangChain class definition"""

import types
import typing as t
from inspect import Signature, Parameter

import pydantic
from langchain_core.tools import StructuredTool as BaseStructuredTool
from langchain_core.runnables.config import RunnableConfig

from composio.core.provider import AgenticProvider, AgenticProviderExecuteFn
from composio.types import Tool
from composio.utils.pydantic import parse_pydantic_error
from composio.utils.shared import (
    get_signature_format_from_schema_params,
    json_schema_to_model,
)


class StructuredTool(BaseStructuredTool):
    def run(self, *args, **kwargs):
        try:
            return super().run(*args, **kwargs)
        except pydantic.ValidationError as e:
            return {"successful": False, "error": parse_pydantic_error(e), "data": None}


class LanggraphProvider(
    AgenticProvider[StructuredTool, t.List[StructuredTool]],
    name="langgraph",
):
    """
    Composio toolset for Langchain framework.
    """

    def __init__(
        self,
        *args,
        add_runnable_config: bool = False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.add_runnable_config = add_runnable_config


    def _wrap_action(
        self,
        tool: str,
        description: str,
        schema_params: t.Dict,
        execute_tool: AgenticProviderExecuteFn,
    ):
        def function(**kwargs: t.Any) -> t.Dict:
            """Wrapper function for composio action."""
            return execute_tool(tool, kwargs)

        action_func = types.FunctionType(
            function.__code__,
            globals=globals(),
            name=tool,
            closure=function.__closure__,
        )

        parameters = get_signature_format_from_schema_params(
            schema_params=schema_params
        )

        if self.add_runnable_config:
            # Add RunnableConfig param with a collision-safe name
            parameters.append(
                Parameter(
                    "__runnable_config__",
                    kind=Parameter.KEYWORD_ONLY,
                    default=None,
                    annotation=RunnableConfig,
                )
            )

        action_func.__signature__ = Signature(  # type: ignore
            parameters=parameters
        )
        action_func.__doc__ = description

        if self.add_runnable_config:
            # Create __annotations__ so langchain can recognize the RunnableConfig
            action_func.__annotations__ = {"__runnable_config__": RunnableConfig}
        
        return action_func

    def wrap_tool(
        self, tool: Tool, execute_tool: AgenticProviderExecuteFn
    ) -> StructuredTool:
        """Wraps composio tool as Langchain StructuredTool object."""
        return t.cast(
            StructuredTool,
            StructuredTool.from_function(
                name=tool.slug,
                description=tool.description,
                args_schema=json_schema_to_model(
                    json_schema=tool.input_parameters,
                    skip_default=self.skip_default,
                ),
                return_schema=True,
                func=self._wrap_action(
                    tool=tool.slug,
                    description=tool.description,
                    schema_params=tool.input_parameters,
                    execute_tool=execute_tool,
                ),
                handle_tool_error=True,
                handle_validation_error=True,
            ),
        )

    def wrap_tools(
        self,
        tools: t.Sequence[Tool],
        execute_tool: AgenticProviderExecuteFn,
    ) -> t.List[StructuredTool]:
        """
        Get composio tools wrapped as Langchain StructuredTool objects.
        """
        return [self.wrap_tool(tool=tool, execute_tool=execute_tool) for tool in tools]
