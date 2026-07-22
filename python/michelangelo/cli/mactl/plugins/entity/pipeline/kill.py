"""Pipeline kill command implementation.

Sets the kill flag on a PipelineRun resource. Rejects kills on runs that
are not in PENDING or RUNNING state, mirroring the Go mactl behavior.
"""

from argparse import ArgumentParser
from inspect import Parameter, Signature
from logging import getLogger
from types import MethodType
from typing import Optional

from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf.message import Message
from grpc import Channel

from michelangelo.cli.mactl.crd import (
    CRD,
    CrdMethodInfo,
    bind_signature,
    crd_method_call,
    get_single_arg,
    inject_func_signature,
)

# Import TypedStruct to register it in the descriptor pool
from michelangelo.gen.api import typed_struct_pb2  # noqa: F401
from michelangelo.gen.api.v2 import pipeline_run_pb2

_LOG = getLogger(__name__)

_KILLABLE_STATES = frozenset(
    {
        pipeline_run_pb2.PIPELINE_RUN_STATE_PENDING,
        pipeline_run_pb2.PIPELINE_RUN_STATE_RUNNING,
    }
)


def add_function_signature(crd: CRD) -> None:
    """Add function signature for pipeline kill command."""
    inject_func_signature(
        crd,
        "kill",
        {
            "help": "Kill a running pipeline.",
            "args": [
                {
                    "func_signature": Parameter(
                        "namespace",
                        Parameter.POSITIONAL_OR_KEYWORD,
                    ),
                    "args": ["-n", "--namespace"],
                    "kwargs": {
                        "type": str,
                        "required": True,
                        "help": "Namespace of the resource",
                    },
                },
                {
                    "func_signature": Parameter(
                        "name",
                        Parameter.POSITIONAL_OR_KEYWORD,
                    ),
                    "args": ["--name"],
                    "kwargs": {
                        "type": str,
                        "required": True,
                        "help": "Name of the pipeline run resource",
                    },
                },
                {
                    "func_signature": Parameter(
                        "yes",
                        Parameter.POSITIONAL_OR_KEYWORD,
                        default=False,
                    ),
                    "args": ["--yes"],
                    "kwargs": {
                        "action": "store_true",
                        "help": (
                            "Automatic yes to prompts; assume 'yes' as answer to "
                            "all prompts and run non-interactively."
                        ),
                    },
                },
            ],
        },
    )


def generate_kill(crd: CRD, channel: Channel, parser: Optional[ArgumentParser] = None):
    """Generate kill function for pipeline CRD.

    Creates a kill command that sets the kill flag on a PipelineRun after
    verifying its state is PENDING or RUNNING.
    """
    _LOG.info("Generating `pipeline kill` for: %s", crd)

    crd.generate_get(channel)

    update_method_info = CrdMethodInfo(
        channel,
        crd.full_name,
        *crd._extract_method_info(channel, crd.full_name, "Update"),
    )

    crd.configure_parser("kill", parser)
    func_signature = crd._read_signatures("kill")

    @bind_signature(func_signature)
    def kill_func(bound_args: Signature) -> Message:
        _LOG.info("Start kill_func for pipeline")
        _LOG.info("Bound arguments: %r", bound_args.arguments)
        _self: CRD = bound_args.arguments["self"]
        _name = get_single_arg(bound_args.arguments, "name")
        _namespace = get_single_arg(bound_args.arguments, "namespace")
        _yes = bound_args.arguments.get("yes", False)

        if not _yes:
            confirmation = input(f" > kill pipeline run '{_name}'? [y/N] ")
            if confirmation.lower() not in ["y", "yes"]:
                print("Kill operation cancelled.")
                return None

        current_resource = _self.get(_namespace, _name)
        _LOG.info("Retrieved PipelineRun resource for kill: %r", current_resource)

        # State pre-check: mirror Go isPipelineRunKillable and error string.
        inner = getattr(current_resource, _self.name)
        state = inner.status.state
        if state not in _KILLABLE_STATES:
            state_name = pipeline_run_pb2.PipelineRunState.Name(state)
            raise ValueError(
                f"the pipelinerun cannot be killed because it's in {state_name} "
                "state. a pipelinerun can be killed if its state is either "
                "PENDING or RUNNING"
            )

        current_dict = MessageToDict(current_resource, preserving_proto_field_name=True)

        resource_name = _self.name
        if resource_name in current_dict and "spec" in current_dict[resource_name]:
            current_dict[resource_name]["spec"]["kill"] = True
        else:
            _LOG.error("Missing required spec field in the resource structure")
            raise ValueError(f"Cannot set kill flag on {resource_name}")

        request_input = update_method_info.input_class()
        ParseDict(current_dict, request_input, ignore_unknown_fields=True)

        _LOG.info(
            "KILL Request input (%r) ready: %r",
            type(request_input),
            request_input,
        )

        response = crd_method_call(update_method_info, request_input)

        response_dict = MessageToDict(response, preserving_proto_field_name=True)
        if (
            resource_name in response_dict
            and "spec" in response_dict[resource_name]
            and response_dict[resource_name]["spec"].get("kill") is True
        ):
            _LOG.info("Kill operation successfully set spec.kill=true")
        else:
            _LOG.error("Kill operation failed: spec.kill not set to true in response")
            raise RuntimeError(
                f"Kill operation failed for {resource_name}: spec.kill not properly set"
            )

        _LOG.info("Kill operation completed (%r): %r", type(response), response)
        return response

    kill_func.__signature__ = func_signature
    crd.kill = MethodType(kill_func, crd)
